import { randomUUID } from "node:crypto";

import { buildAnthropicPrompt, resolveAnthropicModel } from "./anthropic-prompt.js";
import { collectCompletionContent, streamCompletionContent } from "./openai-completion-runner.js";
import { createToolSieve, extractToolAwareOutput } from "./openai-tool-sieve.js";
import { parseToolCallsFromText } from "./openai-tool-parser.js";
import { ensureToolChoiceSatisfied } from "./openai-tool-policy.js";
import { stripLeakedMarkers } from "../utils/strip-markers.js";
import { log } from "../utils/log.js";

function createMessageId() {
  return `msg_${randomUUID()}`;
}

/* ── Anthropic SSE helpers ── */

function writeAnthropicSse(response, event, data) {
  response.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
}

/* ── Non-streaming (collect) ── */

export async function collectAnthropicMessage({
  account,
  body,
  deleteAfterFinish = false
}) {
  const model = resolveAnthropicModel(body?.model);
  const promptRequest = buildAnthropicPrompt({
    messages: body?.messages ?? [],
    system: body?.system,
    tools: body?.tools ?? [],
    toolChoice: body?.tool_choice
  });

  log.info("bridge", `[anthropic-collect] model=${model.id}, accountId=${account.id}`);

  const { content } = await collectCompletionContent({
    account,
    deleteAfterFinish,
    requestOptions: { model, prompt: promptRequest.prompt }
  });

  const parsed = promptRequest.toolNames.length
    ? extractToolAwareOutput(content, promptRequest.toolNames)
    : { content, toolCalls: [] };

  ensureToolChoiceSatisfied(promptRequest.toolChoicePolicy, parsed.toolCalls);

  const cleanContent = stripLeakedMarkers(parsed.content);

  const contentBlocks = [];
  if (cleanContent) {
    contentBlocks.push({ type: "text", text: cleanContent });
  }
  for (const call of (parsed.toolCalls ?? [])) {
    contentBlocks.push({
      type: "tool_use",
      id: call.id,
      name: call.name,
      input: call.input ?? {}
    });
  }

  return {
    id: createMessageId(),
    type: "message",
    role: "assistant",
    content: contentBlocks,
    model: model.id,
    stop_reason: parsed.toolCalls?.length ? "tool_use" : "end_turn",
    stop_sequence: null,
    usage: { input_tokens: 0, output_tokens: 0 }
  };
}

/* ── Streaming ── */

export async function streamAnthropicMessage(options) {
  const {
    account,
    body,
    deleteAfterFinish = false,
    response
  } = options;

  const model = resolveAnthropicModel(body?.model);
  const promptRequest = buildAnthropicPrompt({
    messages: body?.messages ?? [],
    system: body?.system,
    tools: body?.tools ?? [],
    toolChoice: body?.tool_choice
  });

  log.info("bridge", `[anthropic-stream] model=${model.id}, accountId=${account.id}, toolNames=[${promptRequest.toolNames.join(",")}]`);

  const messageId = createMessageId();
  const toolSieve = promptRequest.toolNames.length
    ? createToolSieve(promptRequest.toolNames)
    : null;

  let contentIndex = 0;
  let currentBlockType = null;  // "thinking" | "text" | "tool_use"
  let sawToolCall = false;
  const inputTokens = body?.messages?.reduce((s, m) => s + JSON.stringify(m).length, 0) ?? 0;

  response.writeHead(200, {
    "cache-control": "no-cache, no-transform",
    connection: "keep-alive",
    "content-type": "text/event-stream; charset=utf-8",
    "x-accel-buffering": "no"
  });
  response.flushHeaders?.();

  // message_start
  writeAnthropicSse(response, "message_start", {
    type: "message_start",
    message: {
      id: messageId,
      type: "message",
      role: "assistant",
      content: [],
      model: model.id,
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: Math.round(inputTokens / 4), output_tokens: 0 }
    }
  });

  /* ── Block transition helpers ── */

  const startBlock = (type, extra = {}) => {
    if (currentBlockType === type) return; // already in this block type
    if (currentBlockType) {
      writeAnthropicSse(response, "content_block_stop", { type: "content_block_stop", index: contentIndex - 1 });
    }
    const block = { type: "content_block_start", index: contentIndex, content_block: { type, ...extra } };
    writeAnthropicSse(response, "content_block_start", block);
    currentBlockType = type;
    contentIndex++;
  };

  const emitDelta = (deltaType, field, text) => {
    writeAnthropicSse(response, "content_block_delta", {
      type: "content_block_delta",
      index: contentIndex - 1,
      delta: { type: deltaType, [field]: text }
    });
  };

  const finishCurrentBlock = () => {
    if (currentBlockType) {
      writeAnthropicSse(response, "content_block_stop", { type: "content_block_stop", index: contentIndex - 1 });
      currentBlockType = null;
    }
  };

  /* ── Tool call emission ── */

  const emitToolCalls = (calls) => {
    if (!calls.length) return;
    sawToolCall = true;
    finishCurrentBlock();

    for (const call of calls) {
      startBlock("tool_use", { id: call.id, name: call.name, input: {} });
      emitDelta("input_json_delta", "partial_json", call.argumentsText);
      finishCurrentBlock();
    }
  };

  /* ── Text emission ── */

  const emitTextEvent = (text, kind) => {
    const cleaned = stripLeakedMarkers(text);
    if (!cleaned) return;

    if (kind === "thinking") {
      startBlock("thinking", { thinking: "" });
      emitDelta("thinking_delta", "thinking", cleaned);
    } else {
      startBlock("text", { text: "" });
      emitDelta("text_delta", "text", cleaned);
    }
  };

  /* ── Sieve event dispatch ── */

  const emitSieveEvents = (events, fallbackKind) => {
    for (const event of events) {
      if (event.type === "tool_calls") {
        emitToolCalls(event.calls ?? []);
      } else if (event.type === "text") {
        emitTextEvent(event.text, event.kind ?? fallbackKind);
      }
    }
  };

  /* ── Stream from DeepSeek ── */

  await streamCompletionContent({
    account,
    deleteAfterFinish,
    onText: (delta, kind) => {
      if (toolSieve) {
        emitSieveEvents(toolSieve.push(delta, kind), kind);
        return;
      }
      emitTextEvent(delta, kind);
    },
    requestOptions: { model, prompt: promptRequest.prompt }
  });

  if (toolSieve) {
    emitSieveEvents(toolSieve.flush());
  }

  finishCurrentBlock();

  /* ── Last-ditch tool call detection ── */

  if (promptRequest.toolNames.length && !sawToolCall && toolSieve) {
    const emittedText = toolSieve.emittedText;
    if (emittedText.length > 0) {
      const fallbackCalls = parseToolCallsFromText(emittedText, promptRequest.toolNames);
      if (fallbackCalls.length) {
        log.info("bridge", `[anthropic-stream] Fallback parse found ${fallbackCalls.length} tool call(s)`);
        emitToolCalls(fallbackCalls);
      } else {
        const thinkMatch = emittedText.match(/<think>([\s\S]*?)<\/think>/);
        const thinkLen = thinkMatch ? thinkMatch[1].length : 0;
        const respText = emittedText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        const head = respText.slice(0, 300);
        const tail = respText.length > 600 ? respText.slice(-200) : "";
        const summary = tail ? `${head}\n...\n${tail}` : head;
        log.debug("bridge", `[anthropic-stream] No tool calls (allowed: [${promptRequest.toolNames.join(",")}]). ${emittedText.length} chars total (think: ${thinkLen}, response: ${emittedText.length - thinkLen}).\n${summary}`);
      }
    }
  }

  /* ── message_delta + message_stop ── */

  const stopReason = sawToolCall ? "tool_use" : "end_turn";

  writeAnthropicSse(response, "message_delta", {
    type: "message_delta",
    delta: { stop_reason: stopReason, stop_sequence: null },
    usage: { output_tokens: 0 }
  });

  writeAnthropicSse(response, "message_stop", { type: "message_stop" });

  response.end();
}

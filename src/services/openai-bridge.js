import { randomUUID } from "node:crypto";

import { collectCompletionContent, streamCompletionContent } from "./openai-completion-runner.js";
import { assertNoLegacySearchOptions, resolveOpenAiModel } from "./openai-request.js";
import { createToolSieve, extractToolAwareOutput } from "./openai-tool-sieve.js";
import { parseToolCallsFromText } from "./openai-tool-parser.js";
import { buildOpenAiPrompt } from "./openai-tool-prompt.js";
import { ensureToolChoiceSatisfied, hasChatToolingRequest } from "./openai-tool-policy.js";
import { createOpenAiError } from "./openai-error.js";
import { stripLeakedMarkers } from "../utils/strip-markers.js";
import { log } from "../utils/log.js";

function createCompletionId() {
  return `chatcmpl_${randomUUID()}`;
}

function createChatToolCalls(calls, startIndex = 0) {
  return calls.map((call, offset) => ({
    index: startIndex + offset,
    id: call.id,
    type: "function",
    function: {
      name: call.name,
      arguments: call.argumentsText
    }
  }));
}

function resolveCompletionRequest(body, toolCallsEnabled) {
  assertNoLegacySearchOptions(body);

  if (!toolCallsEnabled && hasChatToolingRequest(body)) {
    log.warn("bridge", "Tool calls rejected: toolCallsEnabled=false but request has tools/tool_choice/tool history");
    throw createOpenAiError(400, "Tool calls are disabled for this API key");
  }

  const model = resolveOpenAiModel(body?.model);
  const toolNames = (body?.tools ?? []).map(t => t?.function?.name ?? t?.name).filter(Boolean);
  log.debug("bridge", `Model: ${model.id}, toolCallsEnabled: ${toolCallsEnabled}, tools: [${toolNames.join(",")}]`);

  const promptRequest = buildOpenAiPrompt({
    messages: body?.messages ?? [],
    toolChoice: toolCallsEnabled ? body?.tool_choice : undefined,
    tools: toolCallsEnabled ? body?.tools ?? [] : []
  });

  return {
    model,
    prompt: promptRequest.prompt,
    toolChoicePolicy: promptRequest.toolChoicePolicy,
    toolNames: promptRequest.toolNames
  };
}

function buildChatCompletionPayload(completionId, requestOptions, content) {
  const parsed = requestOptions.toolNames.length
    ? extractToolAwareOutput(content, requestOptions.toolNames)
    : { content, toolCalls: [] };
  parsed.content = stripLeakedMarkers(parsed.content);

  if (parsed.toolCalls.length) {
    log.debug("bridge", `[collect] Parsed ${parsed.toolCalls.length} tool call(s)`);
  } else if (requestOptions.toolNames.length) {
    const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
    const thinkLen = thinkMatch ? thinkMatch[1].length : 0;
    const respText = content.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
    const head = respText.slice(0, 300);
    const tail = respText.length > 600 ? respText.slice(-200) : "";
    const summary = tail ? `${head}\n...\n${tail}` : head;
    log.debug("bridge", `[collect] No tool calls (allowed: [${requestOptions.toolNames.join(",")}]). ${content.length} chars total (think: ${thinkLen}, response: ${content.length - thinkLen}).\n${summary}`);
  }

  ensureToolChoiceSatisfied(requestOptions.toolChoicePolicy, parsed.toolCalls);

  if (parsed.toolCalls.length) {
    return {
      id: completionId,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model: requestOptions.model.id,
      choices: [
        {
          index: 0,
          finish_reason: "tool_calls",
          message: {
            role: "assistant",
            content: parsed.content.length ? parsed.content : null,
            tool_calls: createChatToolCalls(parsed.toolCalls)
          }
        }
      ]
    };
  }

  return {
    id: completionId,
    object: "chat.completion",
    created: Math.floor(Date.now() / 1000),
    model: requestOptions.model.id,
    choices: [
      {
        index: 0,
        finish_reason: "stop",
        message: {
          role: "assistant",
          content: parsed.content
        }
      }
    ]
  };
}

function buildChunkPayload(completionId, model, delta, finishReason) {
  const choice = finishReason
    ? { index: 0, delta: {}, finish_reason: finishReason }
    : { index: 0, delta };

  return {
    id: completionId,
    object: "chat.completion.chunk",
    created: Math.floor(Date.now() / 1000),
    model,
    choices: [choice]
  };
}

function writeSseChunk(response, payload) {
  response.write(`data: ${JSON.stringify(payload)}\n\n`);
}

export async function collectOpenAiResponse({
  account,
  body,
  deleteAfterFinish = false,
  toolCallsEnabled = false
}) {
  const requestOptions = resolveCompletionRequest(body, toolCallsEnabled);
  log.info("bridge", `[collect] model=${requestOptions.model.id}, accountId=${account.id}`);
  const { content } = await collectCompletionContent({
    account,
    deleteAfterFinish,
    requestOptions
  });

  return buildChatCompletionPayload(createCompletionId(), requestOptions, content);
}

export async function streamOpenAiResponse(options) {
  const {
    account,
    body,
    deleteAfterFinish = false,
    response,
    toolCallsEnabled = false
  } = options;
  const completionId = createCompletionId();
  const requestOptions = resolveCompletionRequest(body, toolCallsEnabled);
  log.info("bridge", `[stream] model=${requestOptions.model.id}, accountId=${account.id}, toolNames=[${requestOptions.toolNames.join(",")}]`);
  const toolSieve = requestOptions.toolNames.length
    ? createToolSieve(requestOptions.toolNames)
    : null;
  let toolCallIndex = 0;
  let sawToolCall = false;

  response.writeHead(200, {
    "cache-control": "no-cache, no-transform",
    connection: "keep-alive",
    "content-type": "text/event-stream; charset=utf-8",
    "x-accel-buffering": "no"
  });
  response.flushHeaders?.();

  writeSseChunk(response, buildChunkPayload(
    completionId,
    requestOptions.model.id,
    { role: "assistant" }
  ));

  const emitToolCalls = (calls) => {
    if (!calls.length) return;
    sawToolCall = true;
    writeSseChunk(response, buildChunkPayload(
      completionId,
      requestOptions.model.id,
      { tool_calls: createChatToolCalls(calls, toolCallIndex) }
    ));
    toolCallIndex += calls.length;
  };

  const emitTextEvent = (text, kind) => {
    const cleaned = stripLeakedMarkers(text);
    if (!cleaned) return;
    const delta = kind === "thinking"
      ? { reasoning_content: cleaned }
      : { content: cleaned };
    writeSseChunk(response, buildChunkPayload(completionId, requestOptions.model.id, delta));
  };

  const emitSieveEvents = (events, fallbackKind) => {
    for (const event of events) {
      if (event.type === "tool_calls") {
        emitToolCalls(event.calls ?? []);
      } else if (event.type === "text") {
        emitTextEvent(event.text, event.kind ?? fallbackKind);
      }
    }
  };

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
    requestOptions
  });

  if (toolSieve) {
    emitSieveEvents(toolSieve.flush());
  }

  writeSseChunk(response, buildChunkPayload(
    completionId,
    requestOptions.model.id,
    {},
    sawToolCall ? "tool_calls" : "stop"
  ));

  if (requestOptions.toolNames.length && !sawToolCall && toolSieve) {
    /* Last-ditch: the streaming sieve already tried hard to find tool calls
     * via standard XML tags; if it found none, the parser's broader
     * strategies (raw patch detection, etc.) might still pick something up
     * from the full emitted text.  This pays for itself only when the model
     * skipped XML entirely, which is rare. */
    const emittedText = toolSieve.emittedText;
    if (emittedText.length > 0) {
      const fallbackCalls = parseToolCallsFromText(emittedText, requestOptions.toolNames);
      if (fallbackCalls.length) {
        log.info("bridge", `[stream] Fallback parse found ${fallbackCalls.length} tool call(s) the sieve missed`);
        emitToolCalls(fallbackCalls);
      } else {
        const thinkMatch = emittedText.match(/<think>([\s\S]*?)<\/think>/);
        const thinkLen = thinkMatch ? thinkMatch[1].length : 0;
        const respText = emittedText.replace(/<think>[\s\S]*?<\/think>/g, '').trim();
        const head = respText.slice(0, 300);
        const tail = respText.length > 600 ? respText.slice(-200) : "";
        const summary = tail ? `${head}\n...\n${tail}` : head;
        log.debug("bridge", `[stream] No tool calls (allowed: [${requestOptions.toolNames.join(",")}]). ${emittedText.length} chars total (think: ${thinkLen}, response: ${emittedText.length - thinkLen}).\n${summary}`);
      }
    }
  }

  response.end("data: [DONE]\n\n");
}

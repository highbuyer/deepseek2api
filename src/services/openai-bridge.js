import { randomUUID } from "node:crypto";

import { collectCompletionContent, streamCompletionContent } from "./openai-completion-runner.js";
import { assertNoLegacySearchOptions, resolveOpenAiModel } from "./openai-request.js";
import { createToolSieve, extractToolAwareOutput } from "./openai-tool-sieve.js";
import { buildOpenAiPrompt } from "./openai-tool-prompt.js";
import { ensureToolChoiceSatisfied, hasChatToolingRequest } from "./openai-tool-policy.js";
import { createOpenAiError } from "./openai-error.js";
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
  log.debug("bridge", `Model: ${model.id}, toolCallsEnabled: ${toolCallsEnabled}, tools: [${toolNames.join(",")}], tool_choice: ${JSON.stringify(body?.tool_choice)}`);

  const promptRequest = buildOpenAiPrompt({
    messages: body?.messages ?? [],
    toolChoice: toolCallsEnabled ? body?.tool_choice : undefined,
    tools: toolCallsEnabled ? body?.tools ?? [] : []
  });

  log.debug("bridge", `Prompt length: ${promptRequest.prompt.length} chars, toolChoicePolicy.mode: ${promptRequest.toolChoicePolicy.mode}, allowedToolNames: [${promptRequest.toolNames.join(",")}]`);

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

  log.debug("bridge", `[collect] Content length: ${content.length}, parsed toolCalls: ${parsed.toolCalls.length}, parsed content length: ${parsed.content.length}`);
  if (parsed.toolCalls.length) {
    log.debug("bridge", `Parsed tool calls:`, parsed.toolCalls.map(c => ({ name: c.name, argsLen: c.argumentsText.length })));
  } else if (requestOptions.toolNames.length) {
    log.warn("bridge", `No tool calls parsed from model output (expected tools: [${requestOptions.toolNames.join(",")}]). Raw content (first 500 chars): ${content.slice(0, 500)}`);
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
  log.info("bridge", `[collect] model=${requestOptions.model.id}, stream=false, accountId=${account.id}, deleteAfter=${deleteAfterFinish}`);
  const { content } = await collectCompletionContent({
    account,
    deleteAfterFinish,
    requestOptions
  });
  log.debug("bridge", `[collect] Raw content length: ${content.length}`);

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
  log.info("bridge", `[stream] model=${requestOptions.model.id}, stream=true, accountId=${account.id}, deleteAfter=${deleteAfterFinish}, toolNames=[${requestOptions.toolNames.join(",")}]`);
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
    if (!calls.length) {
      return;
    }

    sawToolCall = true;
    log.debug("bridge", `[stream] Tool calls detected: ${calls.map(c => c.name).join(",")}`);
    writeSseChunk(response, buildChunkPayload(
      completionId,
      requestOptions.model.id,
      { tool_calls: createChatToolCalls(calls, toolCallIndex) }
    ));
    toolCallIndex += calls.length;
  };

  await streamCompletionContent({
    account,
    deleteAfterFinish,
    onText: (delta) => {
      if (!toolSieve) {
        writeSseChunk(response, buildChunkPayload(
          completionId,
          requestOptions.model.id,
          { content: delta }
        ));
        return;
      }

      const events = toolSieve.push(delta);
      events.forEach((event) => {
        if (event.type === "tool_calls") {
          emitToolCalls(event.calls ?? []);
          return;
        }

        if (event.text) {
          writeSseChunk(response, buildChunkPayload(
            completionId,
            requestOptions.model.id,
            { content: event.text }
          ));
        }
      });
    },
    requestOptions
  });

  if (toolSieve) {
    const tailEvents = toolSieve.flush();
    tailEvents.forEach((event) => {
      if (event.type === "tool_calls") {
        emitToolCalls(event.calls ?? []);
        return;
      }

      if (event.text) {
        writeSseChunk(response, buildChunkPayload(
          completionId,
          requestOptions.model.id,
          { content: event.text }
        ));
      }
    });
  }

  writeSseChunk(response, buildChunkPayload(
    completionId,
    requestOptions.model.id,
    {},
    sawToolCall ? "tool_calls" : "stop"
  ));

  if (requestOptions.toolNames.length && !sawToolCall) {
    log.warn("bridge", `[stream] No tool calls detected in stream (expected tools: [${requestOptions.toolNames.join(",")}])`);
  } else if (sawToolCall) {
    log.info("bridge", `[stream] Completed with ${toolCallIndex} tool call(s)`);
  }

  response.end("data: [DONE]\n\n");
}

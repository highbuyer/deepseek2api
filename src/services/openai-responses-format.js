import { randomUUID } from "node:crypto";

import { createOpenAiError } from "./openai-error.js";
import { assertNoLegacySearchOptions, resolveOpenAiModel } from "./openai-request.js";
import { extractToolAwareOutput } from "./openai-tool-sieve.js";
import { buildOpenAiPrompt } from "./openai-tool-prompt.js";
import { ensureToolChoiceSatisfied, hasChatToolingRequest } from "./openai-tool-policy.js";
import { toStringSafe, toJsonText } from "../utils/safe-string.js";
import { stripLeakedMarkers } from "../utils/strip-markers.js";

export function createResponseId() {
  return `resp_${randomUUID().replaceAll("-", "")}`;
}

export function createOutputItemId(prefix) {
  return `${prefix}_${randomUUID().replaceAll("-", "")}`;
}

function normalizeResponsesContent(content) {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((item) => toStringSafe(item?.text ?? item?.content ?? item?.output_text))
    .filter(Boolean)
    .join("\n");
}

function normalizeResponsesInputItem(item) {
  if (!item || typeof item !== "object") {
    return null;
  }

  const type = toStringSafe(item.type).trim();
  if (type === "message") {
    return {
      content: normalizeResponsesContent(item.content ?? item.text),
      role: toStringSafe(item.role).trim().toLowerCase() || "user"
    };
  }

  if (type === "function_call") {
    return {
      content: "",
      role: "assistant",
      tool_calls: [
        {
          id: toStringSafe(item.call_id).trim(),
          type: "function",
          function: {
            name: toStringSafe(item.name).trim(),
            arguments: toJsonText(item.arguments)
          }
        }
      ]
    };
  }

  if (type === "function_call_output") {
    return {
      content: typeof item.output === "string" ? item.output : JSON.stringify(item.output ?? null),
      role: "tool",
      tool_call_id: toStringSafe(item.call_id).trim()
    };
  }

  if (typeof item.text === "string" || typeof item.content === "string") {
    return { content: normalizeResponsesContent(item.content ?? item.text), role: "user" };
  }

  return null;
}

function resolveResponsesMessages(body) {
  if (Array.isArray(body?.messages) && body.messages.length) {
    return body.messages;
  }

  if (typeof body?.input === "string") {
    return [{ role: "user", content: body.input }];
  }

  if (Array.isArray(body?.input)) {
    return body.input.map(normalizeResponsesInputItem).filter(Boolean);
  }

  if (body?.input && typeof body.input === "object") {
    const normalized = normalizeResponsesInputItem(body.input);
    return normalized ? [normalized] : [];
  }

  return [];
}

function hasResponsesToolingRequest(body) {
  return Boolean(
    hasChatToolingRequest(body)
    || Array.isArray(body?.input) && body.input.some((item) => {
      const type = toStringSafe(item?.type).trim();
      return type === "function_call" || type === "function_call_output";
    })
  );
}

export function resolveResponsesRequest(body, toolCallsEnabled) {
  assertNoLegacySearchOptions(body);

  if (body?.previous_response_id) {
    throw createOpenAiError(400, "previous_response_id is not supported");
  }

  if (!toolCallsEnabled && hasResponsesToolingRequest(body)) {
    throw createOpenAiError(400, "Tool calls are disabled for this API key");
  }

  const messages = resolveResponsesMessages(body);
  if (!messages.length) {
    throw createOpenAiError(400, "Responses request requires input or messages");
  }

  const instructions = toStringSafe(body?.instructions).trim();
  const promptRequest = buildOpenAiPrompt({
    messages: instructions ? [{ role: "system", content: instructions }, ...messages] : messages,
    toolChoice: toolCallsEnabled ? body?.tool_choice : undefined,
    tools: toolCallsEnabled ? body?.tools ?? [] : []
  });

  return {
    model: resolveOpenAiModel(body?.model),
    prompt: promptRequest.prompt,
    stream: body?.stream === true,
    toolChoicePolicy: promptRequest.toolChoicePolicy,
    toolNames: promptRequest.toolNames
  };
}

export function createFunctionCallItem(call) {
  return {
    id: createOutputItemId("fc"),
    type: "function_call",
    call_id: call.id,
    name: call.name,
    arguments: call.argumentsText,
    status: "completed"
  };
}

function createMessageOutputItem(text) {
  return {
    type: "message",
    id: createOutputItemId("msg"),
    role: "assistant",
    content: text.length ? [{ type: "output_text", text }] : []
  };
}

function stripThinkMarkers(text) {
  return stripLeakedMarkers(text);
}

function buildResponseOutputsFromEvents(events) {
  const output = [];
  const toolCalls = [];

  events.forEach((event) => {
    if (event.type === "tool_calls") {
      const calls = event.calls ?? [];
      toolCalls.push(...calls);
      output.push(...calls.map(createFunctionCallItem));
      return;
    }

    const cleanText = stripThinkMarkers(event.text);
    if (cleanText.length && event.kind !== "thinking") {
      output.push(createMessageOutputItem(cleanText));
    }
  });

  return { output, toolCalls };
}

function resolveResponseOutput({ content, outputItems, requestOptions, toolCalls }) {
  if (outputItems?.length) {
    return { output: outputItems, outputText: content, toolCalls: toolCalls ?? [] };
  }

  if (!requestOptions.toolNames.length) {
    const cleanContent = stripThinkMarkers(content);
    return {
      output: [createMessageOutputItem(cleanContent)],
      outputText: cleanContent,
      toolCalls: []
    };
  }

  const parsed = extractToolAwareOutput(content, requestOptions.toolNames);
  const built = buildResponseOutputsFromEvents(parsed.events);
  return {
    output: built.output.length || built.toolCalls.length
      ? built.output
      : [createMessageOutputItem(parsed.content)],
    outputText: parsed.content,
    toolCalls: built.toolCalls
  };
}

export function buildResponseObject({ responseId, requestOptions, content, outputItems, toolCalls }) {
  const resolved = resolveResponseOutput({
    content,
    outputItems,
    requestOptions,
    toolCalls
  });

  ensureToolChoiceSatisfied(requestOptions.toolChoicePolicy, resolved.toolCalls);

  return {
    id: responseId,
    type: "response",
    object: "response",
    created_at: Math.floor(Date.now() / 1000),
    status: "completed",
    model: requestOptions.model.id,
    output: resolved.output,
    output_text: resolved.outputText
  };
}

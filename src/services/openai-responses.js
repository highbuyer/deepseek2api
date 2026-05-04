import { collectCompletionContent, streamCompletionContent } from "./openai-completion-runner.js";
import { getOpenAiResponse, storeOpenAiResponse } from "./openai-response-store.js";
import { createToolSieve, FORMAT_ERROR_MSG } from "./openai-tool-sieve.js";
import { ensureToolChoiceSatisfied } from "./openai-tool-policy.js";
import { stripLeakedMarkers } from "../utils/strip-markers.js";
import {
  buildResponseObject,
  createFunctionCallItem,
  createOutputItemId,
  createResponseId,
  resolveResponsesRequest
} from "./openai-responses-format.js";

function writeResponsesEvent(response, event, payload) {
  response.write(`event: ${event}\n`);
  response.write(`data: ${JSON.stringify(payload)}\n\n`);
}

export function getStoredOpenAiResponse(scope, responseId) {
  return getOpenAiResponse(scope, responseId);
}

export async function collectResponsesResponse({
  account,
  body,
  responseScope,
  deleteAfterFinish = false,
  toolCallsEnabled = false
}) {
  const requestOptions = resolveResponsesRequest(body, toolCallsEnabled);
  const { content } = await collectCompletionContent({
    account,
    deleteAfterFinish,
    requestOptions
  });
  const responseId = createResponseId();
  const payload = buildResponseObject({ responseId, requestOptions, content });
  storeOpenAiResponse(responseScope, responseId, payload);
  return payload;
}

export async function streamResponsesResponse({
  account,
  body,
  response,
  responseScope,
  deleteAfterFinish = false,
  toolCallsEnabled = false
}) {
  const requestOptions = resolveResponsesRequest(body, toolCallsEnabled);
  const responseId = createResponseId();
  const toolSieve = requestOptions.toolNames.length ? createToolSieve(requestOptions.toolNames) : null;
  let nextOutputIndex = 0;
  let outputText = "";
  let activeTextItem = null;
  const outputItems = [];
  const toolCalls = [];

  response.writeHead(200, {
    "cache-control": "no-cache, no-transform",
    connection: "keep-alive",
    "content-type": "text/event-stream; charset=utf-8",
    "x-accel-buffering": "no"
  });
  response.flushHeaders?.();

  writeResponsesEvent(response, "response.created", {
    type: "response.created",
    response: {
      id: responseId,
      type: "response",
      object: "response",
      created_at: Math.floor(Date.now() / 1000),
      status: "in_progress",
      model: requestOptions.model.id,
      output: []
    }
  });

  const openTextOutput = () => {
    if (activeTextItem) {
      return;
    }

    activeTextItem = {
      id: createOutputItemId("msg"),
      outputIndex: nextOutputIndex++,
      text: ""
    };
    writeResponsesEvent(response, "response.output_item.added", {
      type: "response.output_item.added",
      response_id: responseId,
      output_index: activeTextItem.outputIndex,
      item: {
        type: "message",
        id: activeTextItem.id,
        role: "assistant",
        content: []
      }
    });
  };

  const emitText = (text, kind, skipStrip = false) => {
    const cleaned = skipStrip ? text : stripLeakedMarkers(text);
    if (!cleaned) {
      return;
    }

    openTextOutput();
    activeTextItem.text += cleaned;
    outputText += cleaned;
    writeResponsesEvent(response, "response.output_text.delta", {
      type: "response.output_text.delta",
      response_id: responseId,
      item_id: activeTextItem.id,
      output_index: activeTextItem.outputIndex,
      content_index: 0,
      delta: cleaned
    });
  };

  const closeTextOutput = () => {
    if (!activeTextItem) {
      return;
    }

    const item = {
      type: "message",
      id: activeTextItem.id,
      role: "assistant",
      content: activeTextItem.text.length
        ? [{ type: "output_text", text: activeTextItem.text }]
        : []
    };
    outputItems[activeTextItem.outputIndex] = item;
    writeResponsesEvent(response, "response.output_item.done", {
      type: "response.output_item.done",
      response_id: responseId,
      output_index: activeTextItem.outputIndex,
      item
    });
    activeTextItem = null;
  };

  const emitToolCalls = (calls) => {
    closeTextOutput();
    calls.forEach((call) => {
      toolCalls.push(call);
      const item = createFunctionCallItem(call);
      outputItems[nextOutputIndex] = item;
      writeResponsesEvent(response, "response.output_item.added", {
        type: "response.output_item.added",
        response_id: responseId,
        output_index: nextOutputIndex,
        item
      });
      writeResponsesEvent(response, "response.function_call_arguments.done", {
        type: "response.function_call_arguments.done",
        response_id: responseId,
        output_index: nextOutputIndex,
        item_id: item.id,
        call_id: item.call_id,
        name: item.name,
        arguments: item.arguments
      });
      writeResponsesEvent(response, "response.output_item.done", {
        type: "response.output_item.done",
        response_id: responseId,
        output_index: nextOutputIndex,
        item
      });
      nextOutputIndex += 1;
    });
  };

  await streamCompletionContent({
    account,
    deleteAfterFinish,
    onText: (delta, kind) => {
      if (!toolSieve) {
        emitText(delta, kind);
        return;
      }

      toolSieve.push(delta, kind).forEach((event) => {
        if (event.type === "tool_calls") {
          emitToolCalls(event.calls ?? []);
          return;
        }
        if (event.type === "format_error") {
          emitText(FORMAT_ERROR_MSG, "response", true);
          return;
        }
        emitText(event.text, event.kind ?? kind);
      });
    },
    requestOptions
  });

  if (toolSieve) {
    toolSieve.flush().forEach((event) => {
      if (event.type === "tool_calls") {
        emitToolCalls(event.calls ?? []);
        return;
      }
      if (event.type === "format_error") {
        emitText(FORMAT_ERROR_MSG, "response", true);
        return;
      }
      emitText(event.text, event.kind);
    });
  }

  try {
    ensureToolChoiceSatisfied(requestOptions.toolChoicePolicy, toolCalls);
  } catch (error) {
    writeResponsesEvent(response, "response.failed", {
      type: "response.failed",
      response: {
        id: responseId,
        type: "response",
        object: "response",
        created_at: Math.floor(Date.now() / 1000),
        status: "failed",
        model: requestOptions.model.id
      },
      error: {
        code: error.code || "invalid_request_error",
        message: error.message
      }
    });
    response.end("data: [DONE]\n\n");
    return;
  }

  closeTextOutput();
  const payload = buildResponseObject({
    responseId,
    requestOptions,
    content: outputText,
    outputItems: outputItems.filter(Boolean),
    toolCalls
  });
  storeOpenAiResponse(responseScope, responseId, payload);
  writeResponsesEvent(response, "response.completed", {
    type: "response.completed",
    response: payload
  });
  response.end("data: [DONE]\n\n");
}

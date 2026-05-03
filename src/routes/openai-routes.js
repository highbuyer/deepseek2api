import { getApiKeyRecord } from "../services/api-key-service.js";
import { takeRoundRobinAccount } from "../services/account-rotation-service.js";
import { isIncognitoEnabledForOwner } from "../services/incognito-service.js";
import { collectOpenAiResponse, streamOpenAiResponse } from "../services/openai-bridge.js";
import { collectAnthropicMessage, streamAnthropicMessage } from "../services/anthropic-bridge.js";
import { listOpenAiModels } from "../services/openai-request.js";
import { createEmbeddings } from "../services/openai-embeddings.js";
import {
  collectResponsesResponse,
  getStoredOpenAiResponse,
  streamResponsesResponse
} from "../services/openai-responses.js";
import { withOwnerRequestLimit } from "../services/request-limit-service.js";
import { parseJsonBody, readRequestBody, sendError, sendJson } from "../utils/http.js";
import { log } from "../utils/log.js";

function getBearerToken(request) {
  const value = request.headers.authorization ?? "";
  return value.startsWith("Bearer ") ? value.slice(7) : "";
}

function resolveLimitStatus(error) {
  return error.code === "USER_DISABLED" ? 403 : 429;
}

function handleOpenAiError(response, error) {
  if (error.code === "USER_DISABLED" || error.code === "REQUEST_LIMIT") {
    log.warn("route", `Request rejected: code=${error.code} msg=${error.message}`);
    sendError(response, resolveLimitStatus(error), error.message);
    return true;
  }

  if (error instanceof SyntaxError) {
    log.warn("route", "Invalid JSON body");
    sendError(response, 400, "Invalid JSON body");
    return true;
  }

  if (error.statusCode) {
    log.warn("route", `Error ${error.statusCode}: ${error.message}`);
    sendError(response, error.statusCode, error.message);
    return true;
  }

  return false;
}

async function handleModelsRequest(response, apiKeyRecord) {
  await withOwnerRequestLimit(apiKeyRecord.ownerId, async () => {
    sendJson(response, 200, {
      object: "list",
      data: listOpenAiModels()
    });
  });
}

async function handleChatCompletionsRequest(request, response, apiKeyRecord) {
  await withOwnerRequestLimit(apiKeyRecord.ownerId, async () => {
    const body = parseJsonBody(await readRequestBody(request)) ?? {};
    const account = takeRoundRobinAccount(apiKeyRecord);
    if (!account) {
      log.warn("route", "No available account for chat completions");
      sendError(response, 404, "Account not found");
      return;
    }

    const deleteAfterFinish = isIncognitoEnabledForOwner(apiKeyRecord.ownerId);
    log.info("route", `[chat] stream=${!!body.stream} model=${body.model || "default"} toolCalls=${apiKeyRecord.toolCallsEnabled} accountId=${account.id}`);
    if (body.stream) {
      await streamOpenAiResponse({
        response,
        account,
        body,
        deleteAfterFinish,
        toolCallsEnabled: apiKeyRecord.toolCallsEnabled
      });
      return;
    }

    const payload = await collectOpenAiResponse({
      account,
      body,
      deleteAfterFinish,
      toolCallsEnabled: apiKeyRecord.toolCallsEnabled
    });
    sendJson(response, 200, payload);
  });
}

async function handleResponsesCreateRequest(request, response, apiKeyRecord) {
  await withOwnerRequestLimit(apiKeyRecord.ownerId, async () => {
    const body = parseJsonBody(await readRequestBody(request)) ?? {};
    const account = takeRoundRobinAccount(apiKeyRecord);
    if (!account) {
      log.warn("route", "No available account for responses");
      sendError(response, 404, "Account not found");
      return;
    }

    const deleteAfterFinish = isIncognitoEnabledForOwner(apiKeyRecord.ownerId);
    log.info("route", `[responses] stream=${!!body.stream} model=${body.model || "default"} toolCalls=${apiKeyRecord.toolCallsEnabled} accountId=${account.id}`);
    // Debug: dump input structure to see if file content is present
    if (Array.isArray(body.input)) {
      body.input.slice(0, 2).forEach((item, i) => {
        const content = item?.content;
        if (Array.isArray(content)) {
          const types = content.map(c => c?.type || typeof c).join(",");
          const preview = JSON.stringify(content).slice(0, 500);
          log.debug("route", `[responses] input[${i}] content types=[${types}] preview=${preview}`);
        } else if (typeof content === "string") {
          log.debug("route", `[responses] input[${i}] content string len=${content.length} preview=${content.slice(0, 300)}`);
        }
      });
    }
    if (body.stream) {
      await streamResponsesResponse({
        response,
        account,
        body,
        deleteAfterFinish,
        responseScope: apiKeyRecord.id,
        toolCallsEnabled: apiKeyRecord.toolCallsEnabled
      });
      return;
    }

    const payload = await collectResponsesResponse({
      account,
      body,
      deleteAfterFinish,
      responseScope: apiKeyRecord.id,
      toolCallsEnabled: apiKeyRecord.toolCallsEnabled
    });
    sendJson(response, 200, payload);
  });
}

async function handleEmbeddingsRequest(request, response, apiKeyRecord) {
  await withOwnerRequestLimit(apiKeyRecord.ownerId, async () => {
    const body = parseJsonBody(await readRequestBody(request)) ?? {};
    log.info("route", `[embeddings] model=${body.model || "default"}`);
    const payload = await createEmbeddings({ body });
    sendJson(response, 200, payload);
  });
}

async function handleMessagesRequest(request, response, apiKeyRecord) {
  await withOwnerRequestLimit(apiKeyRecord.ownerId, async () => {
    const body = parseJsonBody(await readRequestBody(request)) ?? {};
    const account = takeRoundRobinAccount(apiKeyRecord);
    if (!account) {
      log.warn("route", "No available account for messages");
      sendError(response, 404, "Account not found");
      return;
    }

    const deleteAfterFinish = isIncognitoEnabledForOwner(apiKeyRecord.ownerId);
    log.info("route", `[messages] stream=${!!body.stream} model=${body.model || "default"} accountId=${account.id}`);

    if (body.stream !== false) {
      await streamAnthropicMessage({ response, account, body, deleteAfterFinish });
      return;
    }

    const payload = await collectAnthropicMessage({ account, body, deleteAfterFinish });
    sendJson(response, 200, payload);
  });
}

async function handleResponsesGetRequest(response, apiKeyRecord, url) {
  await withOwnerRequestLimit(apiKeyRecord.ownerId, async () => {
    const responseId = url.pathname.split("/").pop();
    const payload = getStoredOpenAiResponse(apiKeyRecord.id, responseId);
    if (!payload) {
      sendError(response, 404, "Response not found");
      return;
    }

    sendJson(response, 200, payload);
  });
}

export async function handleOpenAiRequest(request, response, url) {
  const apiKey = getBearerToken(request);
  const apiKeyRecord = apiKey ? getApiKeyRecord(apiKey) : null;

  if (!apiKeyRecord) {
    log.warn("route", `Unauthorized request to ${url.pathname}`);
    sendError(response, 401, "Invalid API key");
    return true;
  }

  try {
    if (request.method === "GET" && url.pathname === "/v1/models") {
      await handleModelsRequest(response, apiKeyRecord);
      return true;
    }

    if (request.method === "POST" && url.pathname === "/v1/messages") {
      await handleMessagesRequest(request, response, apiKeyRecord);
      return true;
    }

    if (request.method === "POST" && url.pathname === "/v1/chat/completions") {
      await handleChatCompletionsRequest(request, response, apiKeyRecord);
      return true;
    }

    if (request.method === "POST" && url.pathname === "/v1/embeddings") {
      await handleEmbeddingsRequest(request, response, apiKeyRecord);
      return true;
    }

    if (request.method === "POST" && url.pathname === "/v1/responses") {
      await handleResponsesCreateRequest(request, response, apiKeyRecord);
      return true;
    }

    if (request.method === "GET" && url.pathname.startsWith("/v1/responses/")) {
      await handleResponsesGetRequest(response, apiKeyRecord, url);
      return true;
    }
  } catch (error) {
    if (!handleOpenAiError(response, error)) {
      throw error;
    }
    return true;
  }

  return false;
}

import { resolveLimitStatus } from "../services/api-error.js";
import { getApiKeyRecord } from "../services/api-key-service.js";
import { takeRoundRobinAccount } from "../services/account-rotation-service.js";
import { isIncognitoEnabledForOwner } from "../services/incognito-service.js";
import { collectOpenAiResponse, streamOpenAiResponse } from "../services/openai-bridge.js";
import { collectAnthropicMessage, streamAnthropicMessage } from "../services/anthropic-bridge.js";
import { resolveAnthropicModel } from "../services/anthropic-prompt.js";
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
  if (value.startsWith("Bearer ")) return value.slice(7);
  // Cursor Anthropic provider sends key via x-api-key header
  return (request.headers["x-api-key"] ?? "").trim();
}

function handleOpenAiError(response, error) {
  if (error.code === "USER_DISABLED" || error.code === "REQUEST_LIMIT") {
    log.warn("route", `Request rejected: code=${error.code} msg=${error.message}`);
    sendError(response, resolveLimitStatus(error), error.message, error.code);
    return true;
  }

  if (error instanceof SyntaxError) {
    log.warn("route", "Invalid JSON body");
    sendError(response, 400, "Invalid JSON body", "PARSE_ERROR");
    return true;
  }

  if (error.statusCode) {
    log.warn("route", `Error ${error.statusCode}: ${error.message}`);
    sendError(response, error.statusCode, error.message, error.code);
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
    log.debug("route", `[chat] BODY_KEYS: [${Object.keys(body).join(",")}], tools_type=${typeof body.tools}, tools_len=${Array.isArray(body.tools) ? body.tools.length : 'N/A'}, tools=${JSON.stringify(body.tools)?.slice(0, 300)}`);
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

function extractSearchQuery(body) {
  const userMsg = (body?.messages || []).find((m) => m?.role === "user");
  const content = typeof userMsg?.content === "string" ? userMsg.content : "";
  const match = content.match(/Perform a web search for the query:\s*(.+)/);
  return match ? match[1].trim() : content.slice(0, 200);
}

function hasWebSearch20250305(tools) {
  if (!Array.isArray(tools)) return false;
  return tools.some((t) => t?.type === "web_search_20250305");
}

async function handleWebSearchWithDeepseek(response, body, apiKeyRecord) {
  // web_search_20250305 是 Anthropic 专用 beta tool type。
  // DeepSeek 不支持这个 tool type，但支持 search_enabled 搜索。
  // 用 DeepSeek 的搜索模型替代，搜索结果以文本形式返回。
  const query = extractSearchQuery(body);
  const resolvedModel = resolveAnthropicModel(body.model);
  // Use DeepSeek's search-capable variant (e.g. deepseek-reasoner-expert-search)
  const searchModelId = resolvedModel.id + "-search";
  log.info("route", `[messages] web_search_20250305 → DeepSeek ${searchModelId}, query="${query.slice(0, 80)}"`);

  const modifiedBody = {
    ...body,
    model: searchModelId,
    // Strip web_search_20250305 tools — DeepSeek doesn't understand them.
    // DeepSeek's search_enabled handles the actual search.
    tools: (body.tools || []).filter((t) => t?.type !== "web_search_20250305"),
    // Clear Anthropic-format tool_choice (DeepSeek only supports OpenAI format)
    tool_choice: undefined,
  };
  // Add streaming flag if not explicitly set
  modifiedBody.stream = modifiedBody.stream !== false;

  const account = takeRoundRobinAccount(apiKeyRecord);
  if (!account) {
    log.warn("route", "No available account for web search");
    sendError(response, 404, "Account not found");
    return;
  }

  const deleteAfterFinish = isIncognitoEnabledForOwner(apiKeyRecord.ownerId);
  await streamAnthropicMessage({ response, account, body: modifiedBody, deleteAfterFinish });
}

async function handleMessagesRequest(request, response, apiKeyRecord) {
  await withOwnerRequestLimit(apiKeyRecord.ownerId, async () => {
    const rawBody = await readRequestBody(request);
    const body = parseJsonBody(rawBody) ?? {};

    // web_search_20250305 → use DeepSeek search model instead of Anthropic API
    if (hasWebSearch20250305(body.tools)) {
      await handleWebSearchWithDeepseek(response, body, apiKeyRecord);
      return;
    }

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
    // DEBUG: dump headers to diagnose Cursor Anthropic auth
    log.debug("route", `[auth-debug] pathname=${url.pathname} auth_header="${(request.headers.authorization ?? "").slice(0, 60)}" x_api_key="${(request.headers["x-api-key"] ?? "").slice(0, 60)}" apiKey_len=${apiKey.length}`);
    // Allow embedding requests without API key (used by GitNexus analyze --embeddings)
    if (url.pathname === "/v1/embeddings") {
      const { readStore } = await import("../storage/store.js");
      const firstKey = readStore().apiKeys[0];
      if (firstKey) {
        await handleEmbeddingsRequest(request, response, { ownerId: firstKey.ownerId });
        return true;
      }
    }
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

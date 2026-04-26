import { config } from "../config.js";
import { solvePowChallenge } from "./pow-solver.js";
import { createBaseHeaders, refreshAccountToken } from "./deepseek-auth.js";
import { log } from "../utils/log.js";

function buildTargetUrl(path, query) {
  const url = new URL(path, config.deepseekBaseUrl);

  Object.entries(query ?? {}).forEach(([key, value]) => {
    if (value !== undefined && value !== null && value !== "") {
      url.searchParams.set(key, String(value));
    }
  });

  return url;
}

async function createPowHeader(account, path) {
  const response = await fetch(`${config.deepseekBaseUrl}/api/v0/chat/create_pow_challenge`, {
    method: "POST",
    headers: createBaseHeaders(account.token, { "content-type": "application/json" }),
    body: JSON.stringify({ target_path: path })
  });

  const payload = await response.json();
  const challenge = payload.data.biz_data.challenge;
  const solved = await solvePowChallenge({
    ...challenge,
    expireAt: challenge.expire_at
  });

  return Buffer.from(
    JSON.stringify({
      algorithm: solved.algorithm,
      challenge: solved.challenge,
      salt: solved.salt,
      answer: solved.answer,
      signature: solved.signature,
      target_path: path
    })
  ).toString("base64");
}

async function performRequest({ account, method, path, query, body, headers }) {
  const finalHeaders = createBaseHeaders(account.token, headers);
  log.debug("proxy", `${method} ${path} accountId=${account.id}`);

  if (config.powProtectedPaths.has(path)) {
    log.debug("proxy", `Solving PoW challenge for ${path}`);
    finalHeaders["x-ds-pow-response"] = await createPowHeader(account, path);
  }

  return fetch(buildTargetUrl(path, query), {
    method,
    headers: finalHeaders,
    body
  });
}

async function maybeRefreshAccount(response, account) {
  const contentType = response.headers.get("content-type") ?? "";
  if (contentType.includes("text/event-stream")) {
    return { refreshedAccount: account, response };
  }

  const buffer = Buffer.from(await response.arrayBuffer());
  const payloadText = buffer.toString("utf8");
  const payload = contentType.includes("application/json") ? JSON.parse(payloadText) : null;
  const shouldRefresh = payload?.code === 40002 || payload?.code === 40003;

  if (!shouldRefresh) {
    if (response.status !== 200) {
      log.warn("proxy", `Upstream ${response.status}: ${payloadText.slice(0, 200)}`);
    }
    return {
      refreshedAccount: account,
      response: new Response(buffer, {
        headers: response.headers,
        status: response.status
      })
    };
  }

  log.info("proxy", `Token expired (code=${payload.code}), refreshing for accountId=${account.id}`);
  const refreshedAccount = await refreshAccountToken(account);
  return { refreshedAccount, response: null };
}

export async function proxyDeepseekRequest(options) {
  const { account } = options;
  log.debug("proxy", `Starting request: ${options.method} ${options.path}`);
  const initialResponse = await performRequest(options);
  const firstPass = await maybeRefreshAccount(initialResponse, account);

  if (firstPass.response) {
    log.debug("proxy", `First pass ok: ${options.method} ${options.path} status=${firstPass.response.status}`);
    return firstPass;
  }

  log.info("proxy", `Retrying after token refresh: ${options.method} ${options.path}`);
  const retriedResponse = await performRequest({
    ...options,
    account: firstPass.refreshedAccount
  });

  const secondPass = await maybeRefreshAccount(retriedResponse, firstPass.refreshedAccount);
  if (!secondPass.response) {
    log.error("proxy", `Token refresh failed for accountId=${account.id}`);
    throw new Error("DeepSeek token refresh failed");
  }

  log.debug("proxy", `Retry ok: ${options.method} ${options.path} status=${secondPass.response.status}`);
  return secondPass;
}

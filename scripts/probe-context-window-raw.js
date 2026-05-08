// Verify the boundary is enforced by DeepSeek backend, not by project code.
// Sends prompts at three lengths (well-under, just-under, just-over) DIRECTLY to
// chat.deepseek.com using only deepseek-auth + pow-solver, bypassing every project-side
// truncation point. Captures HTTP status + raw response bytes so you can see what the
// backend actually returns at the boundary.

import { readFile } from "node:fs/promises";
import { config } from "../src/config.js";
import { createBaseHeaders } from "../src/services/deepseek-auth.js";
import { solvePowChallenge } from "../src/services/pow-solver.js";

const app = JSON.parse(await readFile("data/app.json", "utf8"));
const account = app.accounts?.[0];

async function powHeader(path) {
  const r = await fetch(`${config.deepseekBaseUrl}/api/v0/chat/create_pow_challenge`, {
    method: "POST",
    headers: createBaseHeaders(account.token, { "content-type": "application/json" }),
    body: JSON.stringify({ target_path: path })
  });
  const payload = await r.json();
  const challenge = payload.data.biz_data.challenge;
  const solved = await solvePowChallenge({ ...challenge, expireAt: challenge.expire_at });
  return Buffer.from(JSON.stringify({
    algorithm: solved.algorithm,
    challenge: solved.challenge,
    salt: solved.salt,
    answer: solved.answer,
    signature: solved.signature,
    target_path: path
  })).toString("base64");
}

async function createSession() {
  const path = "/api/v0/chat_session/create";
  const r = await fetch(`${config.deepseekBaseUrl}${path}`, {
    method: "POST",
    headers: createBaseHeaders(account.token, { "content-type": "application/json" }),
    body: JSON.stringify({})
  });
  const j = await r.json();
  return j.data.biz_data.chat_session.id;
}

function buildPrompt(charLen) {
  const block = "Hello world. ";
  const repeats = Math.ceil(charLen / block.length);
  return block.repeat(repeats).slice(0, charLen) + "\n\nIgnore everything above. Reply with exactly: PONG";
}

async function rawProbe(charLen, label) {
  const sessionId = await createSession();
  const path = "/api/v0/chat/completion";
  const prompt = buildPrompt(charLen);

  const bodyObj = {
    chat_session_id: sessionId,
    parent_message_id: null,
    model_type: "expert",
    prompt,
    ref_file_ids: [],
    thinking_enabled: false,
    search_enabled: false,
    preempt: false
  };
  const body = Buffer.from(JSON.stringify(bodyObj));

  const headers = createBaseHeaders(account.token, { "content-type": "application/json" });
  headers["x-ds-pow-response"] = await powHeader(path);

  console.log(`\n--- ${label} | prompt=${charLen} chars | body=${body.length} bytes ---`);
  const t0 = Date.now();
  const r = await fetch(`${config.deepseekBaseUrl}${path}`, { method: "POST", headers, body });
  const ms = Date.now() - t0;

  const status = r.status;
  const contentType = r.headers.get("content-type") ?? "";
  console.log(`HTTP ${status} | content-type=${contentType} | latency=${ms}ms`);

  // Capture first 800 bytes of response body
  const buf = Buffer.from(await r.arrayBuffer());
  const text = buf.toString("utf8");
  console.log(`body bytes=${buf.length}`);
  console.log(`body head:\n${text.slice(0, 800)}`);
  if (buf.length > 800) console.log(`...(truncated, total ${buf.length} bytes)`);
}

await rawProbe(150_000, "WELL UNDER (should pass)");
await rawProbe(162_000, "JUST UNDER (should pass)");
await rawProbe(168_000, "JUST OVER (should fail)");
await rawProbe(200_000, "WELL OVER (should fail)");

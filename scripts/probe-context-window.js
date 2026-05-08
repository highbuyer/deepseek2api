// Probe DeepSeek backend context window by binary-searching prompt length.
// Bypasses HTTP layer; calls collectCompletionContent directly with first account from data/app.json.
//
// Usage:
//   node scripts/probe-context-window.js                  # default: chat-expert
//   node scripts/probe-context-window.js reasoner-expert  # test reasoner (thinking)
//
// Output: prints a probe log per length, ends with the largest accepted prompt length (chars + estimated tokens).

import { readFile } from "node:fs/promises";
import { collectCompletionContent } from "../src/services/openai-completion-runner.js";
import { estimateTokens } from "../src/utils/tool-truncation.js";

const MODEL_VARIANTS = {
  "chat-expert": { id: "deepseek-chat-expert", modelType: "expert", thinkingEnabled: false, searchEnabled: false },
  "reasoner-expert": { id: "deepseek-reasoner-expert", modelType: "expert", thinkingEnabled: true, searchEnabled: false },
  "search-reasoner-expert": { id: "deepseek-search-reasoner-expert", modelType: "expert", thinkingEnabled: true, searchEnabled: true }
};

const variant = process.argv[2] ?? "chat-expert";
const model = MODEL_VARIANTS[variant];
if (!model) {
  console.error(`Unknown variant: ${variant}. Pick one of: ${Object.keys(MODEL_VARIANTS).join(", ")}`);
  process.exit(1);
}

const app = JSON.parse(await readFile("data/app.json", "utf8"));
const account = app.accounts?.[0];
if (!account) {
  console.error("No account in data/app.json");
  process.exit(1);
}

console.log(`[probe] variant=${variant} model=${model.id} account=${account.id}`);

function buildPrompt(charLen) {
  // Repeat a neutral 13-char block, then append a tiny instruction that forces a short reply.
  // Short reply means we can detect "did the model reply at all" without waiting on long generations.
  const block = "Hello world. ";
  const repeats = Math.ceil(charLen / block.length);
  const padding = block.repeat(repeats).slice(0, charLen);
  return padding + "\n\nIgnore everything above. Reply with exactly: PONG";
}

async function probe(charLen) {
  const prompt = buildPrompt(charLen);
  const tokenEst = estimateTokens(prompt);
  const t0 = Date.now();
  try {
    const { content } = await collectCompletionContent({
      account,
      deleteAfterFinish: true,
      requestOptions: { prompt, model }
    });
    const ms = Date.now() - t0;
    const ok = content.length > 0;
    return {
      ok,
      charLen,
      tokenEst,
      ms,
      contentLen: content.length,
      sample: content.slice(0, 80).replaceAll("\n", "\\n")
    };
  } catch (e) {
    const ms = Date.now() - t0;
    return { ok: false, charLen, tokenEst, ms, error: String(e?.message ?? e).slice(0, 200) };
  }
}

function fmt(r) {
  if (r.ok) return `OK    chars=${r.charLen} ~tokens=${r.tokenEst} ms=${r.ms} replyLen=${r.contentLen} sample="${r.sample}"`;
  return `FAIL  chars=${r.charLen} ~tokens=${r.tokenEst} ms=${r.ms} error="${r.error ?? "<empty reply>"}"`;
}

// Binary search: find the largest charLen that returns a non-empty reply.
async function binarySearch(lo, hi) {
  console.log(`\n[step 1] sanity-check small prompt @ ${lo}`);
  const small = await probe(lo);
  console.log(fmt(small));
  if (!small.ok) {
    console.log("Sanity check failed — aborting (account may be invalid).");
    return null;
  }

  console.log(`\n[step 2] check upper bound @ ${hi}`);
  const big = await probe(hi);
  console.log(fmt(big));
  if (big.ok) {
    console.log(`Upper bound ${hi} accepted — window is at least this large; raise hi to find true edge.`);
    return { boundary: hi, exceeded: true };
  }

  console.log(`\n[step 3] bisecting [${lo}, ${hi}]`);
  let okLen = lo;
  let failLen = hi;
  while (failLen - okLen > 4096) {
    const mid = Math.floor((okLen + failLen) / 2);
    const r = await probe(mid);
    console.log(fmt(r));
    if (r.ok) okLen = mid;
    else failLen = mid;
  }
  return { boundary: okLen, failsAt: failLen };
}

// Range: 30K chars (~9K tokens) to 500K chars (~150K tokens).
// Should bracket whatever V4 expert window is.
const result = await binarySearch(30_000, 500_000);
console.log("\n=== RESULT ===");
console.log(JSON.stringify(result, null, 2));
if (result?.boundary) {
  const proxy = buildPrompt(result.boundary);
  console.log(`\nLargest accepted prompt: ${result.boundary} chars ≈ ${estimateTokens(proxy)} tokens (estimateTokens heuristic)`);
}

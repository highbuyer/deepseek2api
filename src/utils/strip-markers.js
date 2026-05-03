/* Tags inserted by the thinking tagger or leaked by the model that must
 * never reach user-visible text.  This is the single source of truth —
 * the streaming sieve imports `LEAK_TAG_PREFIXES` to hold partial tags
 * across chunk boundaries, and `stripLeakedMarkers` regex below removes
 * any complete token that slips through. */
const LEAK_TAG_NAMES = Object.freeze([
  "think",
  "tool_calls", "function_calls",
  "tool_call", "function_call",
  "tool_call_name", "function_call_name",
  "invoke", "tool_use", "apply_patch",
  "tool_name", "function_name",
  "tool_result",
  "parameter", "parameters"
]);

/* Prefixes the sieve must recognize as in-progress tags so it withholds
 * them until the closing `>` arrives.  Both `<name` and `</name` forms
 * are listed; the sieve checks `prefix.startsWith(tail)` against each. */
export const LEAK_TAG_PREFIXES = Object.freeze(
  LEAK_TAG_NAMES.flatMap(n => [`<${n}`, `</${n}`])
);

/* Regex that strips individual leak tags (opening and closing) but NOT the
 * content between them.  Used as a first pass for structural tags where only
 * the wrapper is unwanted (e.g. <tool_calls>, <invoke>). */
const LEAK_TAG_REGEX = new RegExp(
  `<\\/?(?:${LEAK_TAG_NAMES.join("|")})(?:_name)?\\b[^>]*\\/?\\s*>`,
  "gi"
);

/* Regexes that strip ENTIRE blocks including their content.
 * These are for tags whose inner content must never reach the user:
 *   - <tool_result>…</tool_result> — echoed tool output (file contents, etc.)
 *   - <think(ing)>…</think(ing)>   — leaked reasoning blocks
 *   - <tool_calls>…</tool_calls>   — full tool-call blocks that slipped through
 *   - <function_calls>…</function_calls> — same, alternate tag name */
const BLOCK_STRIP_PATTERNS = Object.freeze([
  /<(?:tool_result)\b[^>]*>[\s\S]*?<\/(?:tool_result)\s*>/gi,
  /<(?:think|thinking)\b[^>]*>[\s\S]*?<\/(?:think|thinking)\s*>/gi,
  /<(?:tool_calls)\b[^>]*>[\s\S]*?<\/(?:tool_calls)\s*>/gi,
  /<(?:function_calls)\b[^>]*>[\s\S]*?<\/(?:function_calls)\s*>/gi,
]);

/* Substrings that hint at any leak token.  Keeps the fast-path cheap:
 * if none appear, no regex pass runs.  Must stay in sync with the
 * names above — anything addable to LEAK_TAG_NAMES that doesn't
 * include "tool_"/"function_"/"think>"/"invoke>"/"apply_patch"/
 * "parameter" needs a new sentinel here. */
const FAST_PATH_SENTINELS = ["think>", "tool_", "function_", "invoke", "apply_patch", "parameter", "TOOL:"];

export function stripLeakedMarkers(text) {
  if (!FAST_PATH_SENTINELS.some(s => text.includes(s))) {
    // Still check for echoed conversation markers
    if (!text.includes("USER:") && !text.includes("ASSISTANT:")) {
      return text;
    }
  }

  let result = text
    .replace(/^\[proxy\]<\/think\s*>/i, "");

  // First pass: strip ENTIRE blocks (tag + content) for tags whose
  // inner text must never be visible (tool_result, think, tool_calls, etc.)
  for (const pattern of BLOCK_STRIP_PATTERNS) {
    result = result.replace(pattern, "");
  }

  // Second pass: strip any remaining individual leak tags that slipped
  // through as orphans or partials (opening/closing without a matching pair)
  result = result.replace(LEAK_TAG_REGEX, "");

  // Strip echoed conversation role markers and TOOL: prefix
  result = result.replace(/^(?:USER|ASSISTANT|TOOL):\s*/gmi, "");

  return result;
}

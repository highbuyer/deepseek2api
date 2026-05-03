/* Tags inserted by the thinking tagger or leaked by the model that must
 * never reach user-visible text.  This is the single source of truth —
 * the streaming sieve imports `LEAK_TAG_PREFIXES` to hold partial tags
 * across chunk boundaries, and `stripLeakedMarkers` regex below removes
 * any complete token that slips through. */
const LEAK_TAG_NAMES = Object.freeze([
  "think",
  "tool_calls", "function_calls",
  "tool_call", "function_call",
  "invoke", "tool_use", "apply_patch",
  "tool_name", "function_name",
  "parameter", "parameters"
]);

/* Prefixes the sieve must recognize as in-progress tags so it withholds
 * them until the closing `>` arrives.  Both `<name` and `</name` forms
 * are listed; the sieve checks `prefix.startsWith(tail)` against each. */
export const LEAK_TAG_PREFIXES = Object.freeze(
  LEAK_TAG_NAMES.flatMap(n => [`<${n}`, `</${n}`])
);

const LEAK_TAG_REGEX = new RegExp(
  `<\\/?(?:${LEAK_TAG_NAMES.join("|")})(?:_name)?\\b[^>]*\\/?\\s*>`,
  "gi"
);

/* Substrings that hint at any leak token.  Keeps the fast-path cheap:
 * if none appear, no regex pass runs.  Must stay in sync with the
 * names above — anything addable to LEAK_TAG_NAMES that doesn't
 * include "tool_"/"function_"/"think>"/"invoke>"/"apply_patch"/
 * "parameter" needs a new sentinel here. */
const FAST_PATH_SENTINELS = ["think>", "tool_", "function_", "invoke", "apply_patch", "parameter"];

export function stripLeakedMarkers(text) {
  if (!FAST_PATH_SENTINELS.some(s => text.includes(s))) {
    return text;
  }
  return text
    .replace(/^\[proxy\]<\/think\s*>/i, "")
    .replace(LEAK_TAG_REGEX, "");
}

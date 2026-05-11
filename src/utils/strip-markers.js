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
  // Individual tool-call blocks: when the sieve replays a failed parse as
  // text, invoke/tool_call/function_call must be STRIPPED ENTIRELY (tag +
  // content), not just tag-stripped.  Tag-only removal leaves parameter
  // values (file paths, search patterns) visible to the user.
  /<(?:invoke|tool_use)\b[^>]*>[\s\S]*?<\/(?:invoke|tool_use)\s*>/gi,
  /<(?:tool_call|function_call)\b[^>]*>[\s\S]*?<\/(?:tool_call|function_call)\s*>/gi,
]);

/* Substrings that hint at any leak token.  Keeps the fast-path cheap:
 * if none appear, no regex pass runs.  Must stay in sync with the
 * names above — anything addable to LEAK_TAG_NAMES that doesn't
 * include "tool_"/"function_"/"think>"/"invoke>"/"apply_patch"/
 * "parameter" needs a new sentinel here. */
const FAST_PATH_SENTINELS = ["think>", "tool_", "function_", "invoke", "apply_patch", "parameter", "TOOL:", "</calls", "</function_ca", "</tool_ca"];

export function stripLeakedMarkers(text) {
  if (!FAST_PATH_SENTINELS.some(s => text.includes(s))) {
    // Check for echoed conversation markers (with or without colon — model
    // sometimes writes "ASSISTANT\n\n" as a role banner without trailing :)
    if (!text.includes("ASSISTANT") && !text.includes("USER") && !text.includes("TOOL")) {
      return text;
    }
  }

  let result = text
    .replace(/\[proxy\]<\/think\s*>/gi, "");

  // First pass: strip ENTIRE blocks (tag + content) for tags whose
  // inner text must never be visible (tool_result, think, tool_calls, etc.)
  for (const pattern of BLOCK_STRIP_PATTERNS) {
    result = result.replace(pattern, "");
  }

  // Second pass: strip any remaining individual leak tags that slipped
  // through as orphans or partials (opening/closing without a matching pair)
  result = result.replace(LEAK_TAG_REGEX, "");

  // Third pass: strip close-tag fragments that LEAK_TAG_REGEX misses
  // because they lack the closing ">", or lack the proper prefix.
  // strip them literally — these fragments never appear in valid user text.
  for (const frag of ["</calls", "</function_calls", "</tool_calls"]) {
    result = result.replaceAll(frag, "");
  }

  // Strip leaked role markers — covers all banner variants:
  //   "MARKER: content"         (colon separator)
  //   "MARKER content"          (space separator — model uses MARKER as heading)
  //   "MARKER\n\n"              (alone on line, no separator content)
  //   "MARKER `code`"           (space + markdown)
  // Line-start marker + any separator (colon/whitespace/end). Strips only
  // marker + separator; preserves following content (which is model's real
  // response, not part of the role banner).
  result = result.replace(/^\s*(?:USER|ASSISTANT|TOOL)(?:[\s:]+|$)/gmi, "");
  // Inline marker + colon (mid-line, after word boundary)
  result = result.replace(/\b(?:USER|ASSISTANT|TOOL):\s*/gmi, "");

  return result;
}

// Post-stream cleanup: run on the complete text after streaming ends.
// Catches any XML fragments that leaked across SSE chunk boundaries.
export function finalStrip(text) {
  if (!text) return text;

  let result = text;

  // Strip any unclosed XML opens (e.g. "<function_calls", "<tool_call")
  // that leaked because the close tag arrived in a separate chunk
  result = result.replace(/<(?:function_calls|tool_calls|tool_call|function_call|invoke|tool_use|tool_name)\b[^>]*\/?\s*>/gi, "");

  // Strip orphan close tags (including malformed/truncated ones without ">")
  result = result.replace(/<\/(?:function_calls|tool_calls|tool_call|function_call|invoke|tool_use|tool_name|tool_call_name|function_call_name)\b[^>]*>?/gi, "");
  // Bare "</calls" — no function_/tool_ prefix.  Strip literally because
  // \b fails when it runs into adjacent letters (e.g. "</callsvalue").
  result = result.replaceAll("</calls", "");

  // Strip any remaining unclosed/open tool tags without matching close
  result = result.replace(/<(?:function_calls|tool_calls|tool_call|function_call|invoke|tool_use|tool_name|parameter|parameters)\b[^>]*\/?\s*>?/gi, "");

  // Strip leaked CDATA wrappers
  result = result.replace(/<!\[CDATA\[|\]\]>/gi, "");

  // Strip leaked role markers (any separator: colon/space/end-of-line)
  result = result.replace(/^\s*(?:USER|ASSISTANT|TOOL)(?:[\s:]+|$)/gmi, "");
  result = result.replace(/\b(?:USER|ASSISTANT|TOOL):\s*/gmi, "");

  // Collapse multiple blank lines from stripping
  result = result.replace(/\n{3,}/g, "\n\n");

  return result.trim();
}

// Streaming text buffer that prevents TOOL:/USER:/ASSISTANT: role-marker
// leaks across SSE chunk boundaries.  Only holds back the suffix when it
// looks like a partial marker prefix (e.g. "TO" before "OL:" arrives),
// so normal text is emitted immediately without fragmentation.
const MARKER_PREFIXES = ["T","TO","TOO","TOOL","U","US","USE","USER","A","AS","ASS","ASSI","ASSIS","ASSIST","ASSISTA","ASSISTAN","ASSISTANT","[","[p","[pr","[pro","[prox","[proxy"];
const MAX_MARKER_PREFIX = 10;

export function createStreamTextStripper() {
  let buffer = "";

  return {
    push(text) {
      if (!text) return "";
      buffer += text;
      const cleaned = stripLeakedMarkers(buffer);
      // Find the shortest tail that could be a marker prefix
      let hold = 0;
      for (let n = Math.min(MAX_MARKER_PREFIX, cleaned.length); n > 0; n--) {
        const tail = cleaned.slice(-n);
        if (MARKER_PREFIXES.includes(tail)) {
          hold = n;
          break;
        }
      }
      const safeLen = cleaned.length - hold;
      const emit = cleaned.slice(0, safeLen);
      buffer = cleaned.slice(safeLen);
      return emit;
    },
    flush() {
      const result = stripLeakedMarkers(buffer);
      buffer = "";
      return result;
    }
  };
}

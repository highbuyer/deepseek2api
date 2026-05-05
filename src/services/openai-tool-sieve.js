import { parseToolCallsFromText } from "./openai-tool-parser.js";
import { toStringSafe } from "../utils/safe-string.js";
import { stripLeakedMarkers, finalStrip } from "../utils/strip-markers.js";
import { log } from "../utils/log.js";

// Shared format error message — used by all bridge layers when the model
// drifts to an unrecognized tool call XML format (Claude Code tool_use_error pattern).
// NOTE: Contains raw XML tags that must survive stripLeakedMarkers.
// Bridge layers pass this directly to emitTextEvent with skipStrip=true.
// FORMAT_ERROR_MSG must use only plain text — no XML tags, no bracket markers
// that the model could "helpfully" convert back to XML and re-trigger the sieve.
export const FORMAT_ERROR_MSG =
  "[TOOL_FORMAT_ERROR] Your tool call format was incorrect. " +
  "Use the standard format: a function_calls block containing " +
  "one or more invoke elements, each with a name attribute and " +
  "parameter child elements with name and string attributes. " +
  "Example: invoke name=ToolName with parameter name=key string=true value.";

/* ── Tag detection ──
 * We only need to detect BLOCK-LEVEL tool containers.  The parser handles
 * individual <tool_call>/<invoke> extraction from the captured block. */

const TOOL_BLOCK_OPENS = [
  "<function_calls",
  "<tool_calls",
  "<tool_result",
  "<tool_call_name",
  "<tool_call",
  "<function_call_name",
  "<function_call",
  "<invoke",
  "<tool_use",
  "<apply_patch",
];

const TOOL_BLOCK_CLOSES = [
  "</function_calls>",
  "</tool_calls>",
  "</tool_result>",
  "</tool_call_name>",
  "</tool_call>",
  "</function_call_name>",
  "</function_call>",
  "</invoke>",
  "</tool_use>",
  "</apply_patch>",
];

// Max chars to hold at end of buffer for chunk-boundary detection
const LOOKBEHIND = 30;
// Max capture before forced flush
const MAX_CAPTURE = 50000;

function buildBlockTags(allowedToolNames) {
  const opens = [...TOOL_BLOCK_OPENS];
  const closes = [...TOOL_BLOCK_CLOSES];
  for (const name of (allowedToolNames ?? [])) {
    const lower = name.toLowerCase();
    const openTag = `<${lower}`;
    if (!opens.includes(openTag)) {
      opens.push(openTag);
      closes.push(`</${lower}>`);
    }
  }
  return { opens, closes };
}

/* ── Find the first tool block open tag in text ── */
function findFirstOpen(text, opens) {
  // Text is already lowercased by caller to avoid per-call copy
  let best = -1;
  for (const tag of opens) {
    const idx = text.indexOf(tag);
    if (idx >= 0 && (best === -1 || idx < best)) {
      best = idx;
    }
  }
  return best >= 0 ? { index: best } : null;
}

/* ── Find furthest close tag end position ── */
function findCloseEnd(text, closes) {
  let best = -1;
  for (const tag of closes) {
    const idx = text.lastIndexOf(tag);
    if (idx >= 0) {
      const end = idx + tag.length;
      if (end > best) best = end;
    }
  }
  return best;
}

/* ── Structured stream handler (Claude model: detect → buffer → parse) ── */
export function createToolSieve(allowedToolNames = []) {
  const { opens, closes } = buildBlockTags(allowedToolNames);

  const state = {
    allowedToolNames,
    opens,
    closes,
    capture: "",
    capturing: false,
    emittedText: "",
    hold: "",      // small lookbehind for chunk-boundary detection
    lastKind: "response",
    formatErrorEmitted: false
  };

  function pushTextEvent(events, text, kind) {
    if (!text) return;
    state.emittedText += text;
    if (state.emittedText.length > 100000) {
      state.emittedText = state.emittedText.slice(-50000);
    }
    events.push({ type: "text", text, kind: kind ?? state.lastKind ?? "response" });
  }

  function drain() {
    const events = [];

    if (state.capturing) {
      state.capture += state.hold;
      state.hold = "";

      // Overflow guard: flush as text to prevent unbounded memory
      if (state.capture.length > MAX_CAPTURE) {
        log.warn("sieve", `Capture overflow (${state.capture.length} chars), flushing as text`);
        const overflowCalls = parseToolCallsFromText(state.capture, state.allowedToolNames);
        if (overflowCalls.length) {
          events.push({ type: "tool_calls", calls: overflowCalls });
        }
        pushTextEvent(events, state.capture, state.lastKind);
        state.capture = "";
        state.capturing = false;
        return events;
      }

      // Drop <tool_result> blocks — they are echoed prompt content, never valid tool calls.
      // Without this, large tool results overflow and leak file content into the SSE stream.
      const captureLower = state.capture.toLowerCase();
      if (captureLower.indexOf("<tool_result") >= 0) {
        const closeEnd = captureLower.lastIndexOf("</tool_result>");
        if (closeEnd >= 0) {
          const suffix = state.capture.slice(closeEnd + "</tool_result>".length);
          state.capture = "";
          state.capturing = false;
          if (suffix) state.hold = suffix;
          if (suffix) { const more = drain(); events.push(...more); }
          return events;
        }
        // Close not found, keep buffering (will overflow-flush if too large)
      }

      // Try to find close tag
      const closeEnd = findCloseEnd(captureLower, state.closes);
      if (closeEnd > 0) {
        const block = state.capture.slice(0, closeEnd);
        const suffix = state.capture.slice(closeEnd);
        const calls = parseToolCallsFromText(block, state.allowedToolNames);
        if (calls.length) {
          events.push({ type: "tool_calls", calls });
        } else {
          // Only emit format_error if the block looks like a genuine but malformed
          // tool call attempt — not if the model just mentioned XML tags in prose.
          const isRealAttempt = /<(?:[a-z0-9_:-]+:)?invoke\s+name=/i.test(block)
            || /<(?:[a-z0-9_:-]+:)?tool_name\b[^>]*>/i.test(block)
            || /<(?:[a-z0-9_:-]+:)?parameter\s+name=/i.test(block)
            || /<(?:[a-z0-9_:-]+:)?tool_call\s+name=/i.test(block);
          if (isRealAttempt && !state.formatErrorEmitted) {
            state.formatErrorEmitted = true;
            events.push({
              type: "format_error",
              block: block.slice(0, 200),
              message: "Unrecognized tool call format"
            });
          } else if (!isRealAttempt) {
            // Not a tool call attempt — model mentioned XML tags in prose.
            // Replay only the captured block (before the close tag), not the
            // full state.capture.  The suffix (after close) will be handled
            // by the drain() recursion below — using state.capture here
            // would emit suffix twice.
            pushTextEvent(events, block, state.lastKind);
          }
        }
        state.capture = "";
        state.capturing = false;
        if (suffix) state.hold = suffix;
        if (suffix) {
          const more = drain();
          events.push(...more);
        }
        return events;
      }
      // Close not found yet — keep buffering
      return events;
    }

    // Not capturing — scan for tool block start
    const text = state.hold;
    state.hold = "";
    if (!text) return events;

    const open = findFirstOpen(text.toLowerCase(), state.opens);
    if (open) {
      // Emit text before the block
      if (open.index > 0) {
        pushTextEvent(events, text.slice(0, open.index), state.lastKind);
      }
      state.capture = text.slice(open.index);
      state.capturing = true;
      // Recurse to process the capture
      const more = drain();
      events.push(...more);
    } else {
      // No tool block start — check if text ends with '<' (possible partial tag)
      const lastLt = text.lastIndexOf("<");
      if (lastLt >= 0 && !text.slice(lastLt).includes(">")) {
        // Partial tag at end — hold only the partial portion
        const partialLen = text.length - lastLt;
        if (partialLen <= LOOKBEHIND) {
          if (lastLt > 0) pushTextEvent(events, text.slice(0, lastLt), state.lastKind);
          state.hold = text.slice(lastLt);
        } else {
          pushTextEvent(events, text, state.lastKind);
        }
      } else {
        pushTextEvent(events, text, state.lastKind);
      }
    }

    return events;
  }

  return Object.freeze({
    push(chunk, kind) {
      const text = toStringSafe(chunk);
      if (kind) state.lastKind = kind;
      state.hold += text;
      return drain();
    },

    flush() {
      // Flush hold buffer
      const events = [];
      if (state.hold && !state.capturing) {
        pushTextEvent(events, state.hold, state.lastKind);
        state.hold = "";
      }

      if (state.capturing && state.capture) {
        // Try to parse what we have (stream ended mid-block)
        const calls = parseToolCallsFromText(state.capture, state.allowedToolNames);
        if (calls.length) {
          events.push({ type: "tool_calls", calls });
        } else {
          pushTextEvent(events, state.capture, state.lastKind);
        }
        state.capture = "";
        state.capturing = false;
      }

      return events;
    },

    get captureLength() { return state.capture.length; },
    get pendingLength() { return state.hold.length + state.capture.length; },
    get isCapturing() { return state.capturing; },
    get emittedText() { return state.emittedText; }
  });
}

/* ── Non-streaming helpers (unchanged API) ── */

function toTextEvent(chunk, kind = "response") {
  return { type: "text", text: toStringSafe(chunk), kind };
}

function flattenToolEvents(events) {
  return events.reduce((output, event) => {
    if (!output.length || event.type !== "text" || output.at(-1).type !== "text") {
      output.push(event);
      return output;
    }
    output[output.length - 1] = {
      type: "text",
      text: `${output.at(-1).text}${event.text}`,
      kind: output.at(-1).kind
    };
    return output;
  }, []);
}

export function splitToolAwareEvents(text, allowedToolNames = []) {
  if (!allowedToolNames?.length) {
    return [toTextEvent(text)];
  }
  const sieve = createToolSieve(allowedToolNames);
  const events = [...sieve.push(text), ...sieve.flush()];
  return flattenToolEvents(events);
}

export function extractToolAwareOutput(text, allowedToolNames = []) {
  const events = splitToolAwareEvents(text, allowedToolNames);
  let content = events
    .filter((event) => event.type === "text")
    .map((event) => stripLeakedMarkers(event.text))
    .join("")
    .trimStart();
  // Final post-stream cleanup — strips any XML fragments that leaked
  // across chunk boundaries during streaming (Claude Code pattern).
  content = finalStrip(content);
  return {
    events,
    content,
    toolCalls: events.flatMap((event) => event.type === "tool_calls" ? event.calls ?? [] : [])
  };
}

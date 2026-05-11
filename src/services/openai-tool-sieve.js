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
  "Use a ```json code fence with a JSON array of {tool, arguments} objects. " +
  "Example: ```json\\n[{\"tool\": \"read\", \"arguments\": {\"file_path\": \"/src/main.py\"}}]\\n```";

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
    captureKind: null,  // null | "xml" | "json" — how to parse the captured block
    emittedText: "",
    hold: "",      // small lookbehind for chunk-boundary detection
    lastKind: "response",
    inCodeFence: false,  // true when text is inside a ``` fenced code block (non-json)
    formatErrorEmitted: false  // only emit one format_error per stream
  };

  /* ── Detect empty nested tool container tags (model glitch: repeated <tool_calls>
     with no actual tool invocations inside, just more nested containers + whitespace) ── */
  function isEffectivelyEmptyToolBlock(text) {
    const noTags = text.replace(/<[^>]*>/g, "");
    return noTags.trim().length === 0;
  }

  // Check if a block contains inner tool call structure tags (beyond just
  // the outer container) — signals the model genuinely tried a tool call.
  function hasToolCallStructure(text) {
    return /<(?:tool_call\b(?!s)|invoke\b|function_call\b|tool_name\b|function_name\b)/i.test(text);
  }

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
        state.captureKind = null;
        return events;
      }

      // ── JSON capture: look for closing marker (``` or ] for bare arrays) ──
      if (state.captureKind === "json") {
        const hasFence = state.capture.startsWith("```");
        if (hasFence) {
          // Fenced JSON: ```json ... ```
          const prefixLen = state.capture.startsWith("```json") ? 7
            : state.capture.startsWith("```JSON") ? 7
            : 3;
          const closeIdx = state.capture.lastIndexOf("```");
          if (closeIdx > prefixLen) {
            const jsonBlock = state.capture.slice(prefixLen, closeIdx).trim();
            const suffix = state.capture.slice(closeIdx + 3);
            const calls = jsonBlock ? parseToolCallsFromText(jsonBlock, state.allowedToolNames) : [];
            log.debug("sieve", `JSON drain: jsonBlock=${jsonBlock.length} chars, calls=${calls.length}, preview="${jsonBlock.slice(0, 200)}"`);
            if (calls.length) {
              events.push({ type: "tool_calls", calls });
            } else if (jsonBlock) {
              log.debug("sieve", `JSON drain: parse returned 0 calls, emitting as text (${state.capture.slice(0, closeIdx + 3).length} chars)`);
              pushTextEvent(events, state.capture.slice(0, closeIdx + 3), state.lastKind);
            }
            state.capture = "";
            state.capturing = false;
            state.captureKind = null;
            if (suffix) state.hold = suffix;
            if (suffix) { const more = drain(); events.push(...more); }
            return events;
          }
          return events;
        }
        // Bare JSON array: [...] — find matching ] by bracket depth
        let depth = 0, inString = false, escape = false, closeAt = -1;
        for (let i = 0; i < state.capture.length; i++) {
          const ch = state.capture[i];
          if (escape) { escape = false; continue; }
          if (ch === "\\" && inString) { escape = true; continue; }
          if (ch === '"') { inString = !inString; continue; }
          if (inString) continue;
          if (ch === "[" || ch === "{") { depth++; }
          else if (ch === "]" || ch === "}") { depth--; }
          if (depth === 0 && ch === "]") { closeAt = i; break; }
        }
        if (closeAt > 0) {
          const jsonBlock = state.capture.slice(0, closeAt + 1);
          const suffix = state.capture.slice(closeAt + 1);
          const calls = parseToolCallsFromText(jsonBlock, state.allowedToolNames);
          if (calls.length) {
            events.push({ type: "tool_calls", calls });
          } else {
            pushTextEvent(events, jsonBlock, state.lastKind);
          }
          state.capture = "";
          state.capturing = false;
          state.captureKind = null;
          if (suffix) state.hold = suffix;
          if (suffix) { const more = drain(); events.push(...more); }
          return events;
        }
        return events;
      }

      // ── XML capture (existing logic) ──

      // Drop <tool_result> blocks — they are echoed prompt content, never valid tool calls.
      // Without this, large tool results overflow and leak file content into the SSE stream.
      const captureLower = state.capture.toLowerCase();
      log.debug("sieve", `XML capture drain: len=${state.capture.length} kind=${state.captureKind} closeEnd=${findCloseEnd(captureLower, state.closes)} preview="${state.capture.slice(0, 80)}"`);
      if (captureLower.indexOf("<tool_result") >= 0) {
        const closeEnd = captureLower.lastIndexOf("</tool_result>");
        if (closeEnd >= 0) {
          const suffix = state.capture.slice(closeEnd + "</tool_result>".length);
          state.capture = "";
          state.capturing = false;
          state.captureKind = null;
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
        } else if (!state.formatErrorEmitted && (isEffectivelyEmptyToolBlock(block) || hasToolCallStructure(block))) {
          // Block has tool call markers but parser found no valid calls.
          // Emit FORMAT_ERROR to steer model toward ```json fence format.
          log.debug("sieve", `Drain: unparseable tool block (block=${block.length} chars, isEmpty=${isEffectivelyEmptyToolBlock(block)}, hasStructure=${hasToolCallStructure(block)}), emitting format_error`);
          state.formatErrorEmitted = true;
          events.push({ type: "format_error", block });
        } else {
          // Parse returned nothing but block has real text content —
          // replay as text so the model stays in its normal flow.
          log.debug("sieve", `Drain: parse returned 0 calls, isEmpty=${isEffectivelyEmptyToolBlock(block)}, blockPreview="${block.slice(0, 100)}"`);
          pushTextEvent(events, block, state.lastKind);
        }
        state.capture = "";
        state.capturing = false;
        state.captureKind = null;
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

    // If the whole chunk is inside a ``` fence, emit as-is (no tool scanning).
    if (state.inCodeFence) {
      // Detect if this fence is actually a ```json tool call block
      // that was split across chunks (e.g. chunk1="```j", chunk2="son\n[...]")
      if (/^json\b/i.test(text)) {
        // Switch from fence mode to JSON capture
        state.inCodeFence = false;
        state.capture = "```" + text;
        state.capturing = true;
        state.captureKind = "json";
        state.hold = "";
        const more = drain();
        events.push(...more);
        return events;
      }
      // Check if this chunk closes the fence
      const fenceClose = text.indexOf("```");
      if (fenceClose < 0) {
        // Still inside — emit all, keep partial-tag hold
        const lastLt = text.lastIndexOf("<");
        if (lastLt >= 0 && !text.slice(lastLt).includes(">") && text.length - lastLt <= LOOKBEHIND) {
          if (lastLt > 0) pushTextEvent(events, text.slice(0, lastLt), state.lastKind);
          state.hold = text.slice(lastLt);
        } else {
          pushTextEvent(events, text, state.lastKind);
        }
        return events;
      }
      // Fence closes in this chunk — emit up to + including ```, then
      // process the rest normally.
      const before = text.slice(0, fenceClose + 3);
      const after = text.slice(fenceClose + 3);
      pushTextEvent(events, before, state.lastKind);
      state.inCodeFence = false;
      // Recurse with the remaining text
      if (after) { state.hold = after; const more = drain(); events.push(...more); }
      return events;
    }

    // Outside fence — check if a ``` opens a code block or JSON tool call
    const fenceOpen = text.indexOf("```");
    if (fenceOpen >= 0) {
      const afterFence = text.slice(fenceOpen + 3);
      // ```json or ```JSON → tool call block, NOT a regular fence
      if (/^json\b/i.test(afterFence)) {
        log.debug("sieve", `Detected JSON fence at position ${fenceOpen}, starting JSON capture`);
        if (fenceOpen > 0) {
          pushTextEvent(events, text.slice(0, fenceOpen), state.lastKind);
        }
        state.capture = text.slice(fenceOpen);
        state.capturing = true;
        state.captureKind = "json";
        const more = drain();
        events.push(...more);
        return events;
      }
      // Ambiguous: afterFence could be a prefix of "json" (chunk boundary)
      // Hold everything and wait for more text to decide.
      if (afterFence.length < 4 && "json".startsWith(afterFence.toLowerCase())) {
        if (fenceOpen > 0) pushTextEvent(events, text.slice(0, fenceOpen), state.lastKind);
        state.hold = text.slice(fenceOpen);
        return events;
      }
      // Regular ``` fence — emit as text, enter fence mode
      const before = text.slice(0, fenceOpen);
      const after = text.slice(fenceOpen + 3);
      if (before) { state.hold = before; const more = drain(); events.push(...more); }
      pushTextEvent(events, "```", state.lastKind);
      state.inCodeFence = true;
      if (after) { state.hold = after; const more = drain(); events.push(...more); }
      return events;
    }

    // ── Bare JSON array detection: [{ "tool": or [{ "name": ──
    // Model sometimes outputs JSON without the ```json fence.
    // Detection works in two steps: [ or [{ triggers a hold,
    // next chunk confirms if it's a tool call JSON array.
    const bracketIdx = text.search(/\[\s*\{/);
    if (bracketIdx >= 0) {
      const afterBracket = text.slice(bracketIdx);
      const isToolJson = /^\[\s*\{\s*"(?:tool|name)"\s*:/.test(afterBracket);
      if (isToolJson) {
        if (bracketIdx > 0) {
          pushTextEvent(events, text.slice(0, bracketIdx), state.lastKind);
        }
        state.capture = text.slice(bracketIdx);
        state.capturing = true;
        state.captureKind = "json";
        const more = drain();
        events.push(...more);
        return events;
      }
      // Ambiguous: has [{ but "tool"/"name" not (yet) visible —
      // could be split across chunks.  Hold from [ and wait.
      if (!isToolJson && afterBracket.length < 20) {
        if (bracketIdx > 0) pushTextEvent(events, text.slice(0, bracketIdx), state.lastKind);
        state.hold = text.slice(bracketIdx);
        return events;
      }
    }
    // Check if held text from a previous partial [{ now completes
    if (/^\[\s*\{\s*"(?:tool|name)"\s*:/.test(text)) {
      state.capture = text;
      state.capturing = true;
      state.captureKind = "json";
      state.hold = "";
      const more = drain();
      events.push(...more);
      return events;
    }

    const open = findFirstOpen(text.toLowerCase(), state.opens);
    if (open) {
      const matchedTag = state.opens.find(t => text.toLowerCase().indexOf(t) === open.index) || "?";
      log.debug("sieve", `Capture start: tag="${matchedTag}" at pos=${open.index}, kind=${state.lastKind}, textPreview="${text.slice(0, 100)}"`);
      if (open.index > 0) {
        pushTextEvent(events, text.slice(0, open.index), state.lastKind);
      }
      state.capture = text.slice(open.index);
      state.capturing = true;
      state.captureKind = "xml";
      const more = drain();
      events.push(...more);
    } else {
      const lastLt = text.lastIndexOf("<");
      if (lastLt >= 0 && !text.slice(lastLt).includes(">")) {
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
      const prevKind = state.lastKind;
      if (kind) state.lastKind = kind;

      // When switching from thinking to response, force-flush any open
      // XML capture.  Models often mention tool names during reasoning,
      // which triggers capture that spans both think and response blocks.
      // The captured thinking text defeats isEmpty checks, preventing
      // format_error emission for empty response <tool_calls> blocks.
      if (prevKind === "thinking" && kind === "response" && state.capturing && state.captureKind === "xml") {
        log.debug("sieve", `Kind switch thinking→response, flushing capture (${state.capture.length} chars) as text`);
        const events = [];
        pushTextEvent(events, state.capture, prevKind);
        state.capture = "";
        state.capturing = false;
        state.captureKind = null;
        // Also flush any held text before this chunk
        if (state.hold) {
          pushTextEvent(events, state.hold, prevKind);
          state.hold = "";
        }
        state.hold = text;
        const more = drain();
        events.push(...more);
        return events;
      }

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
        let parseText = state.capture;
        if (state.captureKind === "json") {
          log.debug("sieve", `JSON flush: capture=${state.capture.length} chars, hold=${state.hold.length}`);
          // Strip ```json prefix and trailing ``` if present
          parseText = parseText.replace(/^```json\b\s*/i, "").replace(/^```JSON\b\s*/i, "");
          const closeIdx = parseText.lastIndexOf("```");
          if (closeIdx >= 0) parseText = parseText.slice(0, closeIdx);
          parseText = parseText.trimEnd();
          // Stream may have ended before closing ] arrived — try adding it
          if (!parseText.endsWith("]")) {
            parseText += "]";
          }
        }
        const calls = parseToolCallsFromText(parseText, state.allowedToolNames);
        if (calls.length) {
          events.push({ type: "tool_calls", calls });
        } else if (!state.formatErrorEmitted && state.captureKind === "xml" && isEffectivelyEmptyToolBlock(state.capture)) {
          // Stream ended with empty nested tool tags — correct the model
          log.debug("sieve", `Flush: empty tool tags detected (capture=${state.capture.length} chars), emitting format_error`);
          state.formatErrorEmitted = true;
          events.push({ type: "format_error", block: state.capture });
        } else {
          log.debug("sieve", `Flush: parse returned 0 calls, captureKind=${state.captureKind}, formatErrorEmitted=${state.formatErrorEmitted}, isEmpty=${isEffectivelyEmptyToolBlock(state.capture)}, capturePreview="${state.capture.slice(0, 100)}"`);
          pushTextEvent(events, state.capture, state.lastKind);
        }
        state.capture = "";
        state.capturing = false;
        state.captureKind = null;
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

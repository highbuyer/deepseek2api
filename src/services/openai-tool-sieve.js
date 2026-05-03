import { parseToolCallsFromText } from "./openai-tool-parser.js";
import { toStringSafe } from "../utils/safe-string.js";
import { stripLeakedMarkers, LEAK_TAG_PREFIXES } from "../utils/strip-markers.js";
import { log } from "../utils/log.js";

/* Each capture pair has one open and a list of acceptable close variants.
 * `<tool_call>` is closed by either `</tool_call>` (canonical) or
 * `</tool_call_name>` (DeepSeek's frequent garbled form).  Listing both lets
 * the streaming sieve emit tool_calls as soon as the block ends, instead of
 * waiting for end-of-stream and falling back to synthetic close. */
const TOOL_CAPTURE_PAIRS = Object.freeze([
  { open: "<tool_calls", closes: ["</tool_calls>"] },
  { open: "<function_calls", closes: ["</function_calls>"] },
  { open: "<tool_call", closes: ["</tool_call>", "</tool_call_name>"] },
  { open: "<function_call", closes: ["</function_call>", "</function_call_name>"] },
  { open: "<invoke", closes: ["</invoke>"] },
  { open: "<tool_use", closes: ["</tool_use>"] },
  { open: "<apply_patch", closes: ["</apply_patch>"] }
]);

function isInsideCodeFence(state, prefix) {
  const combined = `${state.emittedText}${prefix}`;
  return (combined.match(/```/g)?.length ?? 0) % 2 === 1;
}

function findPartialToolTagStart(text, capturePairs) {
  const lastIndex = text.lastIndexOf("<");
  if (lastIndex < 0 || text.slice(lastIndex).includes(">")) {
    return -1;
  }

  const tail = text.slice(lastIndex).toLowerCase();
  /* Hold the partial fragment if it could grow into either:
   * - any capture-pair open/close (block-style tools)
   * - any orphan leak tag the downstream stripper would remove
   *   (`<tool_name`, `</parameter`, etc.) — without this the stream
   *   boundary can split a leak token and slip the second half past
   *   the regex-based stripper. */
  const matchesCapture = capturePairs.some(({ open, closes }) =>
    open.startsWith(tail) || closes.some(c => c.startsWith(tail))
  );
  if (matchesCapture) return lastIndex;

  return LEAK_TAG_PREFIXES.some(p => p.toLowerCase().startsWith(tail))
    ? lastIndex
    : -1;
}

function findToolSegmentStart(state, text, capturePairs) {
  const lower = text.toLowerCase();
  let offset = 0;

  while (offset < lower.length) {
    let bestIndex = -1;
    let matchedOpen = "";

    for (const { open } of capturePairs) {
      const index = lower.indexOf(open, offset);
      if (index >= 0 && (bestIndex === -1 || index < bestIndex)) {
        bestIndex = index;
        matchedOpen = open;
      }
    }

    if (bestIndex === -1) {
      return -1;
    }

    if (!isInsideCodeFence(state, text.slice(0, bestIndex))) {
      return bestIndex;
    }

    offset = bestIndex + matchedOpen.length;
  }

  return -1;
}

function splitSafeContent(state, text, capturePairs) {
  const partialStart = findPartialToolTagStart(text, capturePairs);
  if (partialStart < 0 || isInsideCodeFence(state, text.slice(0, partialStart))) {
    return { safe: text, hold: "" };
  }

  return { safe: text.slice(0, partialStart), hold: text.slice(partialStart) };
}

function consumeCapturedToolBlock(captured, allowedToolNames, capturePairs) {
  const lower = captured.toLowerCase();

  for (const pair of capturePairs) {
    const openIndex = lower.indexOf(pair.open);
    if (openIndex < 0) {
      continue;
    }

    let closeIndex = -1;
    let chosenClose = pair.closes[0];
    for (const closeStr of pair.closes) {
      const idx = lower.lastIndexOf(closeStr);
      if (idx > closeIndex) {
        closeIndex = idx;
        chosenClose = closeStr;
      }
    }

    if (closeIndex < openIndex) {
      return { ready: false };
    }

    const closeEnd = closeIndex + chosenClose.length;
    const block = captured.slice(openIndex, closeEnd);
    const calls = parseToolCallsFromText(block, allowedToolNames);
    return {
      ready: true,
      prefix: captured.slice(0, openIndex),
      calls,
      suffix: captured.slice(closeEnd)
    };
  }

  return { ready: true, prefix: captured, calls: [], suffix: "" };
}

function pushTextEvent(state, events, text, kind) {
  if (!text) {
    return;
  }

  state.emittedText += text;
  if (state.emittedText.length > 100000) {
    log.warn("sieve", `[pushTextEvent] emittedText length ${state.emittedText.length} exceeds limit, truncating to last 50000 chars`);
    state.emittedText = state.emittedText.slice(-50000);
  }
  events.push({ type: "text", text, kind: kind ?? state.lastKind ?? "response" });
}

export function createToolSieve(allowedToolNames = []) {
  /* Tool-name-specific tags let the sieve also detect bare <ApplyPatch>...
   * style outputs (model omits the <tool_calls> wrapper).  Skip names that
   * already collide with a generic open tag. */
  const capturePairs = [...TOOL_CAPTURE_PAIRS];
  for (const name of (allowedToolNames ?? [])) {
    const lower = name.toLowerCase();
    if (!TOOL_CAPTURE_PAIRS.some(p => lower === p.open.slice(1))) {
      capturePairs.push({ open: `<${lower}`, closes: [`</${lower}>`] });
    }
  }

  const state = {
    allowedToolNames,
    capture: "",
    capturing: false,
    emittedText: "",
    pending: "",
    lastKind: "response"
  };

  function drain() {
    const events = [];

    while (true) {
      if (state.capturing) {
        if (state.pending) {
          state.capture += state.pending;
          state.pending = "";
        }

        if (state.capture.length > 50000) {
          log.warn("sieve", `[drain] Capture length ${state.capture.length} exceeds limit, forcing flush as text`);
          pushTextEvent(state, events, state.capture, state.lastKind);
          state.capture = "";
          state.capturing = false;
          continue;
        }

        const consumed = consumeCapturedToolBlock(state.capture, state.allowedToolNames, capturePairs);
        if (!consumed.ready) {
          break;
        }

        state.capture = "";
        state.capturing = false;
        pushTextEvent(state, events, consumed.prefix ?? "", state.lastKind);
        if (consumed.calls?.length) {
          events.push({ type: "tool_calls", calls: consumed.calls });
        }
        state.pending = `${consumed.suffix ?? ""}${state.pending}`;
        continue;
      }

      if (!state.pending) {
        break;
      }

      const start = findToolSegmentStart(state, state.pending, capturePairs);
      if (start >= 0) {
        pushTextEvent(state, events, state.pending.slice(0, start), state.lastKind);
        state.capture = state.pending.slice(start);
        state.pending = "";
        state.capturing = true;
        continue;
      }

      const { safe, hold } = splitSafeContent(state, state.pending, capturePairs);
      state.pending = hold;
      pushTextEvent(state, events, safe, state.lastKind);
      break;
    }

    return events;
  }

  return Object.freeze({
    flush() {
      const events = drain();

      if (state.capturing) {
        if (state.capture.length > 50000) {
          log.warn("sieve", `[flush] Capture length ${state.capture.length} exceeds limit, forcing flush as text`);
          pushTextEvent(state, events, state.capture, state.lastKind);
          state.capture = "";
          state.capturing = false;
        } else {
          const consumed = consumeCapturedToolBlock(state.capture, state.allowedToolNames, capturePairs);
          if (consumed.ready) {
            pushTextEvent(state, events, consumed.prefix ?? "", state.lastKind);
            if (consumed.calls?.length) {
              events.push({ type: "tool_calls", calls: consumed.calls });
            }
            pushTextEvent(state, events, consumed.suffix ?? "", state.lastKind);
          } else {
            /* Stream ended mid-capture.  Try a synthetic close so the parser
             * has a chance to extract calls; if it can't, drop the capture
             * (emitting raw XML pollutes the conversation). */
            const lowerCapture = state.capture.toLowerCase();
            let syntheticClose = "</tool_calls>";
            for (const pair of capturePairs) {
              if (lowerCapture.indexOf(pair.open) >= 0) {
                syntheticClose = pair.closes[0];
                break;
              }
            }
            const syntheticBlock = `${state.capture}${syntheticClose}`;
            const partialCalls = parseToolCallsFromText(syntheticBlock, state.allowedToolNames);
            if (partialCalls.length) {
              events.push({ type: "tool_calls", calls: partialCalls });
            } else {
              log.debug("sieve", `[flush] Dropping ${state.capture.length} chars of unparseable capture`);
            }
          }
        }
      }

      pushTextEvent(state, events, state.pending, state.lastKind);
      state.capture = "";
      state.capturing = false;
      state.pending = "";
      return events;
    },
    push(chunk, kind) {
      const text = toStringSafe(chunk);
      if (kind) {
        state.lastKind = kind;
      }
      state.pending += text;
      return drain();
    },
    get captureLength() {
      return state.capture.length;
    },
    get pendingLength() {
      return state.pending.length;
    },
    get isCapturing() {
      return state.capturing;
    },
    get emittedText() {
      return state.emittedText;
    }
  });
}

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
  const content = events
    .filter((event) => event.type === "text")
    .map((event) => stripLeakedMarkers(event.text))
    .join("")
    .trimStart();
  return {
    events,
    content,
    toolCalls: events.flatMap((event) => event.type === "tool_calls" ? event.calls ?? [] : [])
  };
}

import { parseToolCallsFromText } from "./openai-tool-parser.js";
import { log } from "../utils/log.js";

const TOOL_CAPTURE_PAIRS = Object.freeze([
  { open: "<tool_calls", close: "</tool_calls>" },
  { open: "<function_calls", close: "</function_calls>" },
  { open: "<tool_call", close: "</tool_call>" },
  { open: "<function_call", close: "</function_call>" },
  { open: "<invoke", close: "</invoke>" },
  { open: "<tool_use", close: "</tool_use>" },
  { open: "<apply_patch", close: "</apply_patch>" }
]);

function isInsideCodeFence(state, prefix) {
  const combined = `${state.emittedText}${prefix}`;
  return (combined.match(/```/g)?.length ?? 0) % 2 === 1;
}

function findPartialToolTagStart(text, capturePairs = TOOL_CAPTURE_PAIRS) {
  const lastIndex = text.lastIndexOf("<");
  if (lastIndex < 0 || text.slice(lastIndex).includes(">")) {
    return -1;
  }

  const tail = text.slice(lastIndex).toLowerCase();
  return capturePairs.some(({ open }) => open.startsWith(tail)) ? lastIndex : -1;
}

function findToolSegmentStart(state, text, capturePairs = TOOL_CAPTURE_PAIRS) {
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

function splitSafeContent(state, text, capturePairs = TOOL_CAPTURE_PAIRS) {
  const partialStart = findPartialToolTagStart(text, capturePairs);
  if (partialStart < 0 || isInsideCodeFence(state, text.slice(0, partialStart))) {
    return { safe: text, hold: "" };
  }

  return { safe: text.slice(0, partialStart), hold: text.slice(partialStart) };
}

function consumeCapturedToolBlock(captured, allowedToolNames, incompleteCount, capturePairs = TOOL_CAPTURE_PAIRS) {
  const lower = captured.toLowerCase();

  for (const pair of capturePairs) {
    const openIndex = lower.indexOf(pair.open);
    if (openIndex < 0) {
      continue;
    }

    const closeIndex = lower.lastIndexOf(pair.close);
    if (closeIndex < openIndex) {
      // Only log every Nth incomplete check to reduce noise
      // Always log first occurrence and then periodically
      const logInterval = Math.max(20, Math.floor(captured.length / 50));
      if (incompleteCount <= 1 || incompleteCount % logInterval === 0) {
        log.debug("sieve", `[consume] Still waiting for </${pair.close}> (checked ${incompleteCount}x, captured ${captured.length} chars)`);
      }
      return { ready: false };
    }

    const closeEnd = closeIndex + pair.close.length;
    const block = captured.slice(openIndex, closeEnd);
    log.debug("sieve", `[consume] Complete block captured: <${pair.open}>...</${pair.close}>, block length=${block.length}, preview="${block.slice(0, 200)}"`);
    const calls = parseToolCallsFromText(block, allowedToolNames);
    log.debug("sieve", `[consume] Parsed ${calls.length} tool call(s) from captured block`);
    return {
      ready: true,
      prefix: captured.slice(0, openIndex),
      calls,
      suffix: captured.slice(closeEnd)
    };
  }

  return { ready: true, prefix: captured, calls: [], suffix: "" };
}

function pushTextEvent(state, events, text) {
  if (!text) {
    return;
  }

  state.emittedText += text;
  events.push({ type: "text", text });
}

export function createToolSieve(allowedToolNames = []) {
  // Build extended capture pairs including tool-name-specific tags
  // E.g. if ApplyPatch is in allowedToolNames, add { open: "<applypatch", close: "</applypatch>" }
  // This allows the sieve to detect when the model outputs <ApplyPatch> instead of <tool_calls>
  const capturePairs = [...TOOL_CAPTURE_PAIRS];
  for (const name of (allowedToolNames ?? [])) {
    const lower = name.toLowerCase();
    // Skip if already covered by a generic pattern (e.g. <tool> covers <tool>)
    if (!TOOL_CAPTURE_PAIRS.some(p => lower === p.open.slice(1))) {
      capturePairs.push({ open: `<${lower}`, close: `</${lower}>` });
    }
  }

  log.debug("sieve", `[createToolSieve] Capture pairs: [${capturePairs.map(p => p.open).join(", ")}] (from tools: [${(allowedToolNames ?? []).join(", ")}])`);

  const state = {
    allowedToolNames,
    capture: "",
    capturing: false,
    emittedText: "",
    pending: "",
    incompleteCount: 0
  };

  function drain() {
    const events = [];

    while (true) {
      if (state.capturing) {
        if (state.pending) {
          state.capture += state.pending;
          state.pending = "";
          state.incompleteCount++;
        }

        const consumed = consumeCapturedToolBlock(state.capture, state.allowedToolNames, state.incompleteCount, capturePairs);
        if (!consumed.ready) {
          break;
        }

        if (state.incompleteCount > 1) {
          log.debug("sieve", `[consume] Block completed after ${state.incompleteCount} checks, ${state.capture.length} chars captured`);
        }
        state.incompleteCount = 0;

        state.capture = "";
        state.capturing = false;
        pushTextEvent(state, events, consumed.prefix ?? "");
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
        log.debug("sieve", `[drain] Tool segment start detected at offset ${start}, entering capture mode`);
        pushTextEvent(state, events, state.pending.slice(0, start));
        state.capture = state.pending.slice(start);
        state.pending = "";
        state.capturing = true;
        continue;
      }

      const { safe, hold } = splitSafeContent(state, state.pending, capturePairs);
      state.pending = hold;
      pushTextEvent(state, events, safe);
      break;
    }

    return events;
  }

  return Object.freeze({
    flush() {
      const events = drain();

      if (state.capturing) {
        const consumed = consumeCapturedToolBlock(state.capture, state.allowedToolNames, 0, capturePairs);
        if (consumed.ready) {
          pushTextEvent(state, events, consumed.prefix ?? "");
          if (consumed.calls?.length) {
            events.push({ type: "tool_calls", calls: consumed.calls });
          }
          pushTextEvent(state, events, consumed.suffix ?? "");
        } else {
          // Stream ended without the closing tag — try to parse the partial capture
          // by appending a synthetic closing tag and attempting to extract tool calls
          log.debug("sieve", `[flush] Stream ended without closing tag, attempting partial capture parse (${state.capture.length} chars captured)`);
          // Detect which capture pair was matched and use its close tag for the synthetic block
          const lowerCapture = state.capture.toLowerCase();
          let syntheticClose = "</tool_calls>";
          for (const pair of capturePairs) {
            if (lowerCapture.indexOf(pair.open) >= 0) {
              syntheticClose = pair.close;
              break;
            }
          }
          const syntheticBlock = `${state.capture}${syntheticClose}`;
          const partialCalls = parseToolCallsFromText(syntheticBlock, state.allowedToolNames);
          if (partialCalls.length) {
            log.debug("sieve", `[flush] Partial capture parse succeeded: found ${partialCalls.length} tool call(s)`);
            events.push({ type: "tool_calls", calls: partialCalls });
          } else {
            // No tool calls could be extracted — emit as plain text
            log.debug("sieve", `[flush] Partial capture parse found no tool calls, emitting as text`);
            pushTextEvent(state, events, state.capture);
          }
        }
      }

      pushTextEvent(state, events, state.pending);
      state.capture = "";
      state.capturing = false;
      state.pending = "";
      return events;
    },
    push(chunk) {
      state.pending += typeof chunk === "string" ? chunk : String(chunk ?? "");
      return drain();
    },
    get captureLength() {
      return state.capture.length;
    },
    get isCapturing() {
      return state.capturing;
    },
    get emittedText() {
      return state.emittedText;
    }
  });
}

function toTextEvent(chunk) {
  return { type: "text", text: typeof chunk === "string" ? chunk : String(chunk ?? "") };
}

function flattenToolEvents(events) {
  return events.reduce((output, event) => {
    if (!output.length || event.type !== "text" || output.at(-1).type !== "text") {
      output.push(event);
      return output;
    }

    output[output.length - 1] = {
      type: "text",
      text: `${output.at(-1).text}${event.text}`
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
  return {
    events,
    content: events
      .filter((event) => event.type === "text")
      .map((event) => event.text)
      .join(""),
    toolCalls: events.flatMap((event) => event.type === "tool_calls" ? event.calls ?? [] : [])
  };
}

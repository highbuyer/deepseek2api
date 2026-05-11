// Shared tool result truncation helpers — used by both OpenAI and Anthropic prompt builders.
// Extracted from openai-tool-prompt.js / anthropic-prompt.js to eliminate ~100 lines of duplication.

import { log } from "./log.js";

/**
 * Rough token estimator: CJK characters ≈ 1 token, everything else ≈ 0.25 token.
 * DeepSeek V4 expert backend measured 2026-05-08:
 *   150K chars of repeated English → 34,659 real tokens (ratio ~0.231)
 *   162K chars of repeated English → 37,429 real tokens (ratio ~0.231)
 * 0.25 is a slightly conservative midpoint for English/code; CJK stays at 1:1.
 *
 * This is an APPROXIMATION. Real ratio varies by content type (JSON/URLs/code may
 * differ ±20%). The goal is to avoid gross under/over-estimation, not perfect accuracy.
 */
export function estimateTokens(text) {
  let cjk = 0;
  let other = 0;
  for (const ch of text) {
    const cp = ch.codePointAt(0);
    // CJK Unified Ideographs + Extensions + Compatibility + Supplement
    // CJK Radicals / Strokes / Symbols / Punctuation
    // Hiragana / Katakana / Hangul
    // Fullwidth forms
    if ((cp >= 0x4E00 && cp <= 0x9FFF) ||   // CJK Unified
        (cp >= 0x3400 && cp <= 0x4DBF) ||   // CJK Ext-A
        (cp >= 0x20000 && cp <= 0x2A6DF) || // CJK Ext-B
        (cp >= 0x2A700 && cp <= 0x2B73F) || // CJK Ext-C
        (cp >= 0x2B740 && cp <= 0x2B81F) || // CJK Ext-D
        (cp >= 0x2B820 && cp <= 0x2CEAF) || // CJK Ext-E
        (cp >= 0x2CEB0 && cp <= 0x2EBEF) || // CJK Ext-F
        (cp >= 0x30000 && cp <= 0x3134F) || // CJK Ext-G
        (cp >= 0x31350 && cp <= 0x323AF) || // CJK Ext-H
        (cp >= 0xF900 && cp <= 0xFAFF) ||   // CJK Compat
        (cp >= 0x2F800 && cp <= 0x2FA1F) || // CJK Compat Supplement
        (cp >= 0x2E80 && cp <= 0x2EFF) ||   // CJK Radicals
        (cp >= 0x3000 && cp <= 0x303F) ||   // CJK Symbols
        (cp >= 0xFF00 && cp <= 0xFFEF) ||   // Halfwidth/Fullwidth
        (cp >= 0x3040 && cp <= 0x309F) ||   // Hiragana
        (cp >= 0x30A0 && cp <= 0x30FF) ||   // Katakana
        (cp >= 0xAC00 && cp <= 0xD7AF)) {   // Hangul
      cjk++;
    } else {
      other++;
    }
  }
  return Math.ceil(cjk + other * 0.25);
}

export function getToolResultBudget(totalContextChars, modelLimit = 128000) {
  if (totalContextChars > modelLimit * 0.9) return 8000;
  if (totalContextChars > modelLimit * 0.6) return 12000;
  if (totalContextChars > modelLimit * 0.3) return 16000;
  return 20000;
}

export function getToolTruncationCategory(toolName) {
  const lower = (toolName || "").toLowerCase();
  if (/^(read|read_file|readfile)$/i.test(lower)) return "read";
  if (/^(bash|shell|execute_command|runcommand|run_command|terminal)$/i.test(lower)) return "bash";
  if (/^(search|grep|find|glob|list)$/i.test(lower)) return "search";
  if (/^(write|edit|apply_patch|applypatch|patch)$/i.test(lower)) return "write";
  return "default";
}

/**
 * Web content tools (browser snapshot, web fetch, web search) produce
 * HTML/XML-laden output that pollutes the prompt with angle brackets.
 * When the model sees these in context, it confuses JSON fence format
 * with XML format, causing degenerate tool call output.
 */
export function isWebContentTool(toolName) {
  const lower = (toolName || "").toLowerCase();
  return /browser_|cursor-ide-browser|webfetch|websearch|web_fetch|web_search|fetch_url/i.test(lower);
}

/**
 * Budget for web content tool results — much tighter than the default
 * 20000 chars to minimize angle-bracket pollution in the prompt.
 */
export const WEB_TOOL_RESULT_BUDGET = 4000;

/**
 * Sanitize web content to prevent angle bracket leakage into the prompt.
 * Replaces < and > with fullwidth angle brackets so the content doesn't
 * look like tool call XML tags to the model.
 */
export function sanitizeWebContent(content) {
  return content.replace(/</g, "＜").replace(/>/g, "＞");
}

export const TOOL_TRUNCATION_STRATEGY = {
  read:    { headRatio: 0.3, tailRatio: 0.3 },
  bash:    { headRatio: 0.2, tailRatio: 0.6 },
  search:  { headRatio: 0.7, tailRatio: 0.15 },
  write:   { headRatio: 0.5, tailRatio: 0.3 },
  default: { headRatio: 0.6, tailRatio: 0.4 }
};

export function formatTruncationHint(contentLen, headBudget, tailBudget, omitted, toolName, lineInfo) {
  const cat = getToolTruncationCategory(toolName);
  const pct = Math.round(omitted / contentLen * 100);
  let hint = "\n\n⚠️ TRUNCATED — ";

  // Directive first: tell the model what to do NOW, not what happened.
  // Models that see "DO NOT" or a problem description tend to complain
  // instead of acting.  Leading with a concrete action fixes that.
  if (lineInfo && lineInfo.omittedLines > 0) {
    hint += `Read with offset=${lineInfo.omittedStart} limit=${lineInfo.omittedLines}` +
      ` to fetch lines ${lineInfo.omittedStart}-${lineInfo.omittedEnd}`;
    if (lineInfo.omittedLines > 500) {
      hint += " (split into smaller offset/limit chunks if needed)";
    }
    hint += ".\n";
  } else {
    hint += `Read with offset/limit or use Grep to fetch the omitted section.\n`;
  }

  hint += `(${pct}% omitted — ${omitted} of ${contentLen} chars, showing head ${headBudget} + tail ${tailBudget}${toolName ? `, ${cat} strategy` : ""})\n`;
  hint += "Grep for patterns before re-reading entire files.\n\n";

  return hint;
}

/**
 * Scan assistant message content for Read tool invocations and build:
 * - toolCallIdToPath: maps tool_call_id → file path
 * - pathReadCount: how many times each path was read
 *
 * Used to detect repeated reads of the same file within a session.
 * OpenAI path: scans XML <invoke name="Read"> blocks.
 * Anthropic path: scans content block arrays with tool_use type.
 *
 * Callers check pathReadCount > 1 for a given path and inject
 * "[Already read in this session — reference earlier result]" warnings.
 */
export function buildReadPathTracker(openaiMessages) {
  const toolCallIdToPath = new Map();
  const pathReadCount = new Map();

  for (const msg of openaiMessages) {
    if (msg.role !== "assistant") continue;
    const content = msg.content ?? "";

    // Anthropic content-block format
    if (Array.isArray(msg._rawContent)) {
      for (const block of msg._rawContent) {
        if (block?.type === "tool_use" && /^read$/i.test(block.name)) {
          const path = block.input?.file_path || block.input?.path || "";
          if (path) {
            toolCallIdToPath.set(block.id, path);
            pathReadCount.set(path, (pathReadCount.get(path) || 0) + 1);
          }
        }
      }
      continue;
    }

    // OpenAI XML format: <invoke name="Read">...</invoke>
    for (const m of content.matchAll(/<invoke\s+name="Read"[^>]*>([\s\S]*?)<\/invoke>/gi)) {
      const inner = m[1];
      const pathMatch = inner.match(/<parameter\s+name="(?:file_path|path)"[^>]*>([^<]+)<\/parameter>/i);
      if (pathMatch) {
        const path = pathMatch[1].trim();
        // Generate a synthetic call ID from path + position
        const callId = `read:${path}:${m.index}`;
        toolCallIdToPath.set(callId, path);
        pathReadCount.set(path, (pathReadCount.get(path) || 0) + 1);
      }
    }
  }

  return { toolCallIdToPath, pathReadCount };
}

export function formatRepeatedReadWarning(path, count) {
  return `\n\n⚠️ REPEATED READ #${count}: "${path}" has already been read in this session.\nReference the earlier result instead of re-reading the same file.\nIf you need a specific section, use Read with offset/limit.\n\n`;
}

export function truncateToolResult(content, budget, toolName) {
  if (content.length <= budget) return { text: content, truncated: false };

  const cat = getToolTruncationCategory(toolName);
  const strategy = TOOL_TRUNCATION_STRATEGY[cat] || TOOL_TRUNCATION_STRATEGY.default;
  const headBudget = Math.floor(budget * strategy.headRatio);
  const tailBudget = Math.floor(budget * strategy.tailRatio);
  const omitted = content.length - headBudget - tailBudget;

  // For read category, provide line-number hints so the model knows
  // exactly which lines to fetch with offset/limit.
  // Align head/tail cuts to line boundaries so line counts are accurate.
  let headEnd = headBudget;
  let tailStart = content.length - tailBudget;
  let lineInfo;
  if (cat === "read") {
    // Align head end to next line boundary (don't cut mid-line)
    const headBoundary = content.indexOf("\n", headBudget);
    if (headBoundary > 0 && headBoundary < content.length - tailBudget) {
      headEnd = headBoundary + 1; // include the newline
    } else {
      const lastHeadBoundary = content.lastIndexOf("\n", headBudget);
      if (lastHeadBoundary > 0) headEnd = lastHeadBoundary + 1;
    }
    // Align tail start to previous line boundary
    const tailBoundary = content.lastIndexOf("\n", content.length - tailBudget);
    if (tailBoundary > headEnd) {
      tailStart = tailBoundary + 1; // start after the newline
    }

    // Guard against alignment pushing head past tail (very long lines)
    if (headEnd >= tailStart) {
      headEnd = headBudget;
      tailStart = content.length - tailBudget;
    }

    const headLines = content.slice(0, headEnd).split("\n").length;
    const tailLines = content.slice(tailStart).split("\n").length;
    const totalLines = content.split("\n").length;
    const omittedStart = headLines + 1;
    const omittedEnd = totalLines - tailLines;
    lineInfo = { omittedStart, omittedEnd, omittedLines: Math.max(0, omittedEnd - omittedStart + 1) };
  }

  const actualOmitted = Math.max(0, tailStart - headEnd);
  const text = content.slice(0, headEnd)
    + formatTruncationHint(content.length, headEnd, content.length - tailStart, actualOmitted, toolName, lineInfo)
    + content.slice(tailStart);

  return { text, truncated: true };
}

export const PER_MESSAGE_TOOL_RESULT_BUDGET = 100000;
export const BUDGET_PREVIEW_CHARS = 500;

export function buildBudgetPreview(content, originalSize, toolName = "") {
  const preview = content.slice(0, BUDGET_PREVIEW_CHARS);
  const sizeKB = Math.round(originalSize / 1024);
  const nameHint = toolName ? ` (${toolName})` : "";
  return (
    `Output too large${nameHint} (${sizeKB} KB) — replaced to stay within context budget.\n` +
    `Use Grep to search within this file, or Read with offset/limit.\n\n` +
    `Preview (first ${BUDGET_PREVIEW_CHARS} chars):\n` +
    preview +
    (originalSize > BUDGET_PREVIEW_CHARS ? "\n[...]" : "")
  );
}

/**
 * Group messages into "rounds". A round starts with a USER message and
 * includes all following assistant/tool messages until the next USER.
 *
 * This mirrors Claude Code's groupMessagesByApiRound pattern:
 * when truncating, we drop whole rounds rather than splitting tool_call/tool_result pairs.
 *
 * Returns array of { messages, hasToolCalls, toolCallIds } objects.
 */
function groupMessagesByRound(msgs) {
  const rounds = [];
  let current = null;

  for (const msg of msgs) {
    if (msg.role === "user" || msg.role === "system") {
      current = { messages: [], hasToolCalls: false, toolCallIds: new Set() };
      rounds.push(current);
    } else if (!current) {
      current = { messages: [], hasToolCalls: false, toolCallIds: new Set() };
      rounds.push(current);
    }

    current.messages.push(msg);

    if (msg.role === "assistant") {
      const tcXml = msg.content ?? "";
      if (tcXml.includes("<tool_calls>") || tcXml.includes("<invoke name=") || /\[\s*\{\s*"(?:tool|name)"\s*:/.test(tcXml)) {
        current.hasToolCalls = true;
      }
    }

    if (msg.role === "tool") {
      const m = (msg.content ?? "").match(/<tool_result\s+id="([^"]+)"/);
      if (m) current.toolCallIds.add(m[1]);
    }
  }

  return rounds;
}

/**
 * Render messages to a prompt block string for budget calculation.
 * `measure` defaults to s.length (char count); pass estimateTokens for token-aware mode.
 */
function renderMessagesBlock(msgs, prependSeparator, measure) {
  const parts = msgs.map(m => `${m.role.toUpperCase()}: ${m.content ?? ""}`);
  const body = parts.join("\n\n");
  const text = prependSeparator ? "\n\n" + body : body;
  return { text, len: measure ? measure(text) : text.length };
}

/**
 * Fit as many messages as possible into a budget, working from newest
 * to oldest. When the boundary cuts through a round, preserves tool_call/tool_result
 * pairing — if an assistant message with tool_calls is included, all its tool_result
 * messages in the same round are also included (or all are dropped).
 *
 * Pattern from Claude Code's adjustIndexToPreserveAPIInvariants.
 */
function fitMessagesInBudget(msgs, budget, prependSeparator, preserveFirstUser, measure) {
  const kept = [];
  let used = 0;
  let i = msgs.length - 1;
  const len = measure || (s => s.length);

  while (i >= 0) {
    const msg = msgs[i];
    const block = `${msg.role.toUpperCase()}: ${msg.content ?? ""}`;
    const separator = kept.length > 0 ? "\n\n" : "";
    const entry = separator + block;
    const entryCost = len(entry);

    if (used + entryCost <= budget) {
      kept.unshift(msg);
      used += entryCost;
      i--;
      continue;
    }

    // Can't fit whole message — try truncated version
    const header = separator + `${msg.role.toUpperCase()}: `;
    const headerCost = len(header);
    const contentBudget = budget - used - headerCost;
    const minContentBudget = 200;
    if (contentBudget > minContentBudget) {
      // If this is the first user message, use lower minimum to squeeze it in
      const isFirstUser = preserveFirstUser && i === 0 && msg.role === "user";
      const minBudget = isFirstUser ? 50 : minContentBudget;
      if (contentBudget > minBudget) {
        // Slice in char space (approximate for token mode — close enough for truncated tails)
        const truncated = {
          ...msg,
          content: (msg.content ?? "").slice(0, contentBudget) + "\n...[truncated]"
        };
        kept.unshift(truncated);
      }
    }
    break;
  }

  // Repair: strip orphaned tool results whose tool_call was dropped
  return repairOrphanedToolResults(kept);
}

/**
 * Keep the most recent messages within a budget, preserving
 * tool_call/tool_result pairings. Fills from newest round backwards;
 * the oldest round is the first to be partially or fully dropped.
 *
 * Pattern from Claude Code's auto-compact and session-memory-compact:
 * - Recent context is more valuable than old context
 * - Tool call/result pairs must never be split
 * - The initial user query gets a lightweight preservation preference
 *
 * `measure` defaults to char count; pass estimateTokens for token-aware mode.
 * `budget` units must match the measure function (chars or tokens).
 */
export function keepRecentMessages(msgs, budget, measure) {
  if (!msgs.length) return [];

  const len = measure || (s => s.length);
  const rounds = groupMessagesByRound(msgs);
  const allKept = [];
  let used = 0;

  // Fill from newest round backward (Claude Code pattern: keep most recent)
  for (let r = rounds.length - 1; r >= 0; r--) {
    const roundMsgs = [...rounds[r].messages];
    const isFirstRound = r === 0;
    const prepend = allKept.length > 0;
    const { text: roundText, len: roundLen } = renderMessagesBlock(roundMsgs, prepend, measure);

    if (used + roundLen <= budget) {
      // Whole round fits
      used += roundLen;
      allKept.unshift(...roundMsgs);
      continue;
    }

    // Round doesn't fit whole — try partial fit
    const remaining = budget - used;
    const partial = fitMessagesInBudget(roundMsgs, remaining, prepend, isFirstRound, measure);
    if (partial.length > 0) {
      allKept.unshift(...partial);
    }
    // Budget exhausted for older rounds
    break;
  }

  // Final repair pass (ensureToolResultPairing pattern)
  return repairOrphanedToolResults(allKept);
}

/**
 * Repair orphaned tool results after truncation.
 *
 * Pattern from Claude Code's ensureToolResultPairing (messages.ts:5133):
 * - If a tool_result references a tool_use that was dropped, strip the orphaned result
 * - If a tool_use has no matching tool_result, insert a synthetic error placeholder
 *
 * In our XML-based format, tool_call IDs aren't explicit — tool results reference
 * tools by name in <tool_result id="Name">. We track which tool names appear in
 * kept assistant messages and drop tool_result blocks for tool names that don't
 * appear in any kept assistant message.
 */
/**
 * Check if Read tool results are bloating the context and inject a reminder
 * telling the model to stop re-reading entire files.
 *
 * Pattern from Claude Code's checkReadResultBloat: when Read results dominate
 * the prompt, remind the model to use Grep or offset/limit instead.
 *
 * Injects at the tail of the LAST user message for maximum attention freshness,
 * rather than appending to system prompt (which gets diluted).
 */
export function injectReadBloatHint(promptMessages) {
  let readResultChars = 0;
  let totalChars = 0;
  for (const msg of promptMessages) {
    totalChars += (msg.content ?? "").length;
    if (msg.role === "tool") {
      const m = (msg.content ?? "").match(/<tool_result id="(read|read_file|readfile)"/i);
      if (m) readResultChars += msg.content.length;
    }
  }

  const readPct = totalChars > 0 ? (readResultChars / totalChars) * 100 : 0;
  // Claude Code pattern: absolute threshold OR proportional threshold (not AND)
  if (readResultChars < 25000 && readPct < 10) return promptMessages;

  const hint = [
    "<system-reminder>",
    `File read results are using ${Math.round(readPct)}% of the context window.`,
    "If you are re-reading files, reference earlier reads instead.",
    "For large files, use Grep to search for patterns or Read with offset/limit to fetch specific sections.",
    "Avoid reading entire files unless you need the full content.",
    "</system-reminder>"
  ].join("\n");

  log.warn("prompt", `Read results at ${Math.round(readPct)}% of context (${readResultChars}/${totalChars} chars), injecting bloat hint`);

  // Inject at tail of last user message for attention freshness (Claude Code pattern)
  for (let i = promptMessages.length - 1; i >= 0; i--) {
    if (promptMessages[i].role === "user") {
      const updated = [...promptMessages];
      updated[i] = { ...updated[i], content: updated[i].content + "\n\n" + hint };
      return updated;
    }
  }

  // Fallback: no user message found, prepend as synthetic system message
  return [{ role: "system", content: hint }, ...promptMessages];
}

// ── Loop circuit breaker ───────────────────────────────────────────────
// Detects when the model repeatedly calls the same tool with the same
// arguments and gets the same error back.  After THRESHOLD consecutive
// identical-signature failures, returns a system interruption message
// that forces the model to stop retrying and switch strategy.
//
// This prevents the "350K prompt from 56 dead rounds" scenario:
// the loop is broken at round 3 instead of round 56.

const FAIL_LOOP_THRESHOLD = 3;

function buildLoopInterruption(toolName, streakCount) {
  return [
    "<system-reminder>",
    `STOP: "${toolName}" with these arguments has failed ${streakCount} times in a row.`,
    `Do NOT call "${toolName}" with these parameters again.`,
    "Switch approach: try a different tool, change arguments, or answer with what you have.",
    "</system-reminder>"
  ].join("\n");
}

function simpleHash(str) {
  let hash = 5381;
  for (let i = 0; i < str.length; i++) {
    hash = ((hash << 5) + hash) + str.charCodeAt(i);
    hash = hash & hash;
  }
  return Math.abs(hash).toString(36);
}

function toolCallSignature(tc) {
  const name = tc?.function?.name || tc?.name || "?";
  let args = {};
  try {
    args = JSON.parse(tc?.function?.arguments || tc?.input || "{}");
  } catch { /* malformed JSON */ }
  const normalized = JSON.stringify(args, Object.keys(args).sort());
  return `${name}:${simpleHash(normalized)}`;
}

function isFailedToolResult(content) {
  if (!content) return false;
  const text = typeof content === "string" ? content.slice(0, 2000) : JSON.stringify(content).slice(0, 2000);
  // Exit code ≠ 0
  if (/Process\s+exited\s+with\s+code\s+[1-9]\d*/i.test(text)) return true;
  // Shell / runtime errors — high-confidence indicators
  if (/\b(?:SyntaxError|ReferenceError|TypeError|RangeError|URIError|EvalError)\b/.test(text)) return true;
  if (/(?:^|\n)\s*Error:/i.test(text)) return true;
  // Tool execution wrapper errors
  if (/Tool\s+execution\s+failed/i.test(text)) return true;
  if (/\bno\s+result\s+provided\b/i.test(text)) return true;
  // Handoff placeholder
  if (/cross[-\s]?provider\s+handoff/i.test(text)) return true;
  return false;
}

/**
 * Scan raw (pre-normalization) messages for repeated failed tool call
 * patterns.  Returns a system interruption string if detected, or null.
 *
 * Checks both OpenAI-style (tool_calls array) and Anthropic-style
 * (content blocks with tool_use/tool_result) formats.
 */
export function detectRepeatedToolFailures(rawMessages, threshold = FAIL_LOOP_THRESHOLD) {
  if (!Array.isArray(rawMessages) || rawMessages.length < threshold * 2) return null;

  // Track per-signature consecutive failure streak
  let currentSig = null;
  let currentStreak = 0;

  for (let i = 0; i < rawMessages.length; i++) {
    const msg = rawMessages[i];

    // ── Assistant message: extract tool_call signatures ──
    if (msg?.role === "assistant") {
      const sigs = [];

      // OpenAI format: msg.tool_calls array
      if (Array.isArray(msg.tool_calls)) {
        for (const tc of msg.tool_calls) {
          sigs.push(toolCallSignature(tc));
        }
      }

      // Anthropic format: content blocks
      if (Array.isArray(msg.content)) {
        for (const block of msg.content) {
          if (block?.type === "tool_use") {
            sigs.push(toolCallSignature({ name: block.name, input: block.input }));
          }
        }
      }

      if (sigs.length > 0) {
        // For multi-tool messages, use the first tool call as the primary signature
        const primarySig = sigs[0];
        if (primarySig === currentSig) {
          // Same signature as last round — streak continues (checked after tool results)
        } else {
          currentSig = primarySig;
          currentStreak = 0; // reset on different signature
        }
      }
      continue;
    }

    // ── Tool result message: check if this round failed ──
    if (msg?.role === "tool" || msg?.role === "function") {
      if (!currentSig) continue;

      if (isFailedToolResult(msg.content)) {
        currentStreak++;
      } else {
        currentStreak = 0;
      }

      if (currentStreak >= threshold) {
        return buildLoopInterruption(currentSig.split(":")[0], currentStreak);
      }
      continue;
    }

    // Anthropic format: tool results come as user messages with content blocks
    if (msg?.role === "user" && Array.isArray(msg.content)) {
      if (!currentSig) continue;

      const hasFailure = msg.content.some(
        b => b?.type === "tool_result" && isFailedToolResult(b.content)
      );

      if (hasFailure) {
        currentStreak++;
      } else if (msg.content.some(b => b?.type === "tool_result")) {
        currentStreak = 0;
      }

      if (currentStreak >= threshold) {
        return buildLoopInterruption(currentSig.split(":")[0], currentStreak);
      }
    }
  }

  return null;
}

function repairOrphanedToolResults(messages) {
  // Collect all tool names called in kept assistant messages
  const calledToolNames = new Set();
  let hasAnyToolCall = false;

  for (const msg of messages) {
    if (msg.role !== "assistant") continue;
    const content = msg.content ?? "";
    for (const m of content.matchAll(/<tool_name>([^<]+)<\/tool_name>/gi)) {
      calledToolNames.add(m[1].trim());
      hasAnyToolCall = true;
    }
    for (const m of content.matchAll(/<invoke\s+name="([^"]+)"/gi)) {
      calledToolNames.add(m[1].trim());
      hasAnyToolCall = true;
    }
  }

  // Collect tool result tool names to detect orphaned results
  const resultToolNames = new Set();
  for (const msg of messages) {
    if (msg.role !== "tool") continue;
    const m = (msg.content ?? "").match(/<tool_result\s+id="([^"]+)"/);
    if (m) resultToolNames.add(m[1].trim());
  }

  // If we have tool_results but NO matching tool_calls at all (all were dropped
  // by truncation), all tool results are orphaned and should be cleared.
  // Pattern from Claude Code's ensureToolResultPairing: orphaned results
  // are replaced with placeholder text rather than keeping stale content.
  if (resultToolNames.size > 0 && (!hasAnyToolCall || [...resultToolNames].every(n => !calledToolNames.has(n)))) {
    const result = [];
    let orphanedCount = 0;
    for (const msg of messages) {
      if (msg.role === "tool") {
        const m = (msg.content ?? "").match(/<tool_result\s+id="([^"]+)"/);
        if (m && !calledToolNames.has(m[1].trim())) {
          orphanedCount++;
          result.push({
            ...msg,
            content: `<tool_result id="${m[1].trim()}">\n[Tool result cleared — earlier context truncated]\n</tool_result>`
          });
          continue;
        }
      }
      result.push(msg);
    }
    return result;
  }

  return messages;
}

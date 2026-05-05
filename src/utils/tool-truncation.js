// Shared tool result truncation helpers — used by both OpenAI and Anthropic prompt builders.
// Extracted from openai-tool-prompt.js / anthropic-prompt.js to eliminate ~100 lines of duplication.

export function getToolResultBudget(totalContextChars) {
  if (totalContextChars > 120000) return 8000;
  if (totalContextChars > 80000) return 12000;
  if (totalContextChars > 40000) return 16000;
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

export const TOOL_TRUNCATION_STRATEGY = {
  read:    { headRatio: 0.5, tailRatio: 0.3 },
  bash:    { headRatio: 0.2, tailRatio: 0.6 },
  search:  { headRatio: 0.7, tailRatio: 0.15 },
  write:   { headRatio: 0.5, tailRatio: 0.3 },
  default: { headRatio: 0.6, tailRatio: 0.4 }
};

export function formatTruncationHint(contentLen, headBudget, tailBudget, omitted, toolName) {
  const cat = getToolTruncationCategory(toolName);
  const label = toolName ? ` (${cat} strategy for ${toolName})` : "";
  return (
    `\n\n⚠️ FILE TOO LARGE — ${omitted} chars omitted (${Math.round(omitted / contentLen * 100)}% of ${contentLen} total).\n` +
    `Showing first ${headBudget} + last ${tailBudget} chars${label}.\n\n` +
    `DO NOT read this file again without offset/limit. Instead:\n` +
    `- Read with offset and limit parameters to fetch specific sections\n` +
    `- Grep for patterns to find exactly what you need before reading\n` +
    `- The truncated content above shows the file structure; search it first\n\n`
  );
}

export function truncateToolResult(content, budget, toolName) {
  if (content.length <= budget) return { text: content, truncated: false };

  const cat = getToolTruncationCategory(toolName);
  const strategy = TOOL_TRUNCATION_STRATEGY[cat] || TOOL_TRUNCATION_STRATEGY.default;
  const headBudget = Math.floor(budget * strategy.headRatio);
  const tailBudget = Math.floor(budget * strategy.tailRatio);
  const omitted = content.length - headBudget - tailBudget;

  const text = content.slice(0, headBudget)
    + formatTruncationHint(content.length, headBudget, tailBudget, omitted, toolName)
    + content.slice(-tailBudget);

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
      if (tcXml.includes("<tool_calls>") || tcXml.includes("<invoke name=")) {
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
 */
function renderMessagesBlock(msgs, prependSeparator) {
  const parts = msgs.map(m => `${m.role.toUpperCase()}: ${m.content ?? ""}`);
  const body = parts.join("\n\n");
  return prependSeparator ? "\n\n" + body : body;
}

/**
 * Fit as many messages as possible into a budget, working from newest
 * to oldest. When the boundary cuts through a round, preserves tool_call/tool_result
 * pairing — if an assistant message with tool_calls is included, all its tool_result
 * messages in the same round are also included (or all are dropped).
 *
 * Pattern from Claude Code's adjustIndexToPreserveAPIInvariants.
 */
function fitMessagesInBudget(msgs, budget, prependSeparator, preserveFirstUser) {
  const kept = [];
  let used = 0;
  let i = msgs.length - 1;

  // Track which tool result indices should be kept to match kept tool_calls
  const toolCallNames = new Set();

  // First pass: find which tool calls are in the range we might keep
  // (for pairing preservation)

  while (i >= 0) {
    const msg = msgs[i];
    const block = `${msg.role.toUpperCase()}: ${msg.content ?? ""}`;
    const separator = kept.length > 0 ? "\n\n" : "";
    const entry = separator + block;

    if (used + entry.length <= budget) {
      kept.unshift(msg);
      used += entry.length;

      // Track tool names called by kept assistant messages
      if (msg.role === "assistant") {
        for (const m of (msg.content ?? "").matchAll(/<tool_name>([^<]+)<\/tool_name>/gi)) {
          toolCallNames.add(m[1].trim());
        }
        for (const m of (msg.content ?? "").matchAll(/<invoke\s+name="([^"]+)"/gi)) {
          toolCallNames.add(m[1].trim());
        }
      }
      i--;
      continue;
    }

    // Can't fit whole message — try truncated version
    const header = separator + `${msg.role.toUpperCase()}: `;
    const contentBudget = budget - used - header.length;
    if (contentBudget > 200) {
      // If this is the first user message and we're told to preserve it,
      // use a lower minimum content budget to squeeze it in
      const isFirstUser = preserveFirstUser && i === 0 && msg.role === "user";
      const minBudget = isFirstUser ? 50 : 200;
      if (contentBudget > minBudget) {
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
 * Keep the most recent messages within a character budget, preserving
 * tool_call/tool_result pairings. Fills from newest round backwards;
 * the oldest round is the first to be partially or fully dropped.
 *
 * Pattern from Claude Code's auto-compact and session-memory-compact:
 * - Recent context is more valuable than old context
 * - Tool call/result pairs must never be split
 * - The initial user query gets a lightweight preservation preference
 */
export function keepRecentMessages(msgs, budget) {
  if (!msgs.length) return [];

  const rounds = groupMessagesByRound(msgs);
  const allKept = [];
  let used = 0;

  // Fill from newest round backward (Claude Code pattern: keep most recent)
  for (let r = rounds.length - 1; r >= 0; r--) {
    const roundMsgs = [...rounds[r].messages];
    const isFirstRound = r === 0;
    const prepend = allKept.length > 0;
    const roundBlock = renderMessagesBlock(roundMsgs, prepend);

    if (used + roundBlock.length <= budget) {
      // Whole round fits
      if (prepend) used += roundBlock.length;
      else used = roundBlock.length;
      allKept.unshift(...roundMsgs);
      continue;
    }

    // Round doesn't fit whole — try partial fit
    const partial = fitMessagesInBudget(roundMsgs, budget - used, prepend, isFirstRound);
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

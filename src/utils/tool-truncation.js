// Shared tool result truncation helpers — used by both OpenAI and Anthropic prompt builders.
// Extracted from openai-tool-prompt.js / anthropic-prompt.js to eliminate ~100 lines of duplication.

export function getToolResultBudget(totalContextChars) {
  if (totalContextChars > 100000) return 4000;
  if (totalContextChars > 60000) return 6000;
  if (totalContextChars > 30000) return 10000;
  return 12000;
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

export const PER_MESSAGE_TOOL_RESULT_BUDGET = 70000;
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

export function keepRecentMessages(msgs, budget) {
  const kept = [];
  let used = 0;
  for (let i = msgs.length - 1; i >= 0; i--) {
    const block = `${msgs[i].role.toUpperCase()}: ${msgs[i].content ?? ""}`;
    const separator = i < msgs.length - 1 ? "\n\n" : "";
    const entry = separator + block;
    if (used + entry.length <= budget) {
      used += entry.length;
      kept.push(msgs[i]);
      continue;
    }
    const header = separator + `${msgs[i].role.toUpperCase()}: `;
    const contentBudget = budget - used - header.length;
    if (contentBudget > 300) {
      const truncated = {
        ...msgs[i],
        content: (msgs[i].content ?? "").slice(0, contentBudget) + "\n...[truncated]"
      };
      kept.push(truncated);
    }
    break;
  }
  return kept.reverse();
}

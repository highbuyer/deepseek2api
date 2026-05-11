import { buildPromptFromMessages } from "../utils/prompt.js";
import { getToolFunction, getToolName, resolveToolChoicePolicy } from "./openai-tool-policy.js";
import { config } from "../config.js";
import { log } from "../utils/log.js";
import { toStringSafe, toJsonText } from "../utils/safe-string.js";
import {
  getToolResultBudget, getToolTruncationCategory, TOOL_TRUNCATION_STRATEGY,
  formatTruncationHint, truncateToolResult,
  PER_MESSAGE_TOOL_RESULT_BUDGET, BUDGET_PREVIEW_CHARS, buildBudgetPreview,
  keepRecentMessages, injectReadBloatHint, estimateTokens,
  detectRepeatedToolFailures
} from "../utils/tool-truncation.js";

function normalizeContentText(content) {
  if (typeof content === "string") {
    return content;
  }

  if (!Array.isArray(content)) {
    return "";
  }

  return content
    .map((item) => {
      if (!item || typeof item !== "object") {
        return "";
      }

      if (typeof item.text === "string") {
        return item.text;
      }

      if (typeof item.output_text === "string") {
        return item.output_text;
      }

      if (typeof item.content === "string") {
        return item.content;
      }

      return "";
    })
    .filter(Boolean)
    .join("\n");
}


function formatPromptToolCalls(toolCalls, toolNameById) {
  if (!Array.isArray(toolCalls) || !toolCalls.length) {
    return "";
  }

  // DEBUG: log tool_calls structure to diagnose ID mismatch
  log.debug("prompt", `[formatPromptToolCalls] tool_calls count=${toolCalls.length}, ids=[${toolCalls.map(c => toStringSafe(c?.id).trim()).filter(Boolean).join(",")}]`);

  const blocks = toolCalls
    .map((call) => {
      const name = getToolName(call);
      const callId = toStringSafe(call?.id).trim();
      const argumentsText = toJsonText(getToolFunction(call)?.arguments ?? getToolFunction(call)?.input);

      if (!name) {
        log.debug("prompt", `[formatPromptToolCalls] SKIP: no name for id="${callId}", call keys=[${Object.keys(call || {}).join(",")}], raw_call="${JSON.stringify(call).slice(0, 300)}"`);
        return "";
      }

      if (callId) {
        toolNameById.set(callId, name);
      }

      let input = {};
      try { input = JSON.parse(argumentsText); } catch { /* keep {} */ }

      return `  ${JSON.stringify({ tool: name, arguments: input })}`;
    })
    .filter(Boolean);

  return blocks.length ? `[\n${blocks.join(",\n")}\n]` : "";
}

function normalizeAssistantPromptContent(message, toolNameById) {
  const content = normalizeContentText(message?.content).trim();
  const toolHistory = formatPromptToolCalls(message?.tool_calls, toolNameById);

  if (!content) {
    return toolHistory;
  }

  if (!toolHistory) {
    return content;
  }

  return `${content}\n\n${toolHistory}`;
}

function normalizeToolPromptContent(message, toolNameById, totalContextChars = 0) {
  let content = normalizeContentText(message?.content).trim() || "null";
  // Strip line-number prefixes (Read output format: "  1→code")
  content = content.replace(/^\s*\d+→/gm, "");

  // Truncate long tool results with dynamic budget + head-tail strategy
  const toolName = toolNameById.get(toStringSafe(message?.tool_call_id).trim()) || toStringSafe(message?.name).trim();
  const budget = getToolResultBudget(totalContextChars, config.maxPromptChars);
  const { text: truncated, truncated: wasTruncated } = truncateToolResult(content, budget, toolName);
  content = truncated;

  if (wasTruncated && !toolName) {
    log.warn("prompt", `Tool result truncated: ${content.length}→${truncated.length} chars (no tool name, default strategy)`);
  }

  // Detect Cursor IDE placeholder results — tool execution failure on the client
  // side.  If passed through, the model sees a fake result and retries in a loop.
  // Replace with a clear error so the model stops retrying the same broken tool.
  if (!toolName || isHandoffPlaceholder(content)) {
    const toolLabel = toolName || `call_${toStringSafe(message?.tool_call_id).trim().slice(0, 8) || "unknown"}`;
    log.warn("prompt", `Handoff placeholder detected for "${toolLabel}", replacing with tool error notification (content preview: "${content.slice(0, 80)}")`);
    return `<tool_result id="${toolLabel}">\nTool execution failed: result was not provided (cross-provider or client-side error). Do NOT retry this tool call — try a completely different approach to accomplish the task.\n</tool_result>`;
  }

  return toolName ? `<tool_result id="${toolName}">\n${content}\n</tool_result>` : content;
}

// Known placeholder patterns from Cursor IDE / cross-provider handoffs that
// indicate a tool call was NOT actually executed.
const HANDOFF_PLACEHOLDER_RE = /\b(?:no\s+result\s+provided|cross[-\s]?provider\s+handoff|not\s+executed|execution\s+failed)\b/i;

// Max length for a text to be considered a handoff placeholder.
// Short text containing the pattern is dominated by it and almost certainly
// a placeholder. Long text that happens to mention these phrases in passing
// (e.g. "subprocess error: no result provided by worker") is legitimate.
const HANDOFF_MAX_LENGTH = 200;

function isHandoffPlaceholder(text) {
  if (text.length === 0 || text === "null") return true;
  // Only flag if the text is short — the placeholder pattern must dominate.
  if (text.length > HANDOFF_MAX_LENGTH) return false;
  return HANDOFF_PLACEHOLDER_RE.test(text);
}

function normalizeMessageRole(role) {
  return role === "developer" ? "system" : role;
}

function normalizeMessagesForPrompt(messages) {
  const toolNameById = new Map();
  const readPathByCallId = new Map();  // callId → file path for Read tools
  const readPathCount = new Map();     // path → count (across all rounds)

  // Two accumulators: rawTotalChars tracks pre-truncation content size
  // for budget decisions; totalChars tracks post-truncation size for
  // final prompt length.  Without rawTotalChars, the budget stays at
  // the 20K tier for early tool results while the actual data volume
  // is already near the limit — producing 350K prompts from 100+ tool results.
  let totalChars = 0;
  let rawTotalChars = 0;
  // Per-batch tool result tracking — reset when a user message is encountered
  let batchToolResultChars = 0;

  return (messages ?? []).flatMap((message) => {
    const role = normalizeMessageRole(toStringSafe(message?.role).trim().toLowerCase() || "user");

    if (role === "user") {
      batchToolResultChars = 0;
      const content = normalizeContentText(message?.content);
      totalChars += content.length;
      rawTotalChars += content.length;
      return [{ role, content }];
    }

    if (role === "assistant") {
      batchToolResultChars = 0;
      // Track Read tool paths for repeated-read detection across rounds
      const toolCalls = message?.tool_calls;
      if (Array.isArray(toolCalls)) {
        for (const tc of toolCalls) {
          if (toStringSafe(tc?.function?.name) === "Read") {
            const callId = toStringSafe(tc?.id).trim();
            try {
              const args = JSON.parse(tc.function.arguments || "{}");
              const path = args.file_path || args.path;
              if (path && callId) {
                readPathByCallId.set(callId, path);
                readPathCount.set(path, (readPathCount.get(path) || 0) + 1);
              }
            } catch { /* malformed JSON — skip */ }
          }
        }
      }
      const content = normalizeAssistantPromptContent(message, toolNameById);
      if (content) {
        totalChars += content.length;
        rawTotalChars += content.length;
      }
      return content ? [{ role, content }] : [];
    }

    if (role === "tool" || role === "function") {
      const callId = toStringSafe(message?.tool_call_id).trim();
      const fallbackName = toStringSafe(message?.name).trim();
      const toolName = toolNameById.get(callId) || fallbackName;

      // Per-message budget enforcement: if adding this result would exceed
      // the batch budget, replace it with a preview instead of truncation.
      // Pattern from Claude Code's enforceToolResultBudget.
      let rawContent = normalizeContentText(message?.content).trim() || "null";
      rawContent = rawContent.replace(/^\s*\d+→/gm, "");
      const rawLen = rawContent.length;

      if (!toolName) {
        log.debug("prompt", `Tool result missing name: tool_call_id="${callId}" not found in map (map has ${toolNameById.size} entries: [${[...toolNameById.entries()].map(([k,v]) => `${k}→${v}`).join(", ")}]), fallback name="${fallbackName}", content preview="${rawContent.slice(0, 80)}"`);
      }

      // Repeated read detection: if this Read result is for a file already read
      // in this session, replace with a short warning so the model references
      // the earlier result instead of wasting context on duplicate content.
      if (/^read$/i.test(toolName)) {
        const readPath = readPathByCallId.get(callId);
        if (readPath && (readPathCount.get(readPath) || 0) > 1) {
          const count = readPathCount.get(readPath);
          const warning = `⚠️ REPEATED READ #${count}: "${readPath}" has already been read in this session.\nReference the earlier result. If you need a specific section, use Read with offset/limit.\n`;
          const content = toolName ? `<tool_result id="${toolName}">\n${warning}\n</tool_result>` : warning;
          log.warn("prompt", `Repeated read detected: "${readPath}" (read ${count} times), replacing result with warning`);
          totalChars += content.length;
          rawTotalChars += rawLen;
          return [{ role: "tool", content }];
        }
      }

      if (batchToolResultChars + rawLen > PER_MESSAGE_TOOL_RESULT_BUDGET && batchToolResultChars > 0) {
        const replacement = buildBudgetPreview(rawContent, rawLen, toolName);
        const content = toolName ? `<tool_result id="${toolName}">\n${replacement}\n</tool_result>` : replacement;
        log.warn("prompt", `Per-message tool result budget exceeded (${batchToolResultChars} + ${rawLen} > ${PER_MESSAGE_TOOL_RESULT_BUDGET}), replacing "${toolName}" with preview`);
        totalChars += content.length;
        rawTotalChars += rawLen;
        return [{ role: "tool", content }];
      }

      // Use rawTotalChars for budget so it tightens as real data volume grows.
      // Without this, totalChars (truncated) lags far behind raw data volume,
      // keeping the budget too generous for early tool results.
      const content = normalizeToolPromptContent(message, toolNameById, rawTotalChars);
      batchToolResultChars += rawLen;
      rawTotalChars += rawLen;
      totalChars += content.length;
      return [{ role: "tool", content }];
    }

    const content = normalizeContentText(message?.content);
    totalChars += content.length;
    rawTotalChars += content.length;
    return [{ role, content }];
  });
}

function normalizeApplyPatchParams(params) {
  // If the tool already has a schema, pass it through as-is.
  // Different clients (Cursor, Claude Code, etc.) have different
  // ApplyPatch schemas — don't override with a guessed format.
  return params;
}

function formatToolSchema(tool) {
  const definition = getToolFunction(tool);
  const name = getToolName(tool);
  if (!name) {
    return "";
  }

  const fullDesc = toStringSafe(definition?.description).trim() || "No description available";
  const truncatedDesc = fullDesc.length > config.maxToolDescChars
    ? fullDesc.slice(0, config.maxToolDescChars) + "..."
    : fullDesc;

  let params = definition?.parameters;

  if (name === "apply_patch" || name === "ApplyPatch") {
    params = normalizeApplyPatchParams(params);
  }

  let paramSummary = "";
  if (params && typeof params === "object") {
    const required = Array.isArray(params.required) ? params.required : [];
    const props = params.properties ?? {};
    const propNames = Object.keys(props);
    if (propNames.length <= 8) {
      paramSummary = JSON.stringify(props);
    } else {
      const summary = {};
      propNames.forEach((key) => {
        const p = props[key];
        summary[key] = p?.type
          ? (p.description && p.description.length <= 40 ? { type: p.type, description: p.description } : { type: p.type })
          : p;
      });
      paramSummary = JSON.stringify(summary);
    }
  }

  return [
    `Tool: ${name}`,
    `Description: ${truncatedDesc}`,
    `Parameters: ${paramSummary || "{}"}`
  ].join("\n");
}

// Build an example arguments object for a tool using its actual parameter names.
// Returns a JSON string like {"param1": "value1", "param2": 42}
function buildExampleArgs(name, tools) {
  const tool = tools.find(t => {
    const n = toStringSafe(t?.name).trim().toLowerCase();
    return n === name.toLowerCase();
  });
  if (!tool) return "{}";
  const params = tool.function?.parameters ?? tool.input_schema ?? {};
  const props = params.properties ?? {};
  const keys = Object.keys(props);
  if (!keys.length) return "{}";
  // Pick the first 1-2 parameters with sensible placeholder values
  const example = {};
  for (let i = 0; i < Math.min(keys.length, 2); i++) {
    const key = keys[i];
    const prop = props[key];
    const type = prop?.type || "string";
    if (type === "number" || type === "integer") example[key] = 1;
    else if (type === "boolean") example[key] = true;
    else if (type === "array") example[key] = [];
    else example[key] = key.includes("path") || key.includes("directory") ? "/project/src" : "value";
  }
  return JSON.stringify(example);
}

function buildToolPrompt(policy, tools) {
  const allowed = new Set(policy.allowedToolNames);
  const toolSchemas = tools
    .filter((tool) => allowed.has(getToolName(tool)))
    .map(formatToolSchema)
    .filter(Boolean);

  if (!toolSchemas.length) {
    return "";
  }

  const concreteTools = policy.allowedToolNames.filter(n => n === "Bash" || n === "Read");
  const firstName = concreteTools[0] || policy.allowedToolNames[0] || "Bash";
  const secondName = concreteTools[1] || policy.allowedToolNames[1] || "Read";
  const args1 = buildExampleArgs(firstName, tools);
  const args2 = buildExampleArgs(secondName, tools);

  let prompt = [
    "When tools are listed below, use them to take action — prefer calling tools over narration.",
    "",
    "=== BEST PRACTICES ===",
    "",
    "1. !!! GREP FIRST !!! For code analysis, ALWAYS use Grep/Search BEFORE Read.",
    "   Grep finds patterns across all files instantly; Read only gets one file at a time.",
    "2. If a Read result shows ⚠️ FILE TRUNCATED (limit: 12000 chars), stop reading.",
    "   Switch to Grep for the rest, or Read with offset/limit in 200-line chunks.",
    "3. After any Read, check for ⚠️ — if present, you do NOT have the full file.",
    "4. Batch independent tool calls in ONE ```json array for parallel execution.",
    "",
    "=== ⚠️ CYPHER HARD RULES (VIOLATE = QUERY FAILS) ===",
    "1. ALL rels use CodeRelation with type property: [:CodeRelation {type: 'IMPORTS'}]",
    "   NEVER use [r:IMPORTS] or [:IMPORTS] — these do NOT exist.",
    "2. Variable-length paths MUST have range: *1..5 (NEVER * alone).",
    "3. Every RETURN with multiple rows MUST have LIMIT 50.",
    "4. ORDER BY uses PROPERTIES (fn.name), NOT node refs (fn).",
    "5. Node labels: File,Folder,Function,Class,Interface,Method,CodeElement,Community,Process",
    "6. Process/Community use heuristicLabel NOT name:",
    "   - Process: p.heuristicLabel, p.processType, p.stepCount (NO p.name)",
    "   - Community: c.heuristicLabel, c.cohesion (NO c.name)",
    "   - All others (File,Function,Class,etc.): use name, filePath",
    "7. Node properties reference:",
    "   - Common: name (STRING), filePath (STRING), startLine (INT32), endLine (INT32)",
    "   - Process: heuristicLabel, processType, stepCount, communities, entryPointId, terminalId",
    "   - Community: heuristicLabel, cohesion, symbolCount, keywords, description",
    "",
    "=== TOOLS ===",
    "",
    toolSchemas.join("\n\n"),
    "",
    "=== TOOL CALL FORMAT ===",
    "",
    "Output tool calls as a JSON array inside a ```json code fence.",
    "This is standard JSON format you already know from training:",
    "",
    "```json",
    "[",
    '  {"tool": "ToolName", "arguments": {"param1": "value1", "param2": 42}},',
    '  {"tool": "ToolName2", "arguments": {"param3": [1, 2, 3]}}',
    "]",
    "```",
    "",
    "RULES:",
    "",
    "1. ONE ```json block containing ALL tool calls as a single JSON array",
    '2. Each object has "tool" (exact tool name) and "arguments" (JSON object)',
    "3. String values in double quotes. Numbers/booleans without quotes.",
    "4. Multi-line content (patch, code) can use literal newlines inside strings",
    "5. Do NOT wrap JSON in XML tags — no function_calls, invoke, parameter, or tool_calls tags",
    "6. Text before/after the ```json block is allowed for natural flow.",
    "",
    "=== FEW-SHOT EXAMPLES ===",
    "USE THE EXACT TOOL NAME AND PARAMETER NAMES FROM THE TOOLS LIST ABOVE.",
    "",
    "Single tool:",
    "```json",
    `[{"tool": "${firstName}", "arguments": ${args1}}]`,
    "```",
    "",
    policy.allowedToolNames.length >= 2
    ? [
        "Multiple independent tools (BATCH when possible):",
        "```json",
        `[{"tool": "${firstName}", "arguments": ${args1}}, {"tool": "${secondName}", "arguments": ${args2}}]`,
        "```",
      ]
    : "",
    // Note: ApplyPatch schema varies by client (Cursor uses "patch",
    // Claude Code uses "target_directory"+"patchText").  The TOOLS list
    // above shows the exact schema — match it precisely.
    "",
    "=== WORKFLOW ===",
    "",
    "1. Thinking: 2-3 lines MAX. Do NOT analyze in depth — ACT.",
    "2. Call tools immediately. Prefer tools over explanations.",
    "3. Batch independent tool calls in ONE ```json array for parallel execution.",
    "4. If a tool fails, try a different approach. Never repeat the same failed call.",
    "5. After tools return results, fix bugs directly.",
    "6. If you do NOT need a tool, answer normally without any ```json block.",
    "7. NEVER echo <tool_result> content in your visible output — it's internal context only."
  ].flat().join("\n");

  if (policy.mode === "required") {
    prompt += "\n9. For this response, you MUST call at least one tool.";
  }

  if (policy.mode === "forced") {
    prompt += `\n9. For this response, you MUST call exactly this tool: ${policy.forcedName}.`;
  }

  return prompt;
}

function injectToolPrompt(messages, tools, policy) {
  if (!policy.allowedToolNames.length) {
    return messages;
  }

  const toolPrompt = buildToolPrompt(policy, tools);
  if (!toolPrompt) {
    return messages;
  }

  const systemIndex = messages.findIndex((message) => message.role === "system");
  if (systemIndex === -1) {
    return [{ role: "system", content: toolPrompt }, ...messages];
  }

  const updated = [...messages];
  updated[systemIndex] = {
    ...updated[systemIndex],
    content: [updated[systemIndex].content, toolPrompt].filter(Boolean).join("\n\n")
  };
  return updated;
}

function truncatePromptMessages(messages, maxBudget, measure) {
  const len = measure || (s => s.length);
  const units = measure ? "tokens" : "chars";
  const systemMsgs = messages.filter((m) => m.role === "system");
  const nonSystemMsgs = messages.filter((m) => m.role !== "system");

  const systemPart = buildPromptFromMessages(systemMsgs);
  const systemLen = len(systemPart);

  if (systemLen >= maxBudget) {
    const MIN_USER_BUDGET = Math.min(2000, Math.floor(maxBudget * 0.1));
    const systemBudget = maxBudget - MIN_USER_BUDGET;
    const systemPartTrimmed = systemPart.slice(0, Math.max(0, systemBudget));
    const budget = maxBudget - len(systemPartTrimmed);
    const kept = keepRecentMessages(nonSystemMsgs, budget, measure);
    const dropped = nonSystemMsgs.length - kept.length;
    const keptPart = buildPromptFromMessages(kept);
    const result = systemPartTrimmed + (kept.length ? "\n\n" + keptPart : "");
    if (dropped > 0) {
      log.warn("prompt", `System prompt alone (${systemLen} ${units}) exceeds limit (${maxBudget}), trimmed system to ${systemBudget}, kept ${kept.length}/${nonSystemMsgs.length} messages (dropped ${dropped}), total ${len(result)} ${units}`);
    }
    return result;
  }

  const budget = maxBudget - systemLen;
  const kept = keepRecentMessages(nonSystemMsgs, budget, measure);
  const dropped = nonSystemMsgs.length - kept.length;
  const keptPart = buildPromptFromMessages(kept);
  const result = systemPart + "\n\n" + keptPart;

  if (dropped > 0) {
    const droppedRounds = dropped > 0 ? Math.max(1, Math.round(dropped / 3)) : 0;
    log.warn("prompt", `Prompt truncated: kept system (${systemLen} ${units}) + ${kept.length}/${nonSystemMsgs.length} messages (~${droppedRounds} round(s) dropped from head), total ${len(result)}/${maxBudget} ${units}`);
  }

  return result;
}

export function buildOpenAiPrompt({ messages, toolChoice, tools }) {
  const policy = resolveToolChoicePolicy({ tools, toolChoice });
  const normalizedMessages = normalizeMessagesForPrompt(messages);

  // Loop circuit breaker: if the model is calling the same tool with the
  // same arguments and getting the same error, inject a hard interrupt
  // to force a strategy change before the prompt explodes.
  const loopInterruption = detectRepeatedToolFailures(messages);
  if (loopInterruption) {
    log.warn("prompt", "Repeated failed tool call loop detected — injecting circuit breaker");
    const sysIdx = normalizedMessages.findIndex(m => m.role === "system");
    if (sysIdx >= 0) {
      normalizedMessages.splice(sysIdx + 1, 0, { role: "system", content: loopInterruption });
    } else {
      normalizedMessages.unshift({ role: "system", content: loopInterruption });
    }
  }

  let promptMessages = injectToolPrompt(normalizedMessages, tools ?? [], policy);

  // Check for Read result bloat (Claude Code checkReadResultBloat pattern)
  promptMessages = injectReadBloatHint(promptMessages);

  let prompt = buildPromptFromMessages(promptMessages);

  const promptLen = config.maxPromptTokens ? estimateTokens(prompt) : prompt.length;
  const promptLimit = config.maxPromptTokens || config.maxPromptChars;
  const measure = config.maxPromptTokens ? estimateTokens : undefined;

  if (promptLen > promptLimit) {
    log.warn("prompt", `Prompt too long: ${promptLen} ${config.maxPromptTokens ? "tokens" : "chars"} (limit: ${promptLimit}), truncating conversation history while preserving system prompt`);
    prompt = truncatePromptMessages(promptMessages, promptLimit, measure);
  }

  log.debug("prompt", `Final prompt length: ${prompt.length} chars, toolNames: [${policy.allowedToolNames.join(",")}]`);

  // DEBUG: log TOOL result blocks to diagnose "No result provided (cross-provider handoff)"
  const toolResultBlocks = [...prompt.matchAll(/TOOL:[\s\S]*?(?=\n\n(?:USER|ASSISTANT|SYSTEM):|\n*$)/g)];
  if (toolResultBlocks.length > 0) {
    log.debug("prompt", `=== TOOL RESULT BLOCKS (${toolResultBlocks.length} total) ===`);
    toolResultBlocks.forEach((match, i) => {
      const block = match[0];
      const truncated = block.length > 300 ? block.slice(0, 300) + `...[${block.length} chars total]` : block;
      log.debug("prompt", `  Block #${i+1}: ${truncated}`);
    });
  } else {
    log.debug("prompt", `No TOOL result blocks found in prompt (${prompt.length} chars)`);
  }

  return {
    prompt,
    toolChoicePolicy: policy,
    toolNames: policy.allowedToolNames
  };
}

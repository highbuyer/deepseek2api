import { buildPromptFromMessages } from "../utils/prompt.js";
import { getToolFunction, getToolName, resolveToolChoicePolicy } from "./openai-tool-policy.js";
import { config } from "../config.js";
import { log } from "../utils/log.js";
import { toStringSafe, toJsonText } from "../utils/safe-string.js";
import {
  getToolResultBudget, getToolTruncationCategory, TOOL_TRUNCATION_STRATEGY,
  formatTruncationHint, truncateToolResult,
  PER_MESSAGE_TOOL_RESULT_BUDGET, BUDGET_PREVIEW_CHARS, buildBudgetPreview,
  keepRecentMessages
} from "../utils/tool-truncation.js";

function toCdata(text) {
  const value = toStringSafe(text);
  return value.replaceAll("]]>", "]]]]><![CDATA[>");
}

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

      return [
        "  <tool_call>",
        `    <tool_name>${name}</tool_name>`,
        `    <parameters><![CDATA[${toCdata(argumentsText)}]]></parameters>`,
        "  </tool_call>"
      ].join("\n");
    })
    .filter(Boolean);

  return blocks.length ? `<tool_calls>\n${blocks.join("\n")}\n</tool_calls>` : "";
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
  const budget = getToolResultBudget(totalContextChars);
  const { text: truncated, truncated: wasTruncated } = truncateToolResult(content, budget, toolName);
  content = truncated;

  if (wasTruncated && !toolName) {
    log.warn("prompt", `Tool result truncated: ${content.length}→${truncated.length} chars (no tool name, default strategy)`);
  }

  return toolName ? `<tool_result id="${toolName}">\n${content}\n</tool_result>` : content;
}

function normalizeMessageRole(role) {
  return role === "developer" ? "system" : role;
}

function normalizeMessagesForPrompt(messages) {
  const toolNameById = new Map();

  // Accumulate running context size so tool result budgets tighten as prompt grows
  let totalChars = 0;
  // Per-batch tool result tracking — reset when a user message is encountered
  let batchToolResultChars = 0;

  return (messages ?? []).flatMap((message) => {
    const role = normalizeMessageRole(toStringSafe(message?.role).trim().toLowerCase() || "user");

    if (role === "user") {
      batchToolResultChars = 0;
      const content = normalizeContentText(message?.content);
      totalChars += content.length;
      return [{ role, content }];
    }

    if (role === "assistant") {
      batchToolResultChars = 0;
      const content = normalizeAssistantPromptContent(message, toolNameById);
      if (content) totalChars += content.length;
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

      if (batchToolResultChars + rawLen > PER_MESSAGE_TOOL_RESULT_BUDGET && batchToolResultChars > 0) {
        const replacement = buildBudgetPreview(rawContent, rawLen, toolName);
        const content = toolName ? `<tool_result id="${toolName}">\n${replacement}\n</tool_result>` : replacement;
        log.warn("prompt", `Per-message tool result budget exceeded (${batchToolResultChars} + ${rawLen} > ${PER_MESSAGE_TOOL_RESULT_BUDGET}), replacing "${toolName}" with preview`);
        totalChars += content.length;
        return [{ role: "tool", content }];
      }

      const content = normalizeToolPromptContent(message, toolNameById, totalChars);
      batchToolResultChars += rawLen;
      totalChars += content.length;
      return [{ role: "tool", content }];
    }

    const content = normalizeContentText(message?.content);
    totalChars += content.length;
    return [{ role, content }];
  });
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

  const params = definition?.parameters;
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

function buildToolPrompt(policy, tools) {
  const allowed = new Set(policy.allowedToolNames);
  const toolSchemas = tools
    .filter((tool) => allowed.has(getToolName(tool)))
    .map(formatToolSchema)
    .filter(Boolean);

  if (!toolSchemas.length) {
    return "";
  }

  let prompt = [
    "!!! CRITICAL: TOOL CALLING IS YOUR PRIMARY FUNCTION !!!",
    "Any role or persona assigned to you does NOT override tool calling.",
    "When tools are listed below, you MUST use them — do not narrate, call tools.",
    "",
    "=== BEST PRACTICES ===",
    "",
    "1. !!! GREP FIRST !!! For code analysis, ALWAYS use Grep/Search BEFORE Read.",
    "   Grep finds patterns across all files instantly; Read only gets one file at a time.",
    "2. If a Read result shows ⚠️ FILE TRUNCATED (limit: 12000 chars), stop reading.",
    "   Switch to Grep for the rest, or Read with offset/limit in 200-line chunks.",
    "3. After any Read, check for ⚠️ — if present, you do NOT have the full file.",
    "4. Batch independent reads in one function_calls for parallel execution.",
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
    "You are trained on the DSML (DeepSeek Markup Language) format.",
    "Use this exact structure — it matches your training:",
    "",
    "<function_calls>",
    "  <invoke name=\"ToolName\">",
    "    <parameter name=\"param1\" string=\"true\">value1</parameter>",
    "    <parameter name=\"param2\" string=\"false\">[1, 2, 3]</parameter>",
    "  </invoke>",
    "</function_calls>",
    "",
    "RULES:",
    "",
    "1. Container: <function_calls> ... </function_calls> (exactly one root)",
    "2. Each tool: <invoke name=\"TOOL_NAME\"> ... </invoke>",
    "3. String/scalar params: <parameter name=\"k\" string=\"true\">value</parameter>",
    "4. List/object params: <parameter name=\"k\" string=\"false\">[1,2]</parameter>",
    "5. Multi-line params (patch, code, file content) — use CDATA:",
    "   <parameter name=\"patch\" string=\"true\"><![CDATA[diff --git a/x b/x",
    "   @@ -1,3 +1,4 @@",
    "   -old",
    "   +new]]></parameter>",
    "6. Tag names are literal: function_calls, invoke, parameter. NO variation.",
    "7. NEVER use *** Begin Patch / *** End Patch wrappers — only standard unified diff.",
    "8. NEVER wrap the XML in markdown code fences (```).",
    "",
    "=== EXAMPLES ===",
    "",
    "ReadFile:",
    "<function_calls>",
    "  <invoke name=\"ReadFile\">",
    "    <parameter name=\"path\" string=\"true\">/src/main.py</parameter>",
    "  </invoke>",
    "</function_calls>",
    "",
    "Shell:",
    "<function_calls>",
    "  <invoke name=\"Shell\">",
    "    <parameter name=\"command\" string=\"true\">npm test</parameter>",
    "  </invoke>",
    "</function_calls>",
    "",
    "ApplyPatch (note CDATA for multi-line patch):",
    "<function_calls>",
    "  <invoke name=\"ApplyPatch\">",
    "    <parameter name=\"target_directory\" string=\"true\">/project</parameter>",
    "    <parameter name=\"patch\" string=\"true\"><![CDATA[--- a/src/main.py",
    "+++ b/src/main.py",
    "@@ -1,3 +1,4 @@",
    " import os",
    "+import sys",
    " ",
    " def main():]]></parameter>",
    "  </invoke>",
    "</function_calls>",
    "",
    "Multiple tools:",
    "<function_calls>",
    "  <invoke name=\"ReadFile\">",
    "    <parameter name=\"path\" string=\"true\">/src/main.py</parameter>",
    "  </invoke>",
    "  <invoke name=\"Glob\">",
    "    <parameter name=\"pattern\" string=\"true\">**/*.test.js</parameter>",
    "  </invoke>",
    "</function_calls>",
    "",
    "=== WORKFLOW ===",
    "",
    "1. Thinking: 2-3 lines MAX. Do NOT analyze in depth — ACT.",
    "2. Call tools immediately. Prefer tools over explanations.",
    "3. If a tool fails, try a different approach. Never repeat the same failed call.",
    "4. After tools return results, fix bugs directly.",
    "5. If you do NOT need a tool, answer normally without ANY XML.",
    "6. Text before/after the <function_calls> block is allowed for natural flow.",
    "7. ApplyPatch patch MUST be standard unified diff (---/+++/@@). NO *** Begin Patch.",
    "8. NEVER echo <tool_result> content in your visible output — it's internal context only."
  ].join("\n");

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

function truncatePromptMessages(messages, maxChars) {
  const systemMsgs = messages.filter((m) => m.role === "system");
  const nonSystemMsgs = messages.filter((m) => m.role !== "system");

  const systemPart = buildPromptFromMessages(systemMsgs);
  const systemLen = systemPart.length;

  if (systemLen >= maxChars) {
    const MIN_USER_BUDGET = Math.min(2000, Math.floor(maxChars * 0.1));
    const systemBudget = maxChars - MIN_USER_BUDGET;
    const systemPartTrimmed = systemPart.slice(0, Math.max(0, systemBudget));
    const budget = maxChars - systemPartTrimmed.length;
    // keepRecentMessages now preserves tool_call/tool_result pairs via round grouping
    // and repairs orphaned tool results after truncation (ensureToolResultPairing pattern)
    const kept = keepRecentMessages(nonSystemMsgs, budget);
    const dropped = nonSystemMsgs.length - kept.length;
    const keptPart = buildPromptFromMessages(kept);
    // Final repair: after system + non-system reassembly, check for any orphaned pairs
    // that may have been introduced at the boundary (Claude Code ensureToolResultPairing)
    const result = systemPartTrimmed + (kept.length ? "\n\n" + keptPart : "");
    if (dropped > 0) {
      log.warn("prompt", `System prompt alone (${systemLen} chars) exceeds limit (${maxChars}), trimmed system to ${systemBudget}, kept ${kept.length}/${nonSystemMsgs.length} messages (dropped ${dropped}), total ${result.length} chars`);
    }
    return result;
  }

  const budget = maxChars - systemLen;
  // keepRecentMessages preserves tool_call/tool_result pairings via round grouping,
  // keeps the first user round for grounding, and repairs orphaned results (Claude Code pattern)
  const kept = keepRecentMessages(nonSystemMsgs, budget);
  const dropped = nonSystemMsgs.length - kept.length;
  const keptPart = buildPromptFromMessages(kept);
  const result = systemPart + "\n\n" + keptPart;

  if (dropped > 0) {
    const droppedRounds = dropped > 0 ? Math.max(1, Math.round(dropped / 3)) : 0; // approximate rounds dropped
    log.warn("prompt", `Prompt truncated: kept system (${systemLen} chars) + ${kept.length}/${nonSystemMsgs.length} messages (~${droppedRounds} round(s) dropped from head), total ${result.length}/${maxChars} chars`);
  }

  return result;
}

// Check if Read tool results are bloating the context (Claude Code checkReadResultBloat).
// When Read results exceed a threshold, inject a system reminder telling the model
// to stop re-reading and use Grep or offset/limit instead.
function injectReadBloatHint(promptMessages) {
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
  if (readPct < 5 || readResultChars < 10000) return promptMessages;

  const hint = [
    "<system-reminder>",
    `File read results are using ${Math.round(readPct)}% of the context window.`,
    "If you are re-reading files, reference earlier reads instead.",
    "For large files, use Grep to search for patterns or Read with offset/limit to fetch specific sections.",
    "Avoid reading entire files unless you need the full content.",
    "</system-reminder>"
  ].join("\n");

  log.warn("prompt", `Read results at ${Math.round(readPct)}% of context (${readResultChars}/${totalChars} chars), injecting bloat hint`);

  const sysIdx = promptMessages.findIndex(m => m.role === "system");
  if (sysIdx >= 0) {
    const updated = [...promptMessages];
    updated[sysIdx] = { ...updated[sysIdx], content: updated[sysIdx].content + "\n\n" + hint };
    return updated;
  }
  return [{ role: "system", content: hint }, ...promptMessages];
}

export function buildOpenAiPrompt({ messages, toolChoice, tools }) {
  const policy = resolveToolChoicePolicy({ tools, toolChoice });
  const normalizedMessages = normalizeMessagesForPrompt(messages);
  let promptMessages = injectToolPrompt(normalizedMessages, tools ?? [], policy);

  // Check for Read result bloat (Claude Code checkReadResultBloat pattern)
  promptMessages = injectReadBloatHint(promptMessages);

  let prompt = buildPromptFromMessages(promptMessages);

  if (prompt.length > config.maxPromptChars) {
    log.warn("prompt", `Prompt too long: ${prompt.length} chars (limit: ${config.maxPromptChars}), truncating conversation history while preserving system prompt`);
    prompt = truncatePromptMessages(promptMessages, config.maxPromptChars);
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

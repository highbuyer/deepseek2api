import { buildPromptFromMessages } from "../utils/prompt.js";
import { getToolFunction, getToolName, resolveToolChoicePolicy } from "./openai-tool-policy.js";
import { config } from "../config.js";
import { log } from "../utils/log.js";
import { toStringSafe, toJsonText } from "../utils/safe-string.js";

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

  const blocks = toolCalls
    .map((call) => {
      const name = getToolName(call);
      const callId = toStringSafe(call?.id).trim();
      const argumentsText = toJsonText(getToolFunction(call)?.arguments ?? getToolFunction(call)?.input);

      if (!name) {
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

function normalizeToolPromptContent(message, toolNameById) {
  let content = normalizeContentText(message?.content).trim() || "null";
  // Strip line-number prefixes (Read output format: "  1→code")
  content = content.replace(/^\s*\d+→/gm, "");
  // Truncate very long tool results
  const MAX_TOOL_RESULT_CHARS = 4000;
  if (content.length > MAX_TOOL_RESULT_CHARS) {
    content = content.slice(0, MAX_TOOL_RESULT_CHARS)
      + `\n...[truncated ${content.length - MAX_TOOL_RESULT_CHARS} chars]`;
  }
  const toolName = toolNameById.get(toStringSafe(message?.tool_call_id).trim()) || toStringSafe(message?.name).trim();
  return toolName ? `<tool_result id="${toolName}">\n${content}\n</tool_result>` : content;
}

function normalizeMessageRole(role) {
  return role === "developer" ? "system" : role;
}

function normalizeMessagesForPrompt(messages) {
  const toolNameById = new Map();

  return (messages ?? []).flatMap((message) => {
    const role = normalizeMessageRole(toStringSafe(message?.role).trim().toLowerCase() || "user");

    if (role === "assistant") {
      const content = normalizeAssistantPromptContent(message, toolNameById);
      return content ? [{ role, content }] : [];
    }

    if (role === "tool" || role === "function") {
      return [{ role: "tool", content: normalizeToolPromptContent(message, toolNameById) }];
    }

    return [{ role, content: normalizeContentText(message?.content) }];
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

function keepRecentMessages(msgs, budget) {
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
    // Try to truncate this message to fit remaining budget
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
    const kept = keepRecentMessages(nonSystemMsgs, budget);
    const dropped = nonSystemMsgs.length - kept.length;
    const result = systemPartTrimmed + (kept.length ? "\n\n" + buildPromptFromMessages(kept) : "");
    log.warn("prompt", `System prompt alone (${systemLen} chars) exceeds limit (${maxChars}), trimmed to ${systemBudget} + ${kept.length}/${nonSystemMsgs.length} recent messages (dropped ${dropped}), total ${result.length} chars`);
    return result;
  }

  const budget = maxChars - systemLen;
  const kept = keepRecentMessages(nonSystemMsgs, budget);
  const dropped = nonSystemMsgs.length - kept.length;
  const result = systemPart + "\n\n" + buildPromptFromMessages(kept);

  if (dropped > 0) {
    log.warn("prompt", `Prompt truncated: kept system (${systemLen} chars) + ${kept.length}/${nonSystemMsgs.length} messages, dropped ${dropped} oldest, total ${result.length} chars`);
  }

  return result;
}

export function buildOpenAiPrompt({ messages, toolChoice, tools }) {
  const policy = resolveToolChoicePolicy({ tools, toolChoice });
  const normalizedMessages = normalizeMessagesForPrompt(messages);
  const promptMessages = injectToolPrompt(normalizedMessages, tools ?? [], policy);

  let prompt = buildPromptFromMessages(promptMessages);

  if (prompt.length > config.maxPromptChars) {
    log.warn("prompt", `Prompt too long: ${prompt.length} chars (limit: ${config.maxPromptChars}), truncating conversation history while preserving system prompt`);
    prompt = truncatePromptMessages(promptMessages, config.maxPromptChars);
  }

  log.debug("prompt", `Final prompt length: ${prompt.length} chars, toolNames: [${policy.allowedToolNames.join(",")}]`);

  return {
    prompt,
    toolChoicePolicy: policy,
    toolNames: policy.allowedToolNames
  };
}

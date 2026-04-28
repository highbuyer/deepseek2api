import { buildPromptFromMessages } from "../utils/prompt.js";
import { getToolFunction, getToolName, resolveToolChoicePolicy } from "./openai-tool-policy.js";
import { config } from "../config.js";
import { log } from "../utils/log.js";

function toStringSafe(value) {
  if (typeof value === "string") {
    return value;
  }

  if (value === null || value === undefined) {
    return "";
  }

  return String(value);
}

function toJsonText(value, fallback = "{}") {
  if (typeof value === "string") {
    return value.trim() || fallback;
  }

  try {
    return JSON.stringify(value ?? {}) || fallback;
  } catch {
    return fallback;
  }
}

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
  const content = normalizeContentText(message?.content).trim() || "null";
  const toolName = toolNameById.get(toStringSafe(message?.tool_call_id).trim()) || toStringSafe(message?.name).trim();
  return toolName ? `Tool result for ${toolName}:\n${content}` : content;
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
    "You have access to these tools:",
    "",
    toolSchemas.join("\n\n"),
    "",
    "When calling tools, emit raw XML inline at the exact point where the tool call should appear.",
    "You may include normal assistant text before and/or after the XML block when appropriate.",
    "If the user explicitly asks for text before or after the tool call, preserve that text around the XML block instead of omitting it.",
    "Do not wrap the XML in markdown code fences.",
    "",
    "<tool_calls>",
    "  <tool_call>",
    "    <tool_name>TOOL_NAME_HERE</tool_name>",
    "    <parameters>{\"key\":\"value\"}</parameters>",
    "  </tool_call>",
    "</tool_calls>",
    "",
    "Example with surrounding text:",
    "开始查询。",
    "<tool_calls>",
    "  <tool_call>",
    "    <tool_name>weather</tool_name>",
    "    <parameters>{\"city\":\"Shanghai\"}</parameters>",
    "  </tool_call>",
    "</tool_calls>",
    "查询已提交。",
    "",
    "RULES:",
    "1) When using a tool, output a raw XML block exactly in the position where that tool call should happen.",
    "2) <parameters> MUST contain a strict JSON object with double-quoted keys and strings.",
    "3) Multiple tools go inside one <tool_calls> root.",
    "4) Normal text is allowed before and after the XML block.",
    "5) Do not wrap the XML in markdown code fences.",
    "6) Use only declared tool names and exact schema field names.",
    "7) If you do not need a tool, answer normally without any XML."
  ].join("\n");

  if (policy.mode === "required") {
    prompt += "\n8) For this response, you MUST call at least one tool.";
  }

  if (policy.mode === "forced") {
    prompt += `\n8) For this response, you MUST call exactly this tool: ${policy.forcedName}.`;
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
    log.warn("prompt", `System prompt alone is ${systemLen} chars (limit: ${maxChars}), keeping it as-is`);
    return systemPart.slice(0, maxChars);
  }

  const budget = maxChars - systemLen;
  const kept = [];
  let used = 0;

  for (let i = nonSystemMsgs.length - 1; i >= 0; i--) {
    const block = `${nonSystemMsgs[i].role.toUpperCase()}: ${nonSystemMsgs[i].content ?? ""}`;
    const entry = i < nonSystemMsgs.length - 1 ? "\n\n" + block : block;
    if (used + entry.length > budget) break;
    used += entry.length;
    kept.unshift(nonSystemMsgs[i]);
  }

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

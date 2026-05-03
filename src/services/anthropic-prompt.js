import { buildPromptFromMessages } from "../utils/prompt.js";
import { getToolFunction, getToolName, resolveToolChoicePolicy } from "./openai-tool-policy.js";
import { config } from "../config.js";
import { log } from "../utils/log.js";
import { toStringSafe, toJsonText } from "../utils/safe-string.js";
import { resolveOpenAiModel } from "./openai-request.js";

/* ── Anthropic → DeepSeek prompt conversion ── */

function normalizeSystemPrompt(system) {
  if (!system) return [];
  if (typeof system === "string") return [{ role: "system", content: system }];
  if (Array.isArray(system)) {
    const text = system
      .filter((b) => b?.type === "text")
      .map((b) => b.text ?? "")
      .filter(Boolean)
      .join("\n\n");
    return text ? [{ role: "system", content: text }] : [];
  }
  return [];
}

function contentBlockToText(block) {
  if (!block || typeof block !== "object") return "";
  if (block.type === "text") return toStringSafe(block.text);
  if (block.type === "thinking") return ""; //  thinking content removed from prompt to save budget
  if (block.type === "tool_use") {
    const name = toStringSafe(block.name);
    const args = toJsonText(block.input);
    return [
      "  <tool_call>",
      `    <tool_name>${name}</tool_name>`,
      `    <parameters><![CDATA[${args.replaceAll("]]>", "]]]]><![CDATA[>")}]]></parameters>`,
      "  </tool_call>"
    ].join("\n");
  }
  if (block.type === "tool_result") {
    const content = typeof block.content === "string"
      ? block.content
      : (Array.isArray(block.content) ? block.content.map(c => c?.text ?? "").join("\n") : "");
    const name = toStringSafe(block.tool_use_id).slice(-20);
    return `Tool result for ${name}:\n${content}`;
  }
  return "";
}

function normalizeAnthropicMessages(messages) {
  return (messages ?? []).flatMap((message) => {
    const role = message?.role === "assistant" ? "assistant"
      : message?.role === "user" ? "user"
        : "user";

    const content = message?.content;
    if (typeof content === "string") return [{ role, content }];
    if (!Array.isArray(content)) return [];

    const textBlocks = content.map(contentBlockToText).filter(Boolean);

    if (role === "user") {
      return [{ role: "user", content: textBlocks.join("\n") || " " }];
    }

    // assistant: keep tool calls as structured info
    const toolCalls = content.filter((b) => b?.type === "tool_use");
    const textContent = content
      .filter((b) => b?.type === "text" || b?.type === "thinking")
      .map(contentBlockToText)
      .filter(Boolean)
      .join("\n");

    const toolHistory = toolCalls.length
      ? `<tool_calls>\n${toolCalls.map(contentBlockToText).filter(Boolean).join("\n")}\n</tool_calls>`
      : "";

    const combined = [textContent, toolHistory].filter(Boolean).join("\n\n");
    return combined ? [{ role: "assistant", content: combined }] : [];
  });
}

function formatAnthropicToolSchema(tool) {
  const name = toStringSafe(tool?.name);
  if (!name) return "";

  const desc = toStringSafe(tool?.description).trim() || "No description";
  const truncatedDesc = desc.length > config.maxToolDescChars
    ? desc.slice(0, config.maxToolDescChars) + "..."
    : desc;

  const params = tool?.input_schema ?? {};
  const required = Array.isArray(params.required) ? params.required : [];
  const props = params.properties ?? {};
  const propNames = Object.keys(props);

  let paramSummary = "{}";
  if (propNames.length > 0 && propNames.length <= 8) {
    paramSummary = JSON.stringify(props);
  } else if (propNames.length > 8) {
    const summary = {};
    propNames.forEach((key) => {
      const p = props[key];
      summary[key] = p?.type
        ? (p.description && p.description.length <= 40 ? { type: p.type, description: p.description } : { type: p.type })
        : p;
    });
    paramSummary = JSON.stringify(summary);
  }

  return [
    `Tool: ${name}`,
    `Description: ${truncatedDesc}`,
    `Parameters: ${paramSummary}`
  ].join("\n");
}

function buildAnthropicToolPrompt(policy, tools) {
  const allowed = new Set(policy.allowedToolNames);
  const schemas = tools
    .filter((t) => allowed.has(toStringSafe(t?.name)))
    .map(formatAnthropicToolSchema)
    .filter(Boolean);

  if (!schemas.length) return "";

  let prompt = [
    "=== TOOLS ===",
    "",
    schemas.join("\n\n"),
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
    "5. Multi-line params (patch, code, file content) — use CDATA",
  ].join("\n");

  if (policy.mode === "required") {
    prompt += "\n6. For this response, you MUST call at least one tool.";
  }
  if (policy.mode === "forced") {
    prompt += `\n6. For this response, you MUST call exactly this tool: ${policy.forcedName}.`;
  }

  return prompt;
}

function injectAnthropicToolPrompt(messages, tools, policy) {
  if (!policy.allowedToolNames.length) return messages;
  const toolPrompt = buildAnthropicToolPrompt(policy, tools);
  if (!toolPrompt) return messages;

  const systemIndex = messages.findIndex((m) => m.role === "system");
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
    const entry = i < msgs.length - 1 ? "\n\n" + block : block;
    if (used + entry.length > budget) break;
    used += entry.length;
    kept.unshift(msgs[i]);
  }
  return kept;
}

function truncatePrompt(messages, maxChars) {
  const systemMsgs = messages.filter((m) => m.role === "system");
  const nonSystemMsgs = messages.filter((m) => m.role !== "system");

  const systemPart = buildPromptFromMessages(systemMsgs);
  const systemLen = systemPart.length;

  if (systemLen >= maxChars) {
    const minUserBudget = Math.min(2000, Math.floor(maxChars * 0.1));
    const systemBudget = maxChars - minUserBudget;
    const trimmed = systemPart.slice(0, Math.max(0, systemBudget));
    const budget = maxChars - trimmed.length;
    const kept = keepRecentMessages(nonSystemMsgs, budget);
    const result = trimmed + (kept.length ? "\n\n" + buildPromptFromMessages(kept) : "");
    log.warn("prompt", `[anthropic] System prompt trimmed: ${systemLen} → ${trimmed.length} chars, kept ${kept.length} messages, total ${result.length}`);
    return result;
  }

  const budget = maxChars - systemLen;
  const kept = keepRecentMessages(nonSystemMsgs, budget);
  const result = systemPart + "\n\n" + buildPromptFromMessages(kept);

  if (kept.length < nonSystemMsgs.length) {
    log.warn("prompt", `[anthropic] Truncated: dropped ${nonSystemMsgs.length - kept.length} oldest messages, total ${result.length} chars`);
  }

  return result;
}

const ANTHROPIC_MODEL_ALIASES = Object.freeze({
  "claude-sonnet-4-6": "deepseek-chat-expert",
  "claude-opus-4-7": "deepseek-reasoner-expert",
  "claude-haiku-4-5": "deepseek-chat-fast",
  "claude-sonnet-4-5": "deepseek-chat-fast",
  "claude-opus-4-5": "deepseek-reasoner-fast",
  "claude-opus-4": "deepseek-reasoner-fast",
  "claude-3-5-sonnet": "deepseek-chat-fast",
  "claude-3-5-haiku": "deepseek-chat-fast"
});

export function resolveAnthropicModel(modelId) {
  const mappedId = ANTHROPIC_MODEL_ALIASES[modelId] ?? modelId;
  return resolveOpenAiModel(mappedId);
}

export function buildAnthropicPrompt({ messages, system, tools, toolChoice }) {
  const systemMsgs = normalizeSystemPrompt(system);
  const normalizedMessages = normalizeAnthropicMessages(messages);
  const allMessages = [...systemMsgs, ...normalizedMessages];

  const anthropicTools = (tools ?? []).map((t) => ({
    ...t,
    function: { name: t.name, description: t.description, parameters: t.input_schema }
  }));

  const policy = resolveToolChoicePolicy({ tools: anthropicTools, toolChoice });
  const promptMessages = injectAnthropicToolPrompt(allMessages, tools ?? [], policy);

  let prompt = buildPromptFromMessages(promptMessages);

  if (prompt.length > config.maxPromptChars) {
    log.warn("prompt", `[anthropic] Prompt too long: ${prompt.length} chars (limit: ${config.maxPromptChars})`);
    prompt = truncatePrompt(promptMessages, config.maxPromptChars);
  }

  log.debug("prompt", `[anthropic] Final prompt length: ${prompt.length} chars, toolNames: [${policy.allowedToolNames.join(",")}]`);

  return {
    prompt,
    toolChoicePolicy: policy,
    toolNames: policy.allowedToolNames
  };
}

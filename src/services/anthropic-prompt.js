import { buildPromptFromMessages } from "../utils/prompt.js";
import { getToolFunction, getToolName, resolveToolChoicePolicy } from "./openai-tool-policy.js";
import { config } from "../config.js";
import { log } from "../utils/log.js";
import { toStringSafe, toJsonText } from "../utils/safe-string.js";
import { resolveOpenAiModel } from "./openai-request.js";

/* ── Shared truncation helpers ── */

function getToolResultBudget(totalContextChars) {
  if (totalContextChars > 100000) return 4000;
  if (totalContextChars > 60000) return 6000;
  if (totalContextChars > 30000) return 10000;
  return 12000;
}

function getToolTruncationCategory(toolName) {
  const lower = (toolName || "").toLowerCase();
  if (/^(read|read_file|readfile)$/i.test(lower)) return "read";
  if (/^(bash|shell|execute_command|runcommand|run_command|terminal)$/i.test(lower)) return "bash";
  if (/^(search|grep|find|glob|list)$/i.test(lower)) return "search";
  if (/^(write|edit|apply_patch|applypatch|patch)$/i.test(lower)) return "write";
  return "default";
}

const TOOL_TRUNCATION_STRATEGY = {
  read:    { headRatio: 0.5, tailRatio: 0.3 },
  bash:    { headRatio: 0.2, tailRatio: 0.6 },
  search:  { headRatio: 0.7, tailRatio: 0.15 },
  write:   { headRatio: 0.5, tailRatio: 0.3 },
  default: { headRatio: 0.6, tailRatio: 0.4 }
};

function formatTruncationHint(contentLen, headBudget, tailBudget, omitted, toolName) {
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

function truncateToolResult(content, budget, toolName) {
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

let _anthropicContextChars = 0;

/* ── Anthropic → DeepSeek prompt conversion ── */

/* ── Slash-command Skill routing ── */

const CLI_BUILTINS = new Set([
  "clear", "help", "config", "compact", "login", "logout", "fast",
  "worktree", "tasks", "doctor", "status", "ide", "theme", "model",
  "agents", "hooks", "mcp", "context", "workspace", "terminal-setup"
]);

function detectSkillSlashCommand(messages) {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role !== "user") continue;
    const content = typeof messages[i].content === "string" ? messages[i].content : "";
    const preview = content.slice(0, 200).replace(/\n/g, "\\n");
    log.debug("prompt", `[anthropic] detectSkillSlashCommand: msg[${i}] role=user content_len=${content.length} preview="${preview}"`);

    // Strategy A: bare /xxx at start of a line — only in non-tool-result messages.
    // Skip if content has <tool_result> blocks (file paths like /home/langshen/...)
    if (!/<tool_result/.test(content)) {
      const bareMatch = content.match(/^\/([\w-]+)(?![\w\/-])/m);
      if (bareMatch && !CLI_BUILTINS.has(bareMatch[1])) {
        log.debug("prompt", `[anthropic] detectSkillSlashCommand: detected bare /${bareMatch[1]}`);
        return bareMatch[1];
      }
    }

    // Strategy B: <command-name>/xxx</command-name> (Claude Code wraps slash commands in XML)
    const tagRegex = /<command-name>\/([\w-]+)<\/command-name>/g;
    let lastSkillCmd = null;
    let tagMatch;
    while ((tagMatch = tagRegex.exec(content)) !== null) {
      if (!CLI_BUILTINS.has(tagMatch[1])) lastSkillCmd = tagMatch[1];
    }
    if (lastSkillCmd) {
      log.debug("prompt", `[anthropic] detectSkillSlashCommand: detected <command-name>/${lastSkillCmd}`);
      return lastSkillCmd;
    }

    log.debug("prompt", `[anthropic] detectSkillSlashCommand: no skill slash command found in latest user message`);
    return null;
  }
  log.debug("prompt", `[anthropic] detectSkillSlashCommand: no user message found`);
  return null;
}

function injectSkillRoutingHint(allMessages, tools) {
  const hasSkillTool = tools.some(t => t.name === "Skill");
  log.debug("prompt", `[anthropic] injectSkillRoutingHint: hasSkillTool=${hasSkillTool} toolCount=${tools.length}`);
  if (!hasSkillTool) return null;

  const command = detectSkillSlashCommand(allMessages);
  if (!command) return null;

  log.debug("prompt", `[anthropic] Injecting Skill routing hint for /${command}`);

  const hint = [
    "",
    "!!! SLASH COMMAND ROUTING DIRECTIVE !!!",
    `The user just typed /${command}. This is a skill invocation.`,
    `You MUST respond by calling: Skill(skill="${command}")`,
    "Do NOT manually execute the slash command yourself.",
    "Call the Skill tool as your first and only action.",
    "!!! END DIRECTIVE !!!"
  ].join("\n");

  const sysIdx = allMessages.findIndex(m => m.role === "system");
  if (sysIdx >= 0) {
    allMessages[sysIdx] = {
      ...allMessages[sysIdx],
      content: allMessages[sysIdx].content + "\n" + hint
    };
  } else {
    allMessages.unshift({ role: "system", content: hint });
  }

  return command;
}

/* ── Prompt normalization ── */

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
    let content = typeof block.content === "string"
      ? block.content
      : (Array.isArray(block.content) ? block.content.map(c => c?.text ?? "").join("\n") : "");
    // Strip line-number prefixes (Claude Code Read format: "  1→code")
    // so the model doesn't learn to echo file content with line numbers
    content = content.replace(/^\s*\d+→/gm, "");
    // Dynamic budget + head-tail truncation (see openai-tool-prompt.js for rationale)
    const budget = getToolResultBudget(_anthropicContextChars ?? 0);
    const { text: truncated, truncated: wasTruncated } = truncateToolResult(content, budget, "");
    content = truncated;
    const name = toStringSafe(block.tool_use_id).slice(-20);
    return `<tool_result id="${name}">\n${content}\n</tool_result>`;
  }
  return "";
}

function normalizeAnthropicMessages(messages) {
  _anthropicContextChars = 0;

  return (messages ?? []).flatMap((message) => {
    const role = message?.role === "assistant" ? "assistant"
      : message?.role === "user" ? "user"
        : "user";

    const content = message?.content;
    if (typeof content === "string") {
      _anthropicContextChars += content.length;
      return [{ role, content }];
    }
    if (!Array.isArray(content)) return [];

    const textBlocks = content.map(block => {
      const text = contentBlockToText(block);
      _anthropicContextChars += text.length;
      return text;
    }).filter(Boolean);

    if (role === "user") {
      const combined = textBlocks.join("\n") || " ";
      return [{ role: "user", content: combined }];
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
    "6. Never echo <tool_result> content in your visible output — it's internal context",
  ].join("\n");

  if (policy.mode === "required") {
    prompt += "\n7. For this response, you MUST call at least one tool.";
  }
  if (policy.mode === "forced") {
    prompt += `\n7. For this response, you MUST call exactly this tool: ${policy.forcedName}.`;
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

  // Route slash commands to Skill tool (deepseek models may not follow system prompt)
  const skillCommand = injectSkillRoutingHint(allMessages, tools ?? []);

  // Force tool_choice to Skill when slash command detected,
  // otherwise deepseek models will manually simulate the skill instead of delegating
  const effectiveToolChoice = skillCommand
    ? { type: "function", function: { name: "Skill" } }
    : toolChoice;

  if (skillCommand) {
    log.debug("prompt", `[anthropic] Overriding tool_choice to force Skill tool for /${skillCommand}`);
  }

  const anthropicTools = (tools ?? []).map((t) => ({
    ...t,
    function: { name: t.name, description: t.description, parameters: t.input_schema }
  }));

  const policy = resolveToolChoicePolicy({ tools: anthropicTools, toolChoice: effectiveToolChoice });
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
    toolNames: policy.allowedToolNames,
    forcedSkillCommand: skillCommand ?? ""
  };
}

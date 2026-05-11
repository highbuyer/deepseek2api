import { buildPromptFromMessages } from "../utils/prompt.js";
import { getToolFunction, getToolName, resolveToolChoicePolicy } from "./openai-tool-policy.js";
import { config } from "../config.js";
import { log } from "../utils/log.js";
import { toStringSafe, toJsonText } from "../utils/safe-string.js";
import { resolveOpenAiModel } from "./openai-request.js";
import {
  getToolResultBudget, getToolTruncationCategory, TOOL_TRUNCATION_STRATEGY,
  formatTruncationHint, truncateToolResult,
  PER_MESSAGE_TOOL_RESULT_BUDGET, BUDGET_PREVIEW_CHARS, buildBudgetPreview,
  keepRecentMessages, injectReadBloatHint, estimateTokens,
  detectRepeatedToolFailures,
  isWebContentTool, WEB_TOOL_RESULT_BUDGET, sanitizeWebContent
} from "../utils/tool-truncation.js";

/* ── Anthropic → DeepSeek prompt conversion ── */

/* ── Slash-command Skill routing ── */

const CLI_BUILTINS = new Set([
  "clear", "help", "config", "compact", "login", "logout", "fast",
  "worktree", "tasks", "doctor", "status", "ide", "theme", "model",
  "agents", "hooks", "mcp", "context", "workspace", "terminal-setup",
  "effort"
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

function contentBlockToText(block, toolNameById, _ctxChars = 0) {
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
    const toolName = toolNameById?.get(toStringSafe(block.tool_use_id)) || "";
    const isWeb = isWebContentTool(toolName);
    // Web content tools get a much tighter budget + angle-bracket sanitization
    // to prevent HTML/XML pollution from confusing tool call format selection
    const budget = isWeb ? WEB_TOOL_RESULT_BUDGET : getToolResultBudget(_ctxChars ?? 0, config.maxPromptChars);
    const { text: truncated } = truncateToolResult(content, budget, toolName);
    content = isWeb ? sanitizeWebContent(truncated) : truncated;
    // Use tool name as id so injectReadBloatHint / repeated-read regexes can match
    const name = toolName || toStringSafe(block.tool_use_id).slice(-20);
    return `<tool_result id="${name}">\n${content}\n</tool_result>`;
  }
  return "";
}

function normalizeAnthropicMessages(messages) {
  // Build tool_use_id → tool_name map and read-path tracker from all
  // assistant messages for category-aware truncation and repeated-read detection.
  const toolNameById = new Map();
  const readPathByCallId = new Map();
  const readPathCount = new Map();
  for (const message of (messages ?? [])) {
    if (message?.role === "assistant" && Array.isArray(message?.content)) {
      for (const block of message.content) {
        if (block?.type === "tool_use" && block.id) {
          const name = toStringSafe(block.name);
          toolNameById.set(toStringSafe(block.id), name);
          if (/^read$/i.test(name)) {
            const path = toStringSafe(block.input?.file_path || block.input?.path);
            if (path) {
              readPathByCallId.set(toStringSafe(block.id), path);
              readPathCount.set(path, (readPathCount.get(path) || 0) + 1);
            }
          }
        }
      }
    }
  }

  let totalChars = 0;
  let rawTotalChars = 0;  // pre-truncation size for budget decisions

  return (messages ?? []).flatMap((message) => {
    const role = message?.role === "assistant" ? "assistant"
      : message?.role === "user" ? "user"
        : "user";

    const content = message?.content;
    if (typeof content === "string") {
      totalChars += content.length;
      rawTotalChars += content.length;
      return [{ role, content }];
    }
    if (!Array.isArray(content)) return [];

    // Per-message tool result budget (Claude Code enforceToolResultBudget pattern)
    let batchToolResultChars = 0;

    const textBlocks = content.map(block => {
      if (block?.type === "tool_result") {
        const raw = typeof block.content === "string" ? block.content
          : Array.isArray(block.content) ? block.content.map(c => c?.text ?? "").join("\n") : "";
        batchToolResultChars += raw.length;

        // Repeated read detection: if this Read result is for a file already
        // read in this session, replace with short warning so the model
        // references the earlier result instead of wasting context.
        const callId = toStringSafe(block.tool_use_id);
        const toolName = toolNameById.get(callId);
        if (/^read$/i.test(toolName)) {
          const readPath = readPathByCallId.get(callId);
          if (readPath && (readPathCount.get(readPath) || 0) > 1) {
            const count = readPathCount.get(readPath);
            const name = toStringSafe(block.tool_use_id).slice(-20);
            const warning = `⚠️ REPEATED READ #${count}: "${readPath}" has already been read in this session.\nReference the earlier result. If you need a specific section, use Read with offset/limit.\n`;
            const text = `<tool_result id="${name}">\n${warning}\n</tool_result>`;
            log.warn("prompt", `[anthropic] Repeated read detected: "${readPath}" (read ${count} times), replacing result with warning`);
            totalChars += text.length;
            rawTotalChars += raw.length;
            return text;
          }
        }

        if (batchToolResultChars > PER_MESSAGE_TOOL_RESULT_BUDGET) {
          // Replace with preview instead of full content
          const preview = buildBudgetPreview(raw, raw.length, "");
          const name = toStringSafe(block.tool_use_id).slice(-20);
          const text = `<tool_result id="${name}">\n${preview}\n</tool_result>`;
          totalChars += text.length;
          rawTotalChars += raw.length;
          return text;
        }
      }
      // Use rawTotalChars for budget so it tightens with real data volume
      const text = contentBlockToText(block, toolNameById, rawTotalChars);
      totalChars += text.length;
      if (block?.type === "tool_result") {
        const raw = typeof block.content === "string" ? block.content
          : Array.isArray(block.content) ? block.content.map(c => c?.text ?? "").join("\n") : "";
        rawTotalChars += raw.length;
      }
      return text;
    }).filter(Boolean);

    // For non-tool blocks whose length wasn't tracked in rawTotalChars
    if (role === "user" && !content.some(b => b?.type === "tool_result")) {
      textBlocks.forEach(t => { rawTotalChars += t.length; });
    }

    if (role === "user") {
      const combined = textBlocks.join("\n") || " ";
      return [{ role: "user", content: combined }];
    }

    // assistant: keep tool calls as structured info
    const toolCalls = content.filter((b) => b?.type === "tool_use");
    const textContent = content
      .filter((b) => b?.type === "text" || b?.type === "thinking")
      .map(b => contentBlockToText(b, toolNameById))
      .filter(Boolean)
      .join("\n");

    const toolHistory = toolCalls.length
      ? `<tool_calls>\n${toolCalls.map(b => contentBlockToText(b, toolNameById)).filter(Boolean).join("\n")}\n</tool_calls>`
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

  const firstName = policy.allowedToolNames[0] || "ToolName";
  const secondName = policy.allowedToolNames[1] || "ToolName2";
  const args1 = (() => {
    const t = tools.find(t => toStringSafe(t?.name).trim().toLowerCase() === firstName.toLowerCase());
    const props = (t?.input_schema ?? t?.function?.parameters ?? {}).properties ?? {};
    const keys = Object.keys(props);
    if (!keys.length) return "{}";
    const ex = {};
    for (let i = 0; i < Math.min(keys.length, 2); i++) {
      const type = props[keys[i]]?.type || "string";
      if (type === "number" || type === "integer") ex[keys[i]] = 1;
      else if (type === "boolean") ex[keys[i]] = true;
      else ex[keys[i]] = keys[i].includes("path") || keys[i].includes("directory") ? "/project/src" : "value";
    }
    return JSON.stringify(ex);
  })();
  const args2 = (() => {
    const t = tools.find(t => toStringSafe(t?.name).trim().toLowerCase() === secondName.toLowerCase());
    const props = (t?.input_schema ?? t?.function?.parameters ?? {}).properties ?? {};
    const keys = Object.keys(props);
    if (!keys.length) return "{}";
    const ex = {};
    for (let i = 0; i < Math.min(keys.length, 2); i++) {
      const type = props[keys[i]]?.type || "string";
      if (type === "number" || type === "integer") ex[keys[i]] = 1;
      else if (type === "boolean") ex[keys[i]] = true;
      else ex[keys[i]] = keys[i].includes("path") || keys[i].includes("directory") ? "/project/src" : "value";
    }
    return JSON.stringify(ex);
  })();

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
    schemas.join("\n\n"),
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
    "5. NO XML tags — no <function_calls>, <invoke>, <parameter>, <tool_calls>",
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
    "7. NEVER echo <tool_result> content in your visible output — it's internal context",
  ].flat().join("\n");

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


function truncatePrompt(messages, maxBudget, measure) {
  const len = measure || (s => s.length);
  const units = measure ? "tokens" : "chars";
  const systemMsgs = messages.filter((m) => m.role === "system");
  const nonSystemMsgs = messages.filter((m) => m.role !== "system");

  const systemPart = buildPromptFromMessages(systemMsgs);
  const systemLen = len(systemPart);

  if (systemLen >= maxBudget) {
    const minUserBudget = Math.min(2000, Math.floor(maxBudget * 0.1));
    const systemBudget = maxBudget - minUserBudget;
    const trimmed = systemPart.slice(0, Math.max(0, systemBudget));
    const budget = maxBudget - len(trimmed);
    const kept = keepRecentMessages(nonSystemMsgs, budget, measure);
    const result = trimmed + (kept.length ? "\n\n" + buildPromptFromMessages(kept) : "");
    log.warn("prompt", `[anthropic] System prompt trimmed: ${systemLen} → ${len(trimmed)} ${units}, kept ${kept.length} messages, total ${len(result)}`);
    return result;
  }

  const budget = maxBudget - systemLen;
  const kept = keepRecentMessages(nonSystemMsgs, budget, measure);
  const result = systemPart + "\n\n" + buildPromptFromMessages(kept);

  if (kept.length < nonSystemMsgs.length) {
    log.warn("prompt", `[anthropic] Truncated: dropped ${nonSystemMsgs.length - kept.length} oldest messages, total ${len(result)} ${units}`);
  }

  return result;
}

const ANTHROPIC_MODEL_ALIASES = Object.freeze({
  // Chat (default)
  "claude-haiku-4-5": "deepseek-chat-fast",
  "claude-3-5-haiku": "deepseek-chat-fast",
  // Chat (expert)
  "claude-sonnet-4-6": "deepseek-chat-expert",
  "claude-sonnet-4-5": "deepseek-chat-expert",
  "claude-3-5-sonnet": "deepseek-chat-expert",
  // Reasoner (default)
  "claude-opus-4-5": "deepseek-reasoner-fast",
  "claude-opus-4": "deepseek-reasoner-fast",
  // Reasoner (expert)
  "claude-opus-4-7": "deepseek-reasoner-expert",
  // Vision
  "claude-opus-4-7-vision": "deepseek-vision",
  "claude-opus-4-7-vision-fast": "deepseek-vision-fast"
});

export function resolveAnthropicModel(modelId) {
  const mappedId = ANTHROPIC_MODEL_ALIASES[modelId] ?? modelId;
  return resolveOpenAiModel(mappedId);
}

export function buildAnthropicPrompt({ messages, system, tools, toolChoice }) {
  const systemMsgs = normalizeSystemPrompt(system);
  const normalizedMessages = normalizeAnthropicMessages(messages);
  const allMessages = [...systemMsgs, ...normalizedMessages];

  // Loop circuit breaker: if the model is calling the same tool with the
  // same arguments and getting the same error, inject a hard interrupt.
  const loopInterruption = detectRepeatedToolFailures(messages);
  if (loopInterruption) {
    log.warn("prompt", "[anthropic] Repeated failed tool call loop detected — injecting circuit breaker");
    const sysIdx = allMessages.findIndex(m => m.role === "system");
    if (sysIdx >= 0) {
      allMessages.splice(sysIdx + 1, 0, { role: "system", content: loopInterruption });
    } else {
      allMessages.unshift({ role: "system", content: loopInterruption });
    }
  }

  // Route slash commands to Skill tool via prompt hint (deepseek models may not follow system prompt).
  // Do NOT force tool_choice to Skill — unverifiable server-side whether a command is a real skill;
  // forcing on unknown commands produces "Unknown skill" errors with no recovery path.
  const skillCommand = injectSkillRoutingHint(allMessages, tools ?? []);
  const effectiveToolChoice = toolChoice;

  const anthropicTools = (tools ?? []).map((t) => ({
    ...t,
    function: { name: t.name, description: t.description, parameters: t.input_schema }
  }));

  const policy = resolveToolChoicePolicy({ tools: anthropicTools, toolChoice: effectiveToolChoice });
  const promptMessages = injectAnthropicToolPrompt(allMessages, tools ?? [], policy);

  // Check for Read result bloat (Claude Code checkReadResultBloat pattern)
  const hintedMessages = injectReadBloatHint(promptMessages);

  let prompt = buildPromptFromMessages(hintedMessages);

  const promptLen = config.maxPromptTokens ? estimateTokens(prompt) : prompt.length;
  const promptLimit = config.maxPromptTokens || config.maxPromptChars;
  const measure = config.maxPromptTokens ? estimateTokens : undefined;

  if (promptLen > promptLimit) {
    log.warn("prompt", `[anthropic] Prompt too long: ${promptLen} ${config.maxPromptTokens ? "tokens" : "chars"} (limit: ${promptLimit})`);
    prompt = truncatePrompt(hintedMessages, promptLimit, measure);
  }

  log.debug("prompt", `[anthropic] Final prompt length: ${prompt.length} chars, toolNames: [${policy.allowedToolNames.join(",")}]`);

  return {
    prompt,
    toolChoicePolicy: policy,
    toolNames: policy.allowedToolNames,
    forcedSkillCommand: skillCommand ?? ""
  };
}

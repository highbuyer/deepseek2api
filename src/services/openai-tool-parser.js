import { randomUUID } from "node:crypto";
import { log } from "../utils/log.js";

/* ── Single tool-call block patterns (original) ── */
const TOOL_BLOCK_PATTERN = /<(?:[a-z0-9_:-]+:)?(tool_call|function_call|invoke)\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1>/gi;
const TOOL_SELFCLOSE_PATTERN = /<(?:[a-z0-9_:-]+:)?invoke\b([^>]*)\/>/gi;

/* ── Container patterns: <tool_calls>...</tool_calls> (plural) ── */
const TOOL_CALLS_CONTAINER_PATTERN = /<(?:[a-z0-9_:-]+:)?(tool_calls|function_calls)\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1>/gi;

/* ── Sub-element patterns inside a container ── */
const TOOL_NAME_PATTERNS = Object.freeze([
  /<(?:[a-z0-9_:-]+:)?tool_name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?tool_name>/i,
  /<(?:[a-z0-9_:-]+:)?function_name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?function_name>/i,
  /<(?:[a-z0-9_:-]+:)?name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?name>/i,
  /<(?:[a-z0-9_:-]+:)?function\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?function>/i
]);
const TOOL_ARGS_PATTERNS = Object.freeze([
  /<(?:[a-z0-9_:-]+:)?input\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?input>/i,
  /<(?:[a-z0-9_:-]+:)?arguments\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?arguments>/i,
  /<(?:[a-z0-9_:-]+:)?argument\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?argument>/i,
  /<(?:[a-z0-9_:-]+:)?parameters\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?parameters>/i,
  /<(?:[a-z0-9_:-]+:)?parameter\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?parameter>/i,
  /<(?:[a-z0-9_:-]+:)?args\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?args>/i,
  /<(?:[a-z0-9_:-]+:)?params\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?params>/i
]);

/* ── Pattern for individual <tool_call name="..."> blocks inside container ── */
const INNER_TOOL_CALL_PATTERN = /<(?:[a-z0-9_:-]+:)?(tool_call|function_call|invoke)\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1>/gi;
const TOOL_ATTR_PATTERN = /(name|function|tool)\s*=\s*"([^"]+)"/i;
const TOOL_KV_PATTERN = /<(?:[a-z0-9_:-]+:)?([a-z0-9_.-]+)\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1>/gi;

/* ── Pattern to split container inner into individual tool entries ── */
const TOOL_ENTRY_SPLIT_PATTERN = /<tool_name>|<function_name>|<name>/i;

function toStringSafe(value) {
  if (typeof value === "string") {
    return value;
  }

  if (value === null || value === undefined) {
    return "";
  }

  return String(value);
}

/**
 * Strip null characters (\x00) and other control chars that corrupt XML parsing.
 * Keeps newline, tab, carriage return.
 */
function sanitizeControlChars(text) {
  return toStringSafe(text).replace(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/g, "");
}

function stripFencedCodeBlocks(text) {
  return toStringSafe(text).replace(/```[\s\S]*?```/g, " ");
}

function decodeXmlText(text) {
  const raw = toStringSafe(text).trim();
  const cdataMatch = raw.match(/^<!\[CDATA\[([\s\S]*?)]]>$/i);
  const source = cdataMatch?.[1] ?? raw;
  return source
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", "\"")
    .replaceAll("&#039;", "'")
    .replaceAll("&#x27;", "'");
}

function parseJsonObject(text) {
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch {
    return null;
  }
}

function findTagValue(text, patterns) {
  const source = toStringSafe(text);

  for (const pattern of patterns) {
    const match = source.match(pattern);
    if (match?.[1] !== undefined) {
      return decodeXmlText(match[1]);
    }
  }

  return "";
}

/**
 * Find ALL matches of a tag pattern (not just the first one).
 * Returns array of decoded values.
 */
function findAllTagValues(text, patterns) {
  const source = toStringSafe(text);
  const results = [];

  for (const pattern of patterns) {
    // Reset lastIndex for global patterns
    const re = new RegExp(pattern.source, pattern.flags.includes("g") ? pattern.flags : pattern.flags + "g");
    let match;
    while ((match = re.exec(source)) !== null) {
      if (match[1] !== undefined) {
        results.push(decodeXmlText(match[1]));
      }
    }
    if (results.length) break; // Use first pattern that matches
  }

  return results;
}

function appendMarkupValue(output, key, value) {
  if (!Object.hasOwn(output, key)) {
    output[key] = value;
    return;
  }

  const current = output[key];
  output[key] = Array.isArray(current) ? [...current, value] : [current, value];
}

function parseMarkupValue(raw) {
  const text = decodeXmlText(raw);

  if (!text.trim()) {
    return "";
  }

  if (text.includes("<") && text.includes(">")) {
    const nested = parseMarkupInput(text);
    if (nested && Object.keys(nested).length > 0) {
      return nested;
    }
  }

  const parsedJson = parseJsonObject(text);
  if (parsedJson) {
    return parsedJson;
  }

  try {
    return JSON.parse(text);
  } catch {
    return text;
  }
}

function parseMarkupObject(text) {
  const output = {};

  for (const match of toStringSafe(text).matchAll(TOOL_KV_PATTERN)) {
    const key = toStringSafe(match[1]).trim();
    if (!key) {
      continue;
    }

    appendMarkupValue(output, key, parseMarkupValue(match[2]));
  }

  return output;
}

function parseMarkupInput(raw) {
  const text = decodeXmlText(raw);
  const markupObject = parseMarkupObject(text);

  if (Object.keys(markupObject).length > 0) {
    return markupObject;
  }

  return parseJsonObject(text) ?? {};
}

function buildParsedToolCall(name, argumentsText) {
  const normalizedArguments = argumentsText.trim() ? argumentsText.trim() : "{}";
  return {
    id: `call_${randomUUID().replaceAll("-", "")}`,
    name,
    argumentsText: normalizedArguments,
    input: parseJsonObject(normalizedArguments) ?? parseMarkupInput(normalizedArguments)
  };
}

function parseMarkupBlock(attrs, inner) {
  log.debug("parser", `[parseMarkupBlock] attrs="${attrs.slice(0, 100)}", inner length=${inner.length}, inner preview="${inner.slice(0, 200)}"`);

  const jsonTool = parseJsonObject(inner);
  if (jsonTool?.name) {
    log.debug("parser", `[parseMarkupBlock] Parsed as JSON tool: name=${jsonTool.name}`);
    return buildParsedToolCall(jsonTool.name, JSON.stringify(jsonTool.input ?? {}));
  }

  const attrName = attrs.match(TOOL_ATTR_PATTERN)?.[2] ?? "";
  const name = attrName.trim() || findTagValue(inner, TOOL_NAME_PATTERNS).trim();
  log.debug("parser", `[parseMarkupBlock] attrName="${attrName}", resolved name="${name}"`);

  if (!name) {
    log.debug("parser", `[parseMarkupBlock] No tool name found, returning null`);
    return null;
  }

  const argsRaw = findTagValue(inner, TOOL_ARGS_PATTERNS);
  const parsedInput = argsRaw ? parseMarkupInput(argsRaw) : parseMarkupObject(inner);
  const argumentsText = JSON.stringify(parsedInput && Object.keys(parsedInput).length ? parsedInput : {});

  log.debug("parser", `[parseMarkupBlock] name="${name}", argsRaw="${(argsRaw || "").slice(0, 100)}", argumentsText="${argumentsText.slice(0, 100)}"`);
  return buildParsedToolCall(name, argumentsText);
}

/**
 * Parse individual <tool_call name="..."> or <invoke name="..."> blocks.
 * (Original logic, unchanged)
 */
function parseSingularToolBlocks(text) {
  const output = [];
  const source = toStringSafe(text).trim();

  for (const match of source.matchAll(TOOL_BLOCK_PATTERN)) {
    const parsed = parseMarkupBlock(toStringSafe(match[2]).trim(), toStringSafe(match[3]).trim());
    if (parsed) {
      output.push(parsed);
    }
  }

  for (const match of source.matchAll(TOOL_SELFCLOSE_PATTERN)) {
    const parsed = parseMarkupBlock(toStringSafe(match[1]).trim(), "");
    if (parsed) {
      output.push(parsed);
    }
  }

  return output;
}

/**
 * Parse <tool_calls> container format.
 * The model outputs:
 *   <tool_calls>
 *     <tool_name>terminal</tool_name>
 *     <parameters>{"command":"ls"}</parameters>
 *   </tool_calls>
 *
 * Or with multiple tools:
 *   <tool_calls>
 *     <tool_name>tool1</tool_name>
 *     <parameters>{...}</parameters>
 *     <tool_name>tool2</tool_name>
 *     <parameters>{...}</parameters>
 *   </tool_calls>
 *
 * Or with nested <tool_call name="..."> inside:
 *   <tool_calls>
 *     <tool_call name="tool1">...</tool_call
 *     <tool_call name="tool2">...</tool_call
 *   </tool_calls>
 */
function parseContainerToolBlocks(text) {
  const output = [];
  const source = toStringSafe(text).trim();

  for (const match of source.matchAll(TOOL_CALLS_CONTAINER_PATTERN)) {
    const containerTag = match[1]; // "tool_calls" or "function_calls"
    const attrs = toStringSafe(match[2]).trim();
    const inner = toStringSafe(match[3]).trim();

    log.debug("parser", `[parseContainer] Found <${containerTag}> container, inner length=${inner.length}, inner preview="${inner.slice(0, 300)}"`);

    // First: check if inner has nested <tool_call name="..."> blocks
    const innerCalls = parseSingularToolBlocks(inner);
    if (innerCalls.length) {
      log.debug("parser", `[parseContainer] Found ${innerCalls.length} nested <tool_call/> block(s) inside <${containerTag}>`);
      output.push(...innerCalls);
      continue;
    }

    // Second: check if inner is a single JSON object with .name
    const jsonTool = parseJsonObject(inner);
    if (jsonTool?.name) {
      log.debug("parser", `[parseContainer] Found JSON tool inside <${containerTag}>: name=${jsonTool.name}`);
      output.push(buildParsedToolCall(jsonTool.name, JSON.stringify(jsonTool.input ?? {})));
      continue;
    }

    // Third: extract <tool_name> + <parameters> pairs directly from inner
    // Handle multiple tools inside one container
    const names = findAllTagValues(inner, TOOL_NAME_PATTERNS);
    const args = findAllTagValues(inner, TOOL_ARGS_PATTERNS);

    if (names.length) {
      log.debug("parser", `[parseContainer] Found ${names.length} <tool_name> tag(s): [${names.join(", ")}], ${args.length} <parameters> tag(s)`);

      for (let i = 0; i < names.length; i++) {
        const toolName = names[i].trim();
        const toolArgs = i < args.length ? args[i].trim() : "{}";

        if (!toolName) continue;

        // Try to parse args as JSON, fallback to markup
        let parsedArgs;
        const jsonArgs = parseJsonObject(toolArgs);
        if (jsonArgs) {
          parsedArgs = JSON.stringify(jsonArgs);
        } else {
          parsedArgs = toolArgs;
        }

        log.debug("parser", `[parseContainer] Building tool call #${i}: name="${toolName}", args="${parsedArgs.slice(0, 100)}"`);
        output.push(buildParsedToolCall(toolName, parsedArgs));
      }
      continue;
    }

    // Fourth: try name from attribute + <parameters> as body
    const attrName = attrs.match(TOOL_ATTR_PATTERN)?.[2] ?? "";
    if (attrName.trim()) {
      const argsRaw = findTagValue(inner, TOOL_ARGS_PATTERNS);
      const parsedInput = argsRaw ? parseMarkupInput(argsRaw) : parseMarkupObject(inner);
      const argumentsText = JSON.stringify(parsedInput && Object.keys(parsedInput).length ? parsedInput : {});
      log.debug("parser", `[parseContainer] Using attr name="${attrName}" from <${containerTag}> tag`);
      output.push(buildParsedToolCall(attrName.trim(), argumentsText));
      continue;
    }

    log.warn("parser", `[parseContainer] <${containerTag}> container found but no tool calls could be extracted. Inner (first 300): "${inner.slice(0, 300)}"`);
  }

  return output;
}

/**
 * Main parser: try both singular blocks and container blocks.
 */
function parseMarkupToolCalls(text) {
  const output = [];
  const source = toStringSafe(text).trim();

  // Pass 1: singular <tool_call name="..."> blocks
  const singularCalls = parseSingularToolBlocks(source);
  if (singularCalls.length) {
    log.debug("parser", `[parseMarkupToolCalls] Pass 1 (singular blocks): found ${singularCalls.length} call(s)`);
    output.push(...singularCalls);
  }

  // Pass 2: container <tool_calls>...</tool_calls> blocks
  const containerCalls = parseContainerToolBlocks(source);
  if (containerCalls.length) {
    log.debug("parser", `[parseMarkupToolCalls] Pass 2 (container blocks): found ${containerCalls.length} call(s)`);
    output.push(...containerCalls);
  }

  if (!output.length) {
    log.debug("parser", `[parseMarkupToolCalls] No tool calls found in either pass. Source length=${source.length}, source preview="${source.slice(0, 300)}"`);
  }

  return output;
}

function filterAllowedToolCalls(calls, allowedToolNames) {
  if (!allowedToolNames?.length) {
    return calls;
  }

  const allowed = new Set(allowedToolNames.map((name) => toStringSafe(name).trim()).filter(Boolean));
  const filtered = calls.filter((call) => allowed.has(call.name));
  if (filtered.length !== calls.length) {
    const rejected = calls.filter((call) => !allowed.has(call.name)).map((call) => call.name);
    log.warn("parser", `Filtered out ${rejected.length} tool call(s) not in allowed list: [${rejected.join(", ")}] (allowed: [${allowedToolNames.join(",")}])`);
  }
  return filtered;
}

export function parseToolCallsFromText(text, allowedToolNames = []) {
  const rawSource = toStringSafe(text);
  if (!rawSource.trim()) {
    return [];
  }

  // Step 1: Sanitize control chars (null bytes, etc.)
  const source = sanitizeControlChars(rawSource);
  const nullCount = rawSource.length - source.length;
  if (nullCount > 0) {
    log.warn("parser", `Stripped ${nullCount} control character(s) from input (including null bytes)`);
  }

  // Step 2: Quick check for any tool-related XML tags
  const stripped = stripFencedCodeBlocks(source);
  if (!stripped.match(/<(tool_calls|tool_call|function_calls|function_call|invoke|tool_use)\b/i)) {
    log.debug("parser", `No tool XML tags found in output (length=${source.length})`);
    return [];
  }

  log.debug("parser", `Tool XML tags detected. Full source length=${source.length}, first 500 chars: "${source.slice(0, 500)}"`);

  // Step 3: Parse
  const calls = filterAllowedToolCalls(parseMarkupToolCalls(source), allowedToolNames);
  log.debug("parser", `Parsed ${calls.length} tool call(s) from text (allowed: [${allowedToolNames.join(",")}])`);
  if (calls.length) {
    calls.forEach((call, i) => {
      log.debug("parser", `  Tool call #${i}: name="${call.name}", argumentsText="${call.argumentsText.slice(0, 100)}"`);
    });
  } else {
    log.warn("parser", `XML tool tags found but no valid calls parsed. Stripped content (first 500): ${stripped.slice(0, 500)}`);
  }
  return calls;
}

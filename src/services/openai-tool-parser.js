import { randomUUID } from "node:crypto";
import { log } from "../utils/log.js";

/* ── Single tool-call block patterns ── */
const TOOL_BLOCK_PATTERN = /<(?:[a-z0-9_:-]+:)?(tool_call|function_call|invoke)\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1\s*>/gi;
const TOOL_SELFCLOSE_PATTERN = /<(?:[a-z0-9_:-]+:)?invoke\b([^>]*)\/>/gi;

/* ── Container patterns: <tool_calls>...</tool_calls> (plural) ── */
const TOOL_CALLS_CONTAINER_PATTERN = /<(?:[a-z0-9_:-]+:)?(tool_calls|function_calls)\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1\s*>/gi;

/* ── Sub-element patterns ── */
const TOOL_NAME_PATTERNS = Object.freeze([
  /<(?:[a-z0-9_:-]+:)?tool_name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?tool_name>/i,
  /<(?:[a-z0-9_:-]+:)?function_name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?function_name>/i,
  /<(?:[a-z0-9_:-]+:)?name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?name>/i,
  /<(?:[a-z0-9_:-]+:)?function\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?function>/i
]);

/* ── Args patterns: <parameters>, <arguments>, <input> etc. (JSON body) ── */
const TOOL_ARGS_PATTERNS = Object.freeze([
  /<(?:[a-z0-9_:-]+:)?parameters\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?parameters>/i,
  /<(?:[a-z0-9_:-]+:)?input\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?input>/i,
  /<(?:[a-z0-9_:-]+:)?arguments\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?arguments>/i,
  /<(?:[a-z0-9_:-]+:)?argument\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?argument>/i,
  /<(?:[a-z0-9_:-]+:)?args\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?args>/i,
  /<(?:[a-z0-9_:-]+:)?params\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?params>/i
]);

/* ── <parameter name="key">value</parameter> pattern (singular with name attr) ── */
const PARAMETER_NAME_ATTR_PATTERN = /<(?:[a-z0-9_:-]+:)?parameter\b[^>]*\bname\s*=\s*"([^"]+)"[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?parameter>/gi;

const TOOL_ATTR_PATTERN = /(name|function|tool)\s*=\s*"([^"]+)"/i;
const TOOL_KV_PATTERN = /<(?:[a-z0-9_:-]+:)?([a-z0-9_.-]+)\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1\s*>/gi;

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

/**
 * Parse <parameter name="key">value</parameter> elements into a JSON object.
 * Handles the format the model sometimes uses:
 *   <parameter name="command">ls -la</parameter>
 *   <parameter name="timeout">5</parameter>
 *   → {"command": "ls -la", "timeout": "5"}
 */
function parseParameterNameAttrs(text) {
  const source = toStringSafe(text);
  const output = {};
  let found = false;

  for (const match of source.matchAll(PARAMETER_NAME_ATTR_PATTERN)) {
    const key = toStringSafe(match[1]).trim();
    const value = decodeXmlText(match[2]);
    if (!key) continue;

    found = true;
    // Try to parse value as JSON (for nested objects, numbers, booleans)
    const jsonValue = parseJsonObject(value);
    if (jsonValue) {
      output[key] = jsonValue;
    } else {
      // Try number / boolean
      if (/^-?\d+(\.\d+)?$/.test(value.trim())) {
        output[key] = Number(value.trim());
      } else if (value.trim() === "true") {
        output[key] = true;
      } else if (value.trim() === "false") {
        output[key] = false;
      } else {
        output[key] = value;
      }
    }
  }

  return found ? output : null;
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

/**
 * Parse a single tool_call block's inner content.
 * Handles multiple argument formats:
 * 1. JSON body: <tool_call name="x">{"key":"val"}</tool_call
 * 2. <parameters>{"key":"val"}</parameters> (JSON body)
 * 3. <parameter name="key">value</parameter> (named attrs)
 * 4. Markup KV: <key>value</key> children
 */
function parseToolCallInner(attrs, inner) {
  log.debug("parser", `[parseToolCallInner] attrs="${attrs.slice(0, 100)}", inner length=${inner.length}, inner preview="${inner.slice(0, 200)}"`);

  // 1. Entire inner is a JSON object with .name
  const jsonTool = parseJsonObject(inner);
  if (jsonTool?.name) {
    log.debug("parser", `[parseToolCallInner] Parsed as JSON tool: name=${jsonTool.name}`);
    return buildParsedToolCall(jsonTool.name, JSON.stringify(jsonTool.input ?? jsonTool.arguments ?? {}));
  }

  // 2. Get tool name from attribute or <tool_name> sub-tag
  const attrName = attrs.match(TOOL_ATTR_PATTERN)?.[2] ?? "";
  const name = attrName.trim() || findTagValue(inner, TOOL_NAME_PATTERNS).trim();
  log.debug("parser", `[parseToolCallInner] attrName="${attrName}", resolved name="${name}"`);

  if (!name) {
    log.debug("parser", `[parseToolCallInner] No tool name found, returning null`);
    return null;
  }

  // 3. Try <parameter name="key">value</parameter> format FIRST
  //    (before <parameters>JSON</parameters> because <parameter> with name attr
  //     gets matched by TOOL_KV_PATTERN as a generic tag, losing the name attribute)
  const namedParams = parseParameterNameAttrs(inner);
  if (namedParams && Object.keys(namedParams).length > 0) {
    const argumentsText = JSON.stringify(namedParams);
    log.debug("parser", `[parseToolCallInner] name="${name}", parsed via <parameter name="..."> format, argumentsText="${argumentsText.slice(0, 100)}"`);
    return buildParsedToolCall(name, argumentsText);
  }

  // 4. Try <parameters>JSON</parameters> or <arguments>JSON</arguments> format
  const argsRaw = findTagValue(inner, TOOL_ARGS_PATTERNS);
  if (argsRaw) {
    const parsedInput = parseMarkupInput(argsRaw);
    const argumentsText = JSON.stringify(parsedInput && Object.keys(parsedInput).length ? parsedInput : {});
    log.debug("parser", `[parseToolCallInner] name="${name}", argsRaw="${argsRaw.slice(0, 100)}", argumentsText="${argumentsText.slice(0, 100)}"`);
    return buildParsedToolCall(name, argumentsText);
  }

  // 5. Fallback: parse inner as generic markup KV
  const markupObject = parseMarkupObject(inner);
  const argumentsText = JSON.stringify(markupObject && Object.keys(markupObject).length ? markupObject : {});
  log.debug("parser", `[parseToolCallInner] name="${name}", fallback markup, argumentsText="${argumentsText.slice(0, 100)}"`);
  return buildParsedToolCall(name, argumentsText);
}

/**
 * Parse standalone <tool_call name="..."> blocks that are NOT inside a <tool_calls> container.
 * Only parses blocks that are direct children of the source text.
 */
function parseStandaloneSingularBlocks(text) {
  const output = [];
  const source = toStringSafe(text).trim();

  for (const match of source.matchAll(TOOL_BLOCK_PATTERN)) {
    const parsed = parseToolCallInner(toStringSafe(match[2]).trim(), toStringSafe(match[3]).trim());
    if (parsed) {
      output.push(parsed);
    }
  }

  for (const match of source.matchAll(TOOL_SELFCLOSE_PATTERN)) {
    const parsed = parseToolCallInner(toStringSafe(match[1]).trim(), "");
    if (parsed) {
      output.push(parsed);
    }
  }

  return output;
}

/**
 * Parse <tool_calls> container format.
 * Supports:
 *   Format A: <tool_call name="..."> nested inside container
 *   Format B: <tool_name>+<parameters> flat pairs inside container
 *   Format C: <tool_name>+<parameter name="..."> named params inside container
 *   Format D: Single JSON with .name inside container
 */
function parseContainerToolBlocks(text) {
  const output = [];
  const source = toStringSafe(text).trim();

  for (const match of source.matchAll(TOOL_CALLS_CONTAINER_PATTERN)) {
    const containerTag = match[1]; // "tool_calls" or "function_calls"
    const attrs = toStringSafe(match[2]).trim();
    const inner = toStringSafe(match[3]).trim();

    log.debug("parser", `[parseContainer] Found <${containerTag}> container, inner length=${inner.length}, inner preview="${inner.slice(0, 300)}"`);

    // Strategy A: nested <tool_call name="..."> blocks inside container
    const innerCalls = parseStandaloneSingularBlocks(inner);
    if (innerCalls.length) {
      log.debug("parser", `[parseContainer] Found ${innerCalls.length} nested <tool_call/> block(s) inside <${containerTag}>`);
      output.push(...innerCalls);
      continue;
    }

    // Strategy B: <tool_name>+<parameters> flat pairs
    const names = findAllTagValues(inner, TOOL_NAME_PATTERNS);
    if (names.length) {
      // Check for <parameter name="..."> named params first
      const namedParams = parseParameterNameAttrs(inner);
      if (namedParams && Object.keys(namedParams).length > 0) {
        // Format C: <tool_name>X</tool_name> + <parameter name="key">value</parameter>
        // If multiple tool_names, we need to split the params per tool
        // For now, if only one tool_name, assign all named params to it
        if (names.length === 1) {
          const toolName = names[0].trim();
          const argumentsText = JSON.stringify(namedParams);
          log.debug("parser", `[parseContainer] Format C: name="${toolName}", <parameter name="..."> args="${argumentsText.slice(0, 100)}"`);
          output.push(buildParsedToolCall(toolName, argumentsText));
          continue;
        }
        // Multiple tools with named params is ambiguous, fall through
      }

      // Format B: <tool_name>+<parameters> pairs
      const args = findAllTagValues(inner, TOOL_ARGS_PATTERNS);
      log.debug("parser", `[parseContainer] Format B: ${names.length} <tool_name> tag(s): [${names.join(", ")}], ${args.length} <parameters> tag(s)`);

      for (let i = 0; i < names.length; i++) {
        const toolName = names[i].trim();
        const toolArgs = i < args.length ? args[i].trim() : "{}";

        if (!toolName) continue;

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

    // Strategy C: inner is a single JSON object with .name
    const jsonTool = parseJsonObject(inner);
    if (jsonTool?.name) {
      log.debug("parser", `[parseContainer] Found JSON tool inside <${containerTag}>: name=${jsonTool.name}`);
      output.push(buildParsedToolCall(jsonTool.name, JSON.stringify(jsonTool.input ?? jsonTool.arguments ?? {})));
      continue;
    }

    // Strategy D: name from container tag attribute + body as args
    const attrName = attrs.match(TOOL_ATTR_PATTERN)?.[2] ?? "";
    if (attrName.trim()) {
      const namedParams = parseParameterNameAttrs(inner);
      if (namedParams && Object.keys(namedParams).length > 0) {
        const argumentsText = JSON.stringify(namedParams);
        log.debug("parser", `[parseContainer] Strategy D: attr name="${attrName}", <parameter name="..."> args="${argumentsText.slice(0, 100)}"`);
        output.push(buildParsedToolCall(attrName.trim(), argumentsText));
      } else {
        const argsRaw = findTagValue(inner, TOOL_ARGS_PATTERNS);
        const parsedInput = argsRaw ? parseMarkupInput(argsRaw) : parseMarkupObject(inner);
        const argumentsText = JSON.stringify(parsedInput && Object.keys(parsedInput).length ? parsedInput : {});
        log.debug("parser", `[parseContainer] Strategy D: attr name="${attrName}", argumentsText="${argumentsText.slice(0, 100)}"`);
        output.push(buildParsedToolCall(attrName.trim(), argumentsText));
      }
      continue;
    }

    log.warn("parser", `[parseContainer] <${containerTag}> container found but no tool calls could be extracted. Inner (first 300): "${inner.slice(0, 300)}"`);
  }

  return output;
}

/**
 * Main parser: Parse tool calls from text.
 *
 * Key logic: If there are <tool_calls> container blocks, we ONLY parse those
 * (because standalone <tool_call name="..."> blocks inside the container will
 * be parsed by the container handler, and parsing them again would cause duplicates).
 *
 * If there are NO container blocks, then parse standalone <tool_call name="..."> blocks.
 */
function parseMarkupToolCalls(text) {
  const source = toStringSafe(text).trim();

  // Check if source contains container blocks
  const hasContainer = TOOL_CALLS_CONTAINER_PATTERN.test(source);
  // Reset lastIndex after test
  TOOL_CALLS_CONTAINER_PATTERN.lastIndex = 0;

  if (hasContainer) {
    // Only parse container blocks (they will handle nested <tool_call/> inside)
    const containerCalls = parseContainerToolBlocks(source);
    log.debug("parser", `[parseMarkupToolCalls] Container mode: found ${containerCalls.length} call(s)`);
    return containerCalls;
  }

  // No containers: parse standalone singular blocks
  const standaloneCalls = parseStandaloneSingularBlocks(source);
  log.debug("parser", `[parseMarkupToolCalls] Standalone mode: found ${standaloneCalls.length} call(s)`);
  return standaloneCalls;
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

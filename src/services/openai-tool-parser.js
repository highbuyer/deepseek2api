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

/* ── Known structural tag names (not tool names) ── */
const KNOWN_STRUCTURAL_TAGS = new Set([
  "tool_calls", "tool_call", "function_calls", "function_call", "invoke", "tool_use",
  "tool_name", "function_name", "name", "function",
  "parameters", "parameter", "input", "arguments", "argument", "args", "params",
  "command", "query", "body", "data", "result", "response", "content", "value",
  "description", "type", "key", "url", "path", "file", "text", "message",
  "output", "error", "code", "script", "shell", "timeout", "limit"
]);

const TOOL_ATTR_PATTERN = /(name|function|tool)\s*=\s*"([^"]+)"/i;
const TOOL_KV_PATTERN = /<(?:[a-z0-9_:-]+:)?([a-z0-9_.-]+)\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1\s*>/gi;
const ANY_TAG_PATTERN = /<([a-z0-9_:-]+)\b[^>]*>/gi;

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

/**
 * Extract the "name" attribute from a tag like <tool_name name="TodoWrite">.
 * Returns the attribute value if found, otherwise empty string.
 */
function extractNameAttrFromTag(text, tagName) {
  const re = new RegExp(`<(?:[a-z0-9_:-]+:)?${tagName}\\b([^>]*)>`, "i");
  const match = toStringSafe(text).match(re);
  if (!match?.[1]) return "";
  const attrMatch = match[1].match(/name\s*=\s*"([^"]+)"/i);
  return attrMatch?.[1]?.trim() ?? "";
}

function findTagValue(text, patterns) {
  const source = toStringSafe(text);

  for (const pattern of patterns) {
    const match = source.match(pattern);
    if (match?.[1] !== undefined) {
      const textContent = decodeXmlText(match[1]).trim();
      if (textContent) return textContent;
    }
  }

  // If text content is empty, check for name attribute on <tool_name> / <function_name> / <name> tags
  for (const tagName of ["tool_name", "function_name", "name"]) {
    const attrVal = extractNameAttrFromTag(source, tagName);
    if (attrVal) return attrVal;
  }

  return "";
}

/**
 * Find ALL matches of a tag pattern (not just the first one).
 * Returns array of decoded values.
 * Also checks name= attribute on matched tags when text content is empty.
 */
function findAllTagValues(text, patterns) {
  const source = toStringSafe(text);
  const results = [];

  for (const pattern of patterns) {
    const re = new RegExp(pattern.source, pattern.flags.includes("g") ? pattern.flags : pattern.flags + "g");
    let match;
    while ((match = re.exec(source)) !== null) {
      if (match[1] !== undefined) {
        const textContent = decodeXmlText(match[1]).trim();
        if (textContent) {
          results.push(textContent);
        } else {
          // Text content empty — try name= attribute on the opening tag
          const fullMatch = match[0];
          const attrVal = fullMatch.match(/name\s*=\s*"([^"]+)"/i)?.[1]?.trim();
          if (attrVal) results.push(attrVal);
        }
      }
    }
    if (results.length) break;
  }

  return results;
}

/**
 * Find tool names by scanning for child tags whose name is in allowedToolNames.
 * This handles the format where the model uses the tool name directly as a tag:
 *   <terminal>
 *     <parameters>...</parameters>
 *   </terminal>
 * Here <terminal> IS the tool name.
 *
 * Returns { name, body } objects for each matched tool tag.
 */
function findToolNameTags(text, allowedToolNames) {
  if (!allowedToolNames?.length) return [];

  const allowed = new Set(allowedToolNames.map(n => toStringSafe(n).trim().toLowerCase()).filter(Boolean));
  const source = toStringSafe(text);
  const results = [];

  for (const match of source.matchAll(ANY_TAG_PATTERN)) {
    const tagName = toStringSafe(match[1]).trim().toLowerCase();
    if (!tagName) continue;

    // Skip known structural tags
    if (KNOWN_STRUCTURAL_TAGS.has(tagName)) continue;

    // Check if this tag name is in allowed list
    if (!allowed.has(tagName)) continue;

    // Found a tool-name tag! Extract its full content.
    const openTag = match[0];
    const openPos = match.index;
    const closeTag = `</${match[1]}`;
    const closePos = source.indexOf(closeTag, openPos + openTag.length);

    if (closePos < 0) continue;

    // Find the actual end of the closing tag
    const closeEnd = source.indexOf(">", closePos) + 1;
    const body = source.slice(openPos + openTag.length, closePos);

    // Find the original-cased tag name from allowedToolNames
    const originalName = allowedToolNames.find(n => toStringSafe(n).trim().toLowerCase() === tagName) || tagName;

    results.push({ name: originalName, body: body.trim() });
  }

  return results;
}

/**
 * Parse <parameter name="key">value</parameter> elements into a JSON object.
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
    const jsonValue = parseJsonObject(value);
    if (jsonValue) {
      output[key] = jsonValue;
    } else if (/^-?\d+(\.\d+)?$/.test(value.trim())) {
      output[key] = Number(value.trim());
    } else if (value.trim() === "true") {
      output[key] = true;
    } else if (value.trim() === "false") {
      output[key] = false;
    } else {
      output[key] = value;
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
 * Parse arguments from a tool call body.
 * Tries multiple formats:
 * 1. <parameter name="key">value</parameter>
 * 2. <parameters>{JSON}</parameters>
 * 3. Generic markup KV
 */
function parseToolCallArguments(body) {
  // 1. Try <parameter name="key">value</parameter> format
  const namedParams = parseParameterNameAttrs(body);
  if (namedParams && Object.keys(namedParams).length > 0) {
    return JSON.stringify(namedParams);
  }

  // 2. Try <parameters>JSON</parameters> format
  const argsRaw = findTagValue(body, TOOL_ARGS_PATTERNS);
  if (argsRaw) {
    const parsedInput = parseMarkupInput(argsRaw);
    if (parsedInput && Object.keys(parsedInput).length > 0) {
      return JSON.stringify(parsedInput);
    }
  }

  // 3. Fallback: parse body as generic markup KV
  const markupObject = parseMarkupObject(body);
  if (markupObject && Object.keys(markupObject).length > 0) {
    return JSON.stringify(markupObject);
  }

  return "{}";
}

/**
 * Parse a single tool_call block's inner content.
 * Now accepts allowedToolNames to support "tag-name-as-tool-name" format.
 */
function parseToolCallInner(attrs, inner, allowedToolNames = []) {
  log.debug("parser", `[parseToolCallInner] attrs="${attrs.slice(0, 100)}", inner length=${inner.length}, inner preview="${inner.slice(0, 200)}"`);

  // 1. Entire inner is a JSON object with .name
  const jsonTool = parseJsonObject(inner);
  if (jsonTool?.name) {
    log.debug("parser", `[parseToolCallInner] Parsed as JSON tool: name=${jsonTool.name}`);
    return buildParsedToolCall(jsonTool.name, JSON.stringify(jsonTool.input ?? jsonTool.arguments ?? {}));
  }

  // 2. Get tool name from attribute or <tool_name> sub-tag
  const attrName = attrs.match(TOOL_ATTR_PATTERN)?.[2] ?? "";
  let name = attrName.trim() || findTagValue(inner, TOOL_NAME_PATTERNS).trim();
  log.debug("parser", `[parseToolCallInner] attrName="${attrName}", resolved name="${name}"`);

  // 3. If no name found, try "tag-name-as-tool-name" format
  //    e.g. <terminal><parameters>...</parameters></terminal>
  if (!name && allowedToolNames.length) {
    const toolNameTags = findToolNameTags(inner, allowedToolNames);
    if (toolNameTags.length) {
      const first = toolNameTags[0];
      name = first.name;
      const argumentsText = parseToolCallArguments(first.body);
      log.debug("parser", `[parseToolCallInner] Found tool name via tag-name match: name="${name}", body="${first.body.slice(0, 100)}", argumentsText="${argumentsText.slice(0, 100)}"`);
      return buildParsedToolCall(name, argumentsText);
    }
  }

  if (!name) {
    log.debug("parser", `[parseToolCallInner] No tool name found (tried attr, tool_name tags, and tag-name matching), returning null`);
    return null;
  }

  // 4. Parse arguments
  const argumentsText = parseToolCallArguments(inner);
  log.debug("parser", `[parseToolCallInner] name="${name}", argumentsText="${argumentsText.slice(0, 100)}"`);
  return buildParsedToolCall(name, argumentsText);
}

/**
 * Parse standalone <tool_call name="..."> blocks.
 */
function parseStandaloneSingularBlocks(text, allowedToolNames = []) {
  const output = [];
  const source = toStringSafe(text).trim();

  for (const match of source.matchAll(TOOL_BLOCK_PATTERN)) {
    const parsed = parseToolCallInner(toStringSafe(match[2]).trim(), toStringSafe(match[3]).trim(), allowedToolNames);
    if (parsed) {
      output.push(parsed);
    }
  }

  for (const match of source.matchAll(TOOL_SELFCLOSE_PATTERN)) {
    const parsed = parseToolCallInner(toStringSafe(match[1]).trim(), "", allowedToolNames);
    if (parsed) {
      output.push(parsed);
    }
  }

  return output;
}

/**
 * Parse <tool_calls> container format.
 * Now handles all known formats including "tag-name-as-tool-name".
 */
function parseContainerToolBlocks(text, allowedToolNames = []) {
  const output = [];
  const source = toStringSafe(text).trim();

  for (const match of source.matchAll(TOOL_CALLS_CONTAINER_PATTERN)) {
    const containerTag = match[1];
    const attrs = toStringSafe(match[2]).trim();
    const inner = toStringSafe(match[3]).trim();

    log.debug("parser", `[parseContainer] Found <${containerTag}> container, inner length=${inner.length}, inner preview="${inner.slice(0, 300)}"`);

    // Strategy A: nested <tool_call name="..."> blocks inside container
    const innerCalls = parseStandaloneSingularBlocks(inner, allowedToolNames);
    if (innerCalls.length) {
      log.debug("parser", `[parseContainer] Strategy A: found ${innerCalls.length} nested <tool_call/invoke> block(s)`);
      output.push(...innerCalls);
      continue;
    }

    // Strategy B: <tool_name>+<parameters> flat pairs
    const names = findAllTagValues(inner, TOOL_NAME_PATTERNS);
    if (names.length) {
      const namedParams = parseParameterNameAttrs(inner);
      if (namedParams && Object.keys(namedParams).length > 0 && names.length === 1) {
        const toolName = names[0].trim();
        const argumentsText = JSON.stringify(namedParams);
        log.debug("parser", `[parseContainer] Strategy B (named params): name="${toolName}", args="${argumentsText.slice(0, 100)}"`);
        output.push(buildParsedToolCall(toolName, argumentsText));
        continue;
      }

      const args = findAllTagValues(inner, TOOL_ARGS_PATTERNS);
      log.debug("parser", `[parseContainer] Strategy B: ${names.length} <tool_name> tag(s), ${args.length} <parameters> tag(s)`);

      for (let i = 0; i < names.length; i++) {
        const toolName = names[i].trim();
        const toolArgs = i < args.length ? args[i].trim() : "{}";
        if (!toolName) continue;

        let parsedArgs;
        const jsonArgs = parseJsonObject(toolArgs);
        parsedArgs = jsonArgs ? JSON.stringify(jsonArgs) : toolArgs;

        output.push(buildParsedToolCall(toolName, parsedArgs));
      }
      continue;
    }

    // Strategy C: "tag-name-as-tool-name" — <terminal><parameters>...</parameters></terminal>
    const toolNameTags = findToolNameTags(inner, allowedToolNames);
    if (toolNameTags.length) {
      log.debug("parser", `[parseContainer] Strategy C: found ${toolNameTags.length} tool-name tag(s): [${toolNameTags.map(t => t.name).join(", ")}]`);
      for (const { name, body } of toolNameTags) {
        const argumentsText = parseToolCallArguments(body);
        log.debug("parser", `[parseContainer] Strategy C: name="${name}", argumentsText="${argumentsText.slice(0, 100)}"`);
        output.push(buildParsedToolCall(name, argumentsText));
      }
      continue;
    }

    // Strategy D: inner is a single JSON object with .name
    const jsonTool = parseJsonObject(inner);
    if (jsonTool?.name) {
      log.debug("parser", `[parseContainer] Strategy D: JSON tool name=${jsonTool.name}`);
      output.push(buildParsedToolCall(jsonTool.name, JSON.stringify(jsonTool.input ?? jsonTool.arguments ?? {})));
      continue;
    }

    // Strategy E: name from container tag attribute
    const attrName = attrs.match(TOOL_ATTR_PATTERN)?.[2] ?? "";
    if (attrName.trim()) {
      const argumentsText = parseToolCallArguments(inner);
      log.debug("parser", `[parseContainer] Strategy E: attr name="${attrName}", argumentsText="${argumentsText.slice(0, 100)}"`);
      output.push(buildParsedToolCall(attrName.trim(), argumentsText));
      continue;
    }

    log.warn("parser", `[parseContainer] <${containerTag}> container found but no tool calls could be extracted. Inner (first 300): "${inner.slice(0, 300)}"`);
  }

  return output;
}

/**
 * Main parser: Parse tool calls from text.
 * If there are <tool_calls> container blocks, only parse those.
 * Otherwise, parse standalone <tool_call name="..."> blocks.
 */
function parseMarkupToolCalls(text, allowedToolNames = []) {
  const source = toStringSafe(text).trim();

  const hasContainer = TOOL_CALLS_CONTAINER_PATTERN.test(source);
  TOOL_CALLS_CONTAINER_PATTERN.lastIndex = 0;

  if (hasContainer) {
    const containerCalls = parseContainerToolBlocks(source, allowedToolNames);
    log.debug("parser", `[parseMarkupToolCalls] Container mode: found ${containerCalls.length} call(s)`);
    return containerCalls;
  }

  const standaloneCalls = parseStandaloneSingularBlocks(source, allowedToolNames);
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
  const calls = filterAllowedToolCalls(parseMarkupToolCalls(source, allowedToolNames), allowedToolNames);
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

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
  "tool_name", "function_name", "name", "function", "tool",
  "parameters", "parameter", "input", "arguments", "argument", "args", "params",
  "command", "query", "body", "data", "result", "response", "content", "value",
  "description", "type", "key", "url", "path", "file", "text", "message",
  "output", "error", "code", "timeout", "limit"
]);

/* ── Argument-carrying tag names (skip in named-block scanning) ── */
/* Also includes name-indicator tags that shouldn't be treated as wrappers */
const ARGUMENT_CARRYING_TAGS = new Set([
  "parameter", "parameters", "argument", "arguments", "input", "args", "params",
  "tool_name", "function_name"
]);

const TOOL_ATTR_PATTERN = /(name|function|tool)\s*=\s*"([^"]+)"/i;
const TOOL_KV_PATTERN = /<(?:[a-z0-9_:-]+:)?([a-z0-9_.-]+)\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1\s*>/gi;
const ANY_TAG_PATTERN = /<([a-z0-9_:-]+)\b[^>]*>/gi;

/* ── Malformed tag pattern: <tool_name="ReadFile"> should be <tool_name name="ReadFile">
    Also handles <tool_name="ReadFile</tool_name> (missing >) → <tool_name name="ReadFile"></tool_name>
    Uses [^"<>\s]+ to stop at < or > or whitespace, preventing over-matching ── */
const MALFORMED_ATTR_EQUALS = /<((?:[a-z0-9_:-]+:)?(?:tool_name|function_name|name|tool|function|call))="([^"<>\s]+)"?\s*(>)?/gi;

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

/**
 * Strip surrounding quotes from a value when the model wraps it in quotes
 * inside an XML element, e.g. <parameter name="glob_pattern">"*.toml"</parameter>
 * Only strips if the value starts AND ends with the same quote character.
 */
function stripSurroundingQuotes(value) {
  const trimmed = toStringSafe(value).trim();
  if (trimmed.length >= 2) {
    const first = trimmed[0];
    const last = trimmed[trimmed.length - 1];
    if ((first === '"' && last === '"') || (first === "'" && last === "'")) {
      // Only strip if it's not a valid JSON string (already parsed by JSON.parse)
      // and looks like a plain quoted value
      const inner = trimmed.slice(1, -1);
      // Don't strip if inner content itself contains unescaped matching quotes
      // (could be a legitimate value)
      return inner;
    }
  }
  return value;
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

/**
 * Find content of an unclosed argument tag (model omitted closing tag).
 * E.g. <parameters>{"command": "ls"} without </parameters>
 * Looks for the opening tag and extracts everything after it to end of source.
 * Only matches when the tag is truly unclosed (more opening than closing tags).
 * Returns the extracted content string, or "" if not found.
 */
function findUnclosedArgContent(text) {
  const source = toStringSafe(text);
  const tagNames = ["parameters", "input", "arguments", "argument", "args", "params"];

  for (const tagName of tagNames) {
    // Only look for unclosed tags — skip if all tags are properly closed
    const openRe = new RegExp(`<(?:[a-z0-9_:-]+:)?${tagName}\\b[^>]*>`, "gi");
    const closeRe = new RegExp(`</(?:[a-z0-9_:-]+:)?${tagName}\\s*>`, "gi");
    const openCount = (source.match(openRe) || []).length;
    const closeCount = (source.match(closeRe) || []).length;

    if (openCount <= closeCount) continue; // All tags are properly closed

    // This tag has unclosed instances — extract content from opening tag to end
    const extractRe = new RegExp(`<(?:[a-z0-9_:-]+:)?${tagName}\\b[^>]*>([\\s\\S]+)$`, "i");
    const match = source.match(extractRe);
    if (match?.[1]) {
      let content = decodeXmlText(match[1].trim());
      // Strip trailing structural closing tags that are not part of the argument content
      content = content.replace(/<\/(?:tool_calls|function_calls|tool_call|function_call|invoke|tool_use)\s*>$/i, "").trim();
      if (content) {
        log.debug("parser", `[findUnclosedArgContent] Found unclosed <${tagName}> tag, extracted content (length=${content.length}): "${content.slice(0, 150)}"`);
        return content;
      }
    }
  }

  return "";
}

function findTagValue(text, patterns) {
  const source = toStringSafe(text);

  for (const pattern of patterns) {
    const match = source.match(pattern);
    if (match?.[1] !== undefined) {
      const textContent = decodeXmlText(match[1]).trim();
      if (textContent) {
        // If text content looks like nested XML (e.g. <parameters>...</parameters> inside <tool_name>),
        // check name= attribute on the opening tag first — it's more likely the actual value
        if (textContent.startsWith("<")) {
          const fullMatch = match[0];
          const attrVal = fullMatch.match(/name\s*=\s*"([^"]+)"/i)?.[1]?.trim();
          if (attrVal) return attrVal;
        }
        return textContent;
      }
    }
  }

  // If text content is empty, check for name attribute on <tool_name> / <function_name> / <name> tags
  for (const tagName of ["tool_name", "function_name", "name"]) {
    const attrVal = extractNameAttrFromTag(source, tagName);
    if (attrVal) return attrVal;
  }

  // Fallback: try unclosed argument tags (model forgot </parameters>)
  const unclosed = findUnclosedArgContent(source);
  if (unclosed) return unclosed;

  return "";
}

/**
 * Find ALL matches of a tag pattern (not just the first one).
 * Returns array of decoded values.
 * Also checks name= attribute on matched tags when text content is empty or looks like XML.
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
        if (textContent && !textContent.startsWith("<")) {
          // Simple text content — use directly
          results.push(textContent);
        } else {
          // Text content empty or looks like nested XML — try name= attribute on the opening tag
          const fullMatch = match[0];
          const attrVal = fullMatch.match(/name\s*=\s*"([^"]+)"/i)?.[1]?.trim();
          if (attrVal) results.push(attrVal);
          else if (textContent) results.push(textContent); // fallback: use XML content as-is
        }
      }
    }
    if (results.length) break;
  }

  return results;
}

/**
 * Find sub-blocks inside a container that have a name= attribute matching
 * an allowed tool name. Handles formats like:
 *   <tool name="Shell"><parameter name="command">...</parameter></tool>
 *   <tool_name name="TodoWrite"><parameters>...</parameters></tool_name>
 *   <call name="Glob"><arguments>...</arguments></call>
 *
 * Returns { name, body, attrs } objects for each matched block.
 */
function findNamedToolBlocks(text, allowedToolNames) {
  if (!allowedToolNames?.length) return [];

  const allowed = new Set(allowedToolNames.map(n => toStringSafe(n).trim()).filter(Boolean));
  const source = toStringSafe(text);
  const results = [];

  // Match any tag with attributes that include name="ToolName"
  // Backreference \1 ensures opening/closing tag names match
  const anyNamedBlock = /<([a-z0-9_:-]+)\b([^>]*)>([\s\S]*?)<\/\1\s*>/gi;

  for (const match of source.matchAll(anyNamedBlock)) {
    const tagName = match[1].toLowerCase();
    const attrs = match[2];
    const body = match[3];

    // Skip container tags themselves
    if (["tool_calls", "function_calls"].includes(tagName)) continue;

    // Skip argument-carrying tags (parameter, parameters, etc.)
    if (ARGUMENT_CARRYING_TAGS.has(tagName)) continue;

    // Check for name= attribute matching an allowed tool name
    const nameMatch = attrs.match(TOOL_ATTR_PATTERN);
    if (!nameMatch) continue;

    const toolName = nameMatch[2]?.trim();
    if (!toolName || !allowed.has(toolName)) continue;

    results.push({ name: toolName, body: body.trim(), attrs: attrs.trim() });
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
    // Case-insensitive search for closing tag
    const closeTagLower = `</${tagName}`;
    let closePos = -1;
    const sourceLower = source.toLowerCase();
    let searchFrom = openPos + openTag.length;
    while (searchFrom < sourceLower.length) {
      const idx = sourceLower.indexOf(closeTagLower, searchFrom);
      if (idx < 0) break;
      // Verify this is actually a closing tag (followed by > or whitespace)
      const afterTag = sourceLower.slice(idx + closeTagLower.length, idx + closeTagLower.length + 1);
      if (afterTag === ">" || afterTag === " " || afterTag === "\n" || afterTag === "\r" || afterTag === "\t") {
        closePos = idx;
        break;
      }
      searchFrom = idx + 1;
    }

    if (closePos < 0) {
      // No closing tag found — still extract tool name, use rest of content as body
      const body = source.slice(openPos + openTag.length);
      const originalName = allowedToolNames.find(n => toStringSafe(n).trim().toLowerCase() === tagName) || tagName;
      log.debug("parser", `[findToolNameTags] Tag <${match[1]}> has no closing tag, using rest-of-content as body (length=${body.length})`);
      results.push({ name: originalName, body: body.trim() });
      continue;
    }

    // Find the actual end of the closing tag
    const closeEnd = source.indexOf(">", closePos) + 1;
    const body = source.slice(openPos + openTag.length, closePos);

    // Find the original-cased tag name from allowedToolNames
    const originalName = allowedToolNames.find(n => toStringSafe(n).trim().toLowerCase() === tagName) || tagName;

    results.push({ name: originalName, body: body.trim() });
  }

  log.debug("parser", `[findToolNameTags] Found ${results.length} tool-name tag(s): [${results.map(r => r.name).join(", ")}]`);
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
      // Strip surrounding quotes that models sometimes add inside XML values
      output[key] = stripSurroundingQuotes(value);
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
  // Pre-process: close unclosed argument tags (model forgot </parameters>)
  const fixed = closeUnclosedArgTags(body);

  // 1. Try <parameter name="key">value</parameter> format
  const namedParams = parseParameterNameAttrs(fixed);
  if (namedParams && Object.keys(namedParams).length > 0) {
    return JSON.stringify(namedParams);
  }

  // 2. Try <parameters>JSON</parameters> format
  const argsRaw = findTagValue(fixed, TOOL_ARGS_PATTERNS);
  if (argsRaw) {
    const parsedInput = parseMarkupInput(argsRaw);
    if (parsedInput && Object.keys(parsedInput).length > 0) {
      return JSON.stringify(parsedInput);
    }
  }

  // 3. Fallback: parse body as generic markup KV
  const markupObject = parseMarkupObject(fixed);
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

  // 1. Try <tool_call name="..."> / <invoke name="..."> blocks
  for (const match of source.matchAll(TOOL_BLOCK_PATTERN)) {
    const parsed = parseToolCallInner(toStringSafe(match[2]).trim(), toStringSafe(match[3]).trim(), allowedToolNames);
    if (parsed) {
      output.push(parsed);
    }
  }

  // 2. Try self-closing <invoke name="..." /> blocks
  for (const match of source.matchAll(TOOL_SELFCLOSE_PATTERN)) {
    const parsed = parseToolCallInner(toStringSafe(match[1]).trim(), "", allowedToolNames);
    if (parsed) {
      output.push(parsed);
    }
  }

  // 3. Try <tool name="..."> / <tool_name name="..."> blocks (generic named tags)
  //    Only if no standard blocks found, to avoid double-parsing
  if (!output.length) {
    for (const { name, body } of findNamedToolBlocks(source, allowedToolNames)) {
      const argumentsText = parseToolCallArguments(body);
      output.push(buildParsedToolCall(name, argumentsText));
    }
  }

  return output;
}

/**
 * Pre-process: fix common tag closing issues.
 * 1. Fix singular/plural mismatch: <parameters>...</parameter> → <parameters>...</parameters>
 * 2. Close unclosed argument tags: <parameters>... without </parameters>
 * The model sometimes writes </parameter> when it means </parameters>,
 * or omits the closing tag entirely and goes straight to </tool_calls>.
 */
function closeUnclosedArgTags(text) {
  const source = toStringSafe(text);
  let result = source;

  // Step 1: Fix singular/plural closing tag mismatches
  // E.g. <parameters>...</parameter> → <parameters>...</parameters>
  const pluralTags = [
    { open: "parameters", wrongClose: "</parameter>" },
    { open: "arguments", wrongClose: "</argument>" },
    { open: "args", wrongClose: "</arg>" }
  ];

  for (const { open: tagName, wrongClose } of pluralTags) {
    const openRe = new RegExp(`<(?:[a-z0-9_:-]+:)?${tagName}\\b[^>]*>`, "gi");
    const correctCloseRe = new RegExp(`</(?:[a-z0-9_:-]+:)?${tagName}\\s*>`, "gi");
    const wrongCloseRe = new RegExp(wrongClose.replace("/", "\\/") + "\\s*>", "gi");

    const openCount = (result.match(openRe) || []).length;
    const correctCloseCount = (result.match(correctCloseRe) || []).length;
    const wrongCloseCount = (result.match(wrongCloseRe) || []).length;

    // If we have <parameters> opening tags with </parameter> wrong closings
    // but no correct </parameters> closings, fix the mismatches
    if (openCount > 0 && wrongCloseCount > 0 && correctCloseCount < openCount) {
      // Replace wrong closings with correct ones, but only as many as needed
      let fixed = 0;
      const needed = openCount - correctCloseCount;
      result = result.replace(wrongCloseRe, (match) => {
        if (fixed < needed) {
          fixed++;
          return `</${tagName}>`;
        }
        return match;
      });
      if (fixed > 0) {
        log.debug("parser", `[closeUnclosedArgTags] Fixed ${fixed} </${wrongClose.slice(2)}> → </${tagName}> mismatch(es)`);
      }
    }
  }

  // Step 2: Close any remaining unclosed tags
  const tagNames = ["parameters", "input", "arguments", "argument", "args", "params"];

  for (const tagName of tagNames) {
    const openRe = new RegExp(`<(?:[a-z0-9_:-]+:)?${tagName}\\b[^>]*>`, "gi");
    const closeRe = new RegExp(`</(?:[a-z0-9_:-]+:)?${tagName}\\s*>`, "gi");

    const openCount = (result.match(openRe) || []).length;
    const closeCount = (result.match(closeRe) || []).length;
    const missing = openCount - closeCount;

    if (missing > 0) {
      log.debug("parser", `[closeUnclosedArgTags] Adding ${missing} missing </${tagName}> closing tag(s)`);
      // Append missing closing tags at the end of the content
      for (let i = 0; i < missing; i++) {
        result += `</${tagName}>`;
      }
    }
  }

  return result;
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

    // Pre-process: flatten nested containers (e.g. <tool_calls><tool_calls>...)</tool_calls></tool_calls>)
    let flatInner = inner;
    const nestedContainer = /^(?:[a-z0-9_:-]+:)?(tool_calls|function_calls)\b[^>]*>([\s\S]*?)<\/\1\s*>$/i.exec(inner.trim());
    if (nestedContainer) {
      flatInner = nestedContainer[2].trim();
      log.debug("parser", `[parseContainer] Flattened nested <${nestedContainer[1]}> container, new inner length=${flatInner.length}`);
    }

    // Pre-process: close unclosed argument tags (model forgot </parameters>)
    flatInner = closeUnclosedArgTags(flatInner);

    // Strategy A: nested <tool_call name="..."> blocks inside container
    const innerCalls = parseStandaloneSingularBlocks(flatInner, allowedToolNames);
    if (innerCalls.length) {
      log.debug("parser", `[parseContainer] Strategy A: found ${innerCalls.length} nested <tool_call/invoke> block(s)`);
      output.push(...innerCalls);
      continue;
    }

    // Strategy A2: <tool name="Shell"> or <tool_name name="TodoWrite"> blocks
    // Any tag with name= attribute matching an allowed tool name
    const namedBlocks = findNamedToolBlocks(flatInner, allowedToolNames);
    if (namedBlocks.length) {
      log.debug("parser", `[parseContainer] Strategy A2: found ${namedBlocks.length} named block(s): [${namedBlocks.map(b => b.name).join(", ")}]`);
      for (const { name, body } of namedBlocks) {
        const argumentsText = parseToolCallArguments(body);
        log.debug("parser", `[parseContainer] Strategy A2: name="${name}", argumentsText="${argumentsText.slice(0, 100)}"`);
        output.push(buildParsedToolCall(name, argumentsText));
      }
      continue;
    }

    // Strategy B: <tool_name>+<parameters> flat pairs
    const names = findAllTagValues(flatInner, TOOL_NAME_PATTERNS);
    if (names.length) {
      const namedParams = parseParameterNameAttrs(flatInner);
      if (namedParams && Object.keys(namedParams).length > 0 && names.length === 1) {
        const toolName = names[0].trim();
        const argumentsText = JSON.stringify(namedParams);
        log.debug("parser", `[parseContainer] Strategy B (named params): name="${toolName}", args="${argumentsText.slice(0, 100)}"`);
        output.push(buildParsedToolCall(toolName, argumentsText));
        continue;
      }

      const args = findAllTagValues(flatInner, TOOL_ARGS_PATTERNS);
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

    // Strategy C: "tag-name-as-tool-name" — <Glob><parameters>...</parameters></Glob>
    const toolNameTags = findToolNameTags(flatInner, allowedToolNames);
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
    const jsonTool = parseJsonObject(flatInner);
    if (jsonTool?.name) {
      log.debug("parser", `[parseContainer] Strategy D: JSON tool name=${jsonTool.name}`);
      output.push(buildParsedToolCall(jsonTool.name, JSON.stringify(jsonTool.input ?? jsonTool.arguments ?? {})));
      continue;
    }

    // Strategy E: name from container tag attribute
    const attrName = attrs.match(TOOL_ATTR_PATTERN)?.[2] ?? "";
    if (attrName.trim()) {
      const argumentsText = parseToolCallArguments(flatInner);
      log.debug("parser", `[parseContainer] Strategy E: attr name="${attrName}", argumentsText="${argumentsText.slice(0, 100)}"`);
      output.push(buildParsedToolCall(attrName.trim(), argumentsText));
      continue;
    }

    log.warn("parser", `[parseContainer] <${containerTag}> container found but no tool calls could be extracted. FlatInner (first 300): "${flatInner.slice(0, 300)}"`);
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

  // Step 1.5: Fix malformed tags like <tool_name="ReadFile"> → <tool_name name="ReadFile">
  // Also fixes missing >: <tool_name="ReadFile"</tool_name> → <tool_name name="ReadFile"></tool_name>
  const fixed = source.replace(MALFORMED_ATTR_EQUALS, '<$1 name="$2">');
  if (fixed !== source) {
    log.debug("parser", `Preprocessed malformed tags (e.g. <tool_name="..."> → <tool_name name="...">)`);
  }

  // Step 2: Quick check for any tool-related XML tags
  // Also matches <tool name="...">, <tool_name name="...">, <function name="...">
  const stripped = stripFencedCodeBlocks(fixed);
  if (!stripped.match(/<(?:tool_calls|tool_call|tool_name|tool|function_calls|function_call|function_name|function|invoke|tool_use)\b/i)) {
    log.debug("parser", `No tool XML tags found in output (length=${fixed.length})`);
    return [];
  }

  log.debug("parser", `Tool XML tags detected. Full source length=${fixed.length}, first 500 chars: "${fixed.slice(0, 500)}"`);

  // Step 3: Parse
  const calls = filterAllowedToolCalls(parseMarkupToolCalls(fixed, allowedToolNames), allowedToolNames);
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

// Export for testing
export { MALFORMED_ATTR_EQUALS };

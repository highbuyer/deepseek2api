import { randomUUID } from "node:crypto";
import { log } from "../utils/log.js";

/* ── Single tool-call block patterns ── */
/* Includes <tool> to handle models that output <tool name="Shell">...</tool> */
/* Two patterns needed because DeepSeek sometimes closes <tool_call ...> with </tool_call_name>
 * (backreference \1 can't express "tool_call" OR "tool_call_name" as valid close) */
const TOOL_BLOCK_PATTERNS = Object.freeze([
  // <tool_call ...>...</tool_call_name> (most common: name-indicator close)
  /<(?:[a-z0-9_:-]+:)?(tool_call|function_call|invoke|tool)\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1_name\s*>/gi,
  // <tool_call ...>...</tool_call_name> (same tag match, backreference)
  /<(?:[a-z0-9_:-]+:)?(tool_call|function_call|invoke|tool)\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1\s*>/gi,
]);
const TOOL_SELFCLOSE_PATTERN = /<(?:[a-z0-9_:-]+:)?invoke\b([^>]*)\/>/gi;

/* ── Container patterns: <tool_calls>...</tool_calls> (plural) ── */
const TOOL_CALLS_CONTAINER_PATTERN = /<(?:[a-z0-9_:-]+:)?(tool_calls|function_calls)\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1\s*>/gi;

/* ── Sub-element patterns ── */
/* Fuzzy match both opening AND closing tags for tool_name / function_name.
 *   Handles garbled tags like:
 *     <tool_cool_name>ReadFile</tool_name>       (garbled open only)
 *     <tool_cool_name>ReadFile</tool_call_name>  (both garbled)
 *     <tool_cool_name>ReadFile</tool_coll>       (garbled close, no "name")
 *   Also handles NO closing tag:
 *     <tool_cool_name>ReadFile\n<parameters>...
 * Order: most specific → least specific */
const TOOL_NAME_PATTERNS = Object.freeze([
  // garbled open + correct close (e.g. <tool_cool_name>ReadFile</tool_name>)
  /<(?:[a-z0-9_:-]+:)?tool[a-z0-9_]*name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?tool_name>/i,
  // garbled open + garbled close with "name" (e.g. <tool_cool_name>ReadFile</tool_call_name>)
  /<(?:[a-z0-9_:-]+:)?tool[a-z0-9_]*name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?tool[a-z0-9_]*name>/i,
  // garbled open + no close — capture text until next < (e.g. <tool_cool_name>ReadFile\n<parameters>)
  /<(?:[a-z0-9_:-]+:)?tool[a-z0-9_]*name\b[^>]*>([^<]+)/i,
  // same three for function_name
  /<(?:[a-z0-9_:-]+:)?function[a-z0-9_]*name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?function_name>/i,
  /<(?:[a-z0-9_:-]+:)?function[a-z0-9_]*name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?function[a-z0-9_]*name>/i,
  /<(?:[a-z0-9_:-]+:)?function[a-z0-9_]*name\b[^>]*>([^<]+)/i,
  // standard <name> and <function>
  /<(?:[a-z0-9_:-]+:)?name\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?name>/i,
  /<(?:[a-z0-9_:-]+:)?function\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?function>/i
]);

/* ── Args patterns: <parameters>, <arguments>, <input> etc. (JSON body) ── */
/* Also handles garbled variants like <tool_call_parameters> ── */
const TOOL_ARGS_PATTERNS = Object.freeze([
  /<(?:[a-z0-9_:-]+:)?tool[a-z0-9_]*parameters\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?tool[a-z0-9_]*parameters>/i,
  /<(?:[a-z0-9_:-]+:)?function[a-z0-9_]*parameters\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?function[a-z0-9_]*parameters>/i,
  /<(?:[a-z0-9_:-]+:)?parameters\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?parameters>/i,
  /<(?:[a-z0-9_:-]+:)?input\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?input>/i,
  /<(?:[a-z0-9_:-]+:)?arguments\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?arguments>/i,
  /<(?:[a-z0-9_:-]+:)?argument\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?argument>/i,
  /<(?:[a-z0-9_:-]+:)?args\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?args>/i,
  /<(?:[a-z0-9_:-]+:)?params\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?params>/i
]);

/* ── <parameter name="key">value</parameter> pattern (singular with name attr) ── */
/* Also matches <param name="key">value</param> (shorthand used by some models) */
const PARAMETER_NAME_ATTR_PATTERN = /<(?:[a-z0-9_:-]+:)?param(?:eter)?\b[^>]*\bname\s*=\s*"([^"]+)"[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?param(?:eter)?\s*>/gi;

/* ── Known structural tag names (not tool names) ── */
/* NOTE: "tool" is NOT here because <tool name="Shell"> is a valid tool call wrapper.
   It's handled directly by TOOL_BLOCK_PATTERNS and findNamedToolBlocks instead. */
const KNOWN_STRUCTURAL_TAGS = new Set([
  "tool_calls", "tool_call", "function_calls", "function_call", "invoke", "tool_use",
  "tool_name", "function_name", "name", "function",
  "tool_call_name", "tool_call_parameters", "function_call_name", "function_call_parameters",
  "parameters", "parameter", "input", "arguments", "argument", "args", "params",
  "command", "query", "body", "data", "result", "response", "content", "value",
  "description", "type", "key", "url", "path", "file", "text", "message",
  "output", "error", "code", "timeout", "limit"
]);

/* ── Argument-carrying tag names (skip in named-block scanning) ── */
/* Also includes name-indicator tags that shouldn't be treated as wrappers */
/* NOTE: "tool" is NOT here because <tool name="Shell"> is a valid tool call wrapper */
const ARGUMENT_CARRYING_TAGS = new Set([
  "parameter", "parameters", "argument", "arguments", "input", "args", "params",
  "tool_name", "function_name",
  "tool_call_parameters", "function_call_parameters"
]);

const TOOL_ATTR_PATTERN = /(name|function|tool)\s*=\s*"([^"]+)"/i;
const TOOL_KV_PATTERN = /<(?:[a-z0-9_:-]+:)?([a-z0-9_.-]+)\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?\1\s*>/gi;
const ANY_TAG_PATTERN = /<([a-z0-9_:-]+)\b[^>]*>/gi;

/* ── Malformed tag patterns ── */
/* Pattern 1: <tool_name="ReadFile"> should be <tool_name name="ReadFile">
   Also handles <tool_name="ReadFile</tool_name> (missing >) → <tool_name name="ReadFile"></tool_name>
   Uses [^"<>\s]+ to stop at < or > or whitespace, preventing over-matching */
const MALFORMED_ATTR_EQUALS = /<((?:[a-z0-9_:-]+:)?(?:tool_name|function_name|name|tool|function|call))="([^"<>\s]+)"?\s*(>)?/gi;

/* Pattern 2: <tool_name name="WebSearch</tool_name> — name attribute value
   extends to include the closing tag (no closing quote before </).
   Extract the tool name and rewrite as <tool_name>WebSearch</tool_name> */
const MALFORMED_NAME_ATTR_UNCLOSED = /<((?:[a-z0-9_:-]+:)?(?:tool_name|function_name))\s+name="([a-zA-Z0-9_]+)<\/\1>/gi;

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

/**
 * Fix literal control characters inside JSON string values.
 * Models sometimes output CDATA JSON with literal newlines/tabs/carriage returns
 * inside string values (e.g. {"patch":"*** Begin Patch\n*** Modify..."}),
 * which is invalid JSON. This function escapes those control characters
 * ONLY inside JSON string values, preserving structural whitespace outside strings.
 */
function fixJsonControlChars(text) {
  let result = "";
  let inString = false;
  let escape = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];

    if (escape) {
      result += ch;
      escape = false;
      continue;
    }

    if (ch === "\\" && inString) {
      result += ch;
      escape = true;
      continue;
    }

    if (ch === '"') {
      inString = !inString;
      result += ch;
      continue;
    }

    if (inString) {
      if (ch === "\n") { result += "\\n"; continue; }
      if (ch === "\r") { result += "\\r"; continue; }
      if (ch === "\t") { result += "\\t"; continue; }
    }

    result += ch;
  }

  return result;
}

function parseJsonObject(text) {
  try {
    const value = JSON.parse(text);
    return value && typeof value === "object" && !Array.isArray(value) ? value : null;
  } catch {
    // Retry: fix literal control characters inside JSON string values
    // (model sometimes outputs CDATA JSON with literal newlines inside strings)
    try {
      const fixed = fixJsonControlChars(text);
      const value = JSON.parse(fixed);
      return value && typeof value === "object" && !Array.isArray(value) ? value : null;
    } catch {
      return null;
    }
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
  return findAllTagValuesImpl(text, patterns, /* skipNameAttr */ false);
}

/**
 * Find ALL matches of an argument tag pattern (e.g. <parameters>).
 * Returns array of raw decoded content strings — does NOT apply the name= attribute
 * fallback that findAllTagValues uses for tool-name extraction.
 * This prevents <parameter name="path"> sub-elements from being mis-extracted as the args value.
 */
function findAllArgsValues(text, patterns) {
  return findAllTagValuesImpl(text, patterns, /* skipNameAttr */ true);
}

function findAllTagValuesImpl(text, patterns, skipNameAttr) {
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
        } else if (!skipNameAttr) {
          // Text content empty or looks like nested XML — try name= attribute on the opening tag
          // (only for tool-name extraction, NOT for args extraction)
          const fullMatch = match[0];
          const attrVal = fullMatch.match(/name\s*=\s*"([^"]+)"/i)?.[1]?.trim();
          if (attrVal) results.push(attrVal);
          else if (textContent) results.push(textContent); // fallback: use XML content as-is
        } else {
          // Args extraction: always return raw content, never the name= attribute
          if (textContent) results.push(textContent);
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
 * 3. Generic markup KV (excluding tool_name/function_name/parameters tags)
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
  //    Strip <tool_name>, <function_name>, <name>, <parameters> tags first
  //    to avoid treating them as argument keys
  const cleanedForKV = fixed
    .replace(/<(?:[a-z0-9_:-]+:)?(?:tool_name|function_name|tool_call_name|function_call_name)\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?(?:tool_name|function_name|tool_call_name|function_call_name)>/gi, '')
    .replace(/<(?:[a-z0-9_:-]+:)?(?:tool_call_parameters|function_call_parameters|parameters)\b[^>]*>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?(?:tool_call_parameters|function_call_parameters|parameters)>/gi, '')
    .trim();
  const markupObject = parseMarkupObject(cleanedForKV);
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
  //    If the first match isn't in allowedToolNames, try subsequent matches
  //    (model sometimes outputs stale/ghost <tool_name>Write</tool_name> before the real one)
  const attrName = attrs.match(TOOL_ATTR_PATTERN)?.[2] ?? "";
  let name = "";
  if (attrName.trim()) {
    name = attrName.trim();
  } else {
    const allNames = findAllTagValues(inner, TOOL_NAME_PATTERNS);
    const allowedSet = new Set(allowedToolNames.map(n => toStringSafe(n).trim()).filter(Boolean));
    if (allowedSet.size > 0) {
      // Prefer the first name that's in the allowed list
      name = allNames.find(n => allowedSet.has(n.trim()))?.trim() ?? allNames[0]?.trim() ?? "";
    } else {
      name = allNames[0]?.trim() ?? "";
    }
  }
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

  // 4. If no name found, try plain-text "Key: Value" format (e.g. "Tool: Shell")
  if (!name && allowedToolNames.length) {
    const kvResults = parseKeyValueToolFormat(inner, allowedToolNames);
    if (kvResults.length) {
      const first = kvResults[0];
      name = first.name;
      const argumentsText = first.argumentsText;
      log.debug("parser", `[parseToolCallInner] Found tool name via KV format: name="${name}", argumentsText="${argumentsText.slice(0, 100)}"`);
      return buildParsedToolCall(name, argumentsText);
    }
  }

  if (!name) {
    log.debug("parser", `[parseToolCallInner] No tool name found (tried attr, tool_name tags, tag-name matching, and KV format), returning null`);
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
  //    Try both _name close and exact-match close patterns
  for (const pattern of TOOL_BLOCK_PATTERNS) {
    for (const match of source.matchAll(pattern)) {
      const parsed = parseToolCallInner(toStringSafe(match[2]).trim(), toStringSafe(match[3]).trim(), allowedToolNames);
      if (parsed) {
        output.push(parsed);
      }
    }
    if (output.length) break; // Found matches, no need to try next pattern
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

  // 4. Try "tag-name-as-tool-name" format (e.g. <ApplyPatch>...</ApplyPatch>)
  //    This handles models that output the tool name directly as an XML tag
  //    without wrapping it in <tool_call> or <tool_calls>.
  if (!output.length && allowedToolNames.length) {
    const toolNameTags = findToolNameTags(source, allowedToolNames);
    if (toolNameTags.length) {
      log.debug("parser", `[parseStandaloneSingularBlocks] Strategy 4: found ${toolNameTags.length} tool-name tag(s): [${toolNameTags.map(t => t.name).join(", ")}]`);
      for (const { name, body } of toolNameTags) {
        const argumentsText = parseToolCallArguments(body);
        output.push(buildParsedToolCall(name, argumentsText));
      }
    }
  }

  return output;
}

/**
 * Pre-process: fix common tag closing issues.
 * 1. Strip garbage characters after CDATA closing (e.g. ]]></parameter>}}})
 * 2. Fix singular/plural mismatch: <parameters>...</parameter> → <parameters>...</parameters>
 * 3. Close unclosed argument tags: <parameters>... without </parameters>
 * The model sometimes writes </parameter> when it means </parameters>,
 * or omits the closing tag entirely and goes straight to </tool_calls>.
 * It also sometimes outputs garbage like ]]>}}}</parameter> after CDATA sections.
 */
function closeUnclosedArgTags(text) {
  const source = toStringSafe(text);
  let result = source;

  // Step 0: Strip garbage characters between CDATA closing ]]> and closing tags
  // The model sometimes outputs: ]]>}}}</parameter> or ]]>}}}</parameters>
  // This garbage prevents proper tag matching. Strip it before processing.
  // Pattern: ]]> followed by } characters, then </tag>
  const beforeStep0 = result;
  result = result.replace(/\]\]>(\}+)(<\/(?:[a-z0-9_:-]+:)?(?:parameters?|arguments?|args?|params|input))/gi, ']]>$2');
  // Also handle garbage AFTER closing tags: </parameter>}}} or </parameters>}}}
  result = result.replace(/(<\/(?:[a-z0-9_:-]+:)?(?:parameters?|arguments?|args?|params|input)\s*>)(\}+)/gi, '$1');
  if (result !== beforeStep0) {
    log.debug("parser", `[closeUnclosedArgTags] Step 0: Stripped garbage after CDATA/tags, length ${beforeStep0.length} → ${result.length}`);
  }

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
    // wrongClose is like "</parameter>" — strip the trailing > before adding \s*>
    // otherwise we'd get <\/parameter>\s*> which requires TWO > characters
    const wrongCloseBody = wrongClose.slice(0, -1); // e.g. "</parameter" (no trailing >)
    const wrongCloseRe = new RegExp(wrongCloseBody.replace("/", "\\/") + "\\s*>", "gi");

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
  const tagNames = ["parameters", "input", "arguments", "argument", "args", "params",
    "tool_call_parameters", "function_call_parameters"];

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
 * Infer tool name from parameter names when the model omits the tool name.
 * Uses a simple heuristic: certain parameter names are strong indicators
 * of specific tools (e.g. "command" → Shell, "path" → ReadFile).
 * Only returns a name if it's in the allowed list.
 */
function inferToolNameFromParams(params, allowedToolNames) {
  if (!params || !allowedToolNames?.length) return null;
  const allowed = new Set(allowedToolNames.map(n => toStringSafe(n).trim()).filter(Boolean));
  const paramNames = new Set(Object.keys(params).map(k => k.toLowerCase()));

  // Parameter name → candidate tool names (ordered by specificity)
  const HINTS = [
    { param: "command", tools: ["Shell", "Bash"] },
    { param: "patch", tools: ["ApplyPatch"] },
    { param: "pattern", tools: ["Glob"] },
    { param: "path", tools: ["ReadFile", "Read"] },
    { param: "query", tools: ["WebSearch"] },
    { param: "url", tools: ["WebFetch"] },
    { param: "glob_pattern", tools: ["Glob"] },
    { param: "glob", tools: ["Glob"] },
  ];

  for (const { param, tools } of HINTS) {
    if (paramNames.has(param)) {
      for (const toolName of tools) {
        if (allowed.has(toolName)) {
          log.debug("parser", `[inferToolNameFromParams] Param "${param}" → tool "${toolName}"`);
          return toolName;
        }
      }
    }
  }

  return null;
}

/**
 * Fix mismatched wrapper closing tags.
 * DeepSeek sometimes closes <tool_call ...> with </tool_calls> instead of </tool_call_name>.
 * E.g. <tool_call name="Shell">...</tool_calls> → <tool_call name="Shell">...</tool_call_name>
 *
 * Also handles the inverse: <tool_calls>...</tool_call_name> (rare but possible).
 *
 * Only fixes when the closing tag is clearly wrong — i.e. the open tag has name= attr
 * (proving it's a tool_call, not a container) but the close uses the plural form.
 */
function fixMismatchedWrapperClosings(text) {
  let result = toStringSafe(text);

  // Fix: <tool_call name="...">...</tool_calls> → </tool_call_name>
  // Match <tool_call ...>content</tool_calls> where content does NOT contain </tool_call_name>
  result = result.replace(
    /<(?:[a-z0-9_:-]+:)?tool_call\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?tool_calls\s*>/gi,
    (full, attrs, content) => {
      if (content.includes("</tool_call")) return full; // Already has proper close inside
      log.debug("parser", `[fixMismatchedWrapperClosings] Fixed </tool_calls> → </tool_call_name>`);
      return `<tool_call${attrs}>${content}</tool_call_name>`;
    }
  );

  // Same for function_call / function_calls
  result = result.replace(
    /<(?:[a-z0-9_:-]+:)?function_call\b([^>]*)>([\s\S]*?)<\/(?:[a-z0-9_:-]+:)?function_calls\s*>/gi,
    (full, attrs, content) => {
      if (content.includes("</function_call")) return full;
      log.debug("parser", `[fixMismatchedWrapperClosings] Fixed </function_calls> → </function_call_name>`);
      return `<function_call${attrs}>${content}</function_call_name>`;
    }
  );

  return result;
}

/**
 * Close unclosed tool wrapper tags like <tool>, <tool_call, <function_call, <invoke>.
 * When the model omits the closing tag (stream truncation, etc.), this appends
 * synthetic closing tags so that TOOL_BLOCK_PATTERNS and findNamedToolBlocks can match.
 * Only closes tags that are genuinely unclosed (more opens than closes).
 */
function closeUnclosedToolWrapperTags(text) {
  const source = toStringSafe(text);
  let result = source;

  const wrapperTags = ["tool", "tool_call", "function_call", "invoke", "tool_use"];

  for (const tagName of wrapperTags) {
    const openRe = new RegExp(`<(?:[a-z0-9_:-]+:)?${tagName}\\b[^>]*>`, "gi");
    const closeRe = new RegExp(`</(?:[a-z0-9_:-]+:)?${tagName}\\s*>`, "gi");

    const openCount = (result.match(openRe) || []).length;
    const closeCount = (result.match(closeRe) || []).length;
    const missing = openCount - closeCount;

    if (missing > 0) {
      log.debug("parser", `[closeUnclosedToolWrapperTags] Adding ${missing} missing </${tagName}> closing tag(s)`);
      for (let i = 0; i < missing; i++) {
        result += `</${tagName}>`;
      }
    }
  }

  return result;
}

/**
 * Parse plain-text "Key: Value" tool call format.
 * Handles models that output non-XML text inside <tool_calls> containers:
 *   Tool: Shell
 *   command: ls -la
 *
 * Also handles bare tool names:
 *   Shell
 *
 * And multiple tool calls separated by blank lines:
 *   Tool: Shell
 *   command: ls -la
 *
 *   Tool: Glob
 *   pattern: *.toml
 *
 * Returns array of { name, argumentsText } objects.
 */
function parseKeyValueToolFormat(text, allowedToolNames = []) {
  if (!allowedToolNames?.length) return [];

  const allowed = new Set(allowedToolNames.map(n => toStringSafe(n).trim()).filter(Boolean));
  const source = toStringSafe(text).trim();

  // Skip if content looks like XML (starts with <) — not a KV format
  if (source.startsWith("<")) return [];

  const results = [];

  // Recognized keys that indicate a tool name
  const TOOL_NAME_KEYS = new Set(["tool", "name", "function", "action", "tool_name", "function_name", "call"]);

  // Split into blocks separated by blank lines (each block = one tool call)
  const blocks = source.split(/\n\s*\n/).filter(b => b.trim());

  for (const block of blocks) {
    const lines = block.split("\n").map(l => l.trim()).filter(Boolean);
    if (!lines.length) continue;

    let toolName = "";
    const args = {};

    for (const line of lines) {
      const colonIdx = line.indexOf(":");
      if (colonIdx < 0) {
        // No colon — could be a bare tool name
        const candidate = line.trim();
        if (allowed.has(candidate) && !toolName) {
          toolName = candidate;
        }
        continue;
      }

      const key = line.slice(0, colonIdx).trim().toLowerCase();
      const value = line.slice(colonIdx + 1).trim();

      if (TOOL_NAME_KEYS.has(key)) {
        // This line specifies the tool name
        const candidate = value.trim();
        if (candidate && !toolName) {
          toolName = candidate;
        }
      } else if (key) {
        // This is an argument line
        args[key] = value;
      }
    }

    // If no tool name found from "Tool: Name" lines, check if first line is a bare tool name
    if (!toolName && lines.length > 0) {
      const firstLine = lines[0].trim();
      // Check if first line is just a tool name (no colon, or the part before colon is not a known key)
      if (allowed.has(firstLine)) {
        toolName = firstLine;
      } else {
        // Try the value part of the first line if it looks like "Key: ToolName"
        const colonIdx = firstLine.indexOf(":");
        if (colonIdx >= 0) {
          const possibleName = firstLine.slice(colonIdx + 1).trim();
          if (allowed.has(possibleName)) {
            toolName = possibleName;
          }
        }
      }
    }

    if (toolName && allowed.has(toolName)) {
      const argumentsText = Object.keys(args).length > 0 ? JSON.stringify(args) : "{}";
      log.debug("parser", `[parseKeyValueToolFormat] Found tool: name="${toolName}", argumentsText="${argumentsText.slice(0, 100)}"`);
      results.push({ name: toolName, argumentsText });
    } else if (toolName) {
      log.debug("parser", `[parseKeyValueToolFormat] Found tool name "${toolName}" but not in allowed list, skipping`);
    }
  }

  // Fallback: if no blocks parsed but the entire content is a single allowed tool name
  if (!results.length && allowed.has(source)) {
    log.debug("parser", `[parseKeyValueToolFormat] Entire content is a bare tool name: "${source}"`);
    results.push({ name: source, argumentsText: "{}" });
  }

  return results;
}

/**
 * Strip repeated/duplicate container opening tags from content.
 * When the model glitches, it may output dozens or hundreds of repeated
 * <tool_calls> tags: <tool_calls>\n<tool_calls>\n<tool_calls>...
 * This strips all leading duplicate container open tags, keeping only
 * the actual content after them. Also strips trailing duplicate close tags.
 *
 * Returns the cleaned content.
 */
function stripDuplicateContainerTags(text) {
  let result = toStringSafe(text).trim();
  const originalLength = result.length;

  // Strip leading repeated container open tags (with optional whitespace/newlines between)
  // E.g. "<tool_calls>\n<tool_calls>\n<tool_calls>\n<tool_name>..." → "<tool_name>..."
  const containerOpenRe = /^<(?:[a-z0-9_:-]+:)?(?:tool_calls|function_calls)\b[^>]*>\s*/i;
  let stripped = 0;
  while (containerOpenRe.test(result)) {
    const next = result.replace(containerOpenRe, "");
    if (next === result) break; // No change, stop
    // Check that what remains is NOT just more container tags or empty
    const remaining = next.trim();
    if (!remaining) break; // Don't strip if nothing left
    // Check if remaining is just closing tags (would leave nothing useful)
    if (/^<\/(?:[a-z0-9_:-]+:)?(?:tool_calls|function_calls)\s*>\s*$/i.test(remaining)) break;
    result = next;
    stripped++;
    if (stripped > 50) break; // Safety limit
  }

  // Strip trailing repeated container close tags
  const containerCloseRe = /\s*<\/(?:[a-z0-9_:-]+:)?(?:tool_calls|function_calls)\s*>$/i;
  let closeStripped = 0;
  while (containerCloseRe.test(result)) {
    const next = result.replace(containerCloseRe, "");
    if (next === result) break;
    const remaining = next.trim();
    if (!remaining) break;
    result = next;
    closeStripped++;
    if (closeStripped > 50) break;
  }

  if (stripped > 1 || closeStripped > 1) {
    log.debug("parser", `[stripDuplicateContainerTags] Stripped ${stripped} duplicate open tag(s) and ${closeStripped} duplicate close tag(s), length ${originalLength} → ${result.length}`);
  }

  return result.trim();
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

    // Pre-process: strip duplicate container tags (model glitch: <tool_calls> repeated 200+ times)
    flatInner = stripDuplicateContainerTags(flatInner);

    // Pre-process: fix mismatched wrapper closings (e.g. <tool_call ...>...</tool_calls>)
    flatInner = fixMismatchedWrapperClosings(flatInner);

    // Pre-process: close unclosed argument tags (model forgot </parameters>)
    flatInner = closeUnclosedArgTags(flatInner);

    // Pre-process: close unclosed tool wrapper tags (model forgot </tool> or </tool_call)
    // e.g. <tool name="Shell"><parameters>...</parameters>  ← no </tool>
    flatInner = closeUnclosedToolWrapperTags(flatInner);

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
      // Filter ghost tool names: prefer ones in allowed list
      const allowedSet = new Set(allowedToolNames.map(n => toStringSafe(n).trim()).filter(Boolean));
      let effectiveNames = names;
      if (allowedSet.size > 0 && names.length > 1) {
        const allowedNames = names.filter(n => allowedSet.has(n.trim()));
        if (allowedNames.length > 0) effectiveNames = allowedNames;
      }

      const namedParams = parseParameterNameAttrs(flatInner);
      if (namedParams && Object.keys(namedParams).length > 0 && effectiveNames.length === 1) {
        const toolName = effectiveNames[0].trim();
        const argumentsText = JSON.stringify(namedParams);
        log.debug("parser", `[parseContainer] Strategy B (named params): name="${toolName}", args="${argumentsText.slice(0, 100)}"`);
        output.push(buildParsedToolCall(toolName, argumentsText));
        continue;
      }

      const args = findAllArgsValues(flatInner, TOOL_ARGS_PATTERNS);
      log.debug("parser", `[parseContainer] Strategy B: ${effectiveNames.length} <tool_name> tag(s) (from ${names.length} total), ${args.length} <parameters> tag(s)`);

      for (let i = 0; i < effectiveNames.length; i++) {
        const toolName = effectiveNames[i].trim();
        const toolArgs = i < args.length ? args[i].trim() : "{}";
        if (!toolName) continue;

        let parsedArgs;
        // First try <parameter name="key">value</parameter> format (handles XML sub-elements)
        const namedParams = parseParameterNameAttrs(toolArgs);
        if (namedParams && Object.keys(namedParams).length > 0) {
          parsedArgs = JSON.stringify(namedParams);
        } else {
          const jsonArgs = parseJsonObject(toolArgs);
          if (jsonArgs) {
            parsedArgs = JSON.stringify(jsonArgs);
          } else {
            // Not JSON — try parsing as XML markup KV (e.g. <pattern>...</pattern><path>...</path>)
            const markupArgs = parseMarkupInput(toolArgs);
            if (markupArgs && Object.keys(markupArgs).length > 0) {
              parsedArgs = JSON.stringify(markupArgs);
            } else {
              parsedArgs = toolArgs;
            }
          }
        }

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

    // Strategy F: plain-text "Key: Value" format (e.g. "Tool: Shell")
    // Handles models that output non-XML text inside <tool_calls> containers:
    //   <tool_calls>
    //   Tool: Shell
    //   </tool_calls>
    // Or with arguments:
    //   <tool_calls>
    //   Tool: Shell
    //   command: ls -la
    //   </tool_calls>
    // Or just a bare tool name:
    //   <tool_calls>
    //   Shell
    //   </tool_calls>
    const kvToolCalls = parseKeyValueToolFormat(flatInner, allowedToolNames);
    if (kvToolCalls.length) {
      log.debug("parser", `[parseContainer] Strategy F: found ${kvToolCalls.length} key-value tool call(s): [${kvToolCalls.map(c => c.name).join(", ")}]`);
      for (const kv of kvToolCalls) {
        output.push(buildParsedToolCall(kv.name, kv.argumentsText));
      }
      continue;
    }

    // Strategy H: Infer tool name from parameter names
    // When the model forgets the tool name entirely but still outputs <parameter name="command">,
    // we can guess: command→Shell, path→ReadFile, pattern→Glob, etc.
    const namedParams = parseParameterNameAttrs(flatInner);
    if (namedParams && Object.keys(namedParams).length > 0) {
      const inferredToolName = inferToolNameFromParams(namedParams, allowedToolNames);
      if (inferredToolName) {
        const argumentsText = JSON.stringify(namedParams);
        log.debug("parser", `[parseContainer] Strategy H: inferred tool name="${inferredToolName}" from params [${Object.keys(namedParams).join(",")}], argumentsText="${argumentsText.slice(0, 100)}"`);
        output.push(buildParsedToolCall(inferredToolName, argumentsText));
        continue;
      }
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

  // Fallback: check for unclosed container tags (model glitch — repeated <tool_calls> without closing)
  // If there are <tool_calls> opening tags but no closing tags, strip the duplicates and try parsing
  const containerOpenRe = /<(?:[a-z0-9_:-]+:)?(?:tool_calls|function_calls)\b[^>]*>/gi;
  const containerCloseRe = /<\/(?:[a-z0-9_:-]+:)?(?:tool_calls|function_calls)\s*>/gi;
  const openCount = (source.match(containerOpenRe) || []).length;
  const closeCount = (source.match(containerCloseRe) || []).length;

  if (openCount > 0 && closeCount === 0) {
    // Has opening container tags but no closing tags — strip duplicates and try to parse inner content
    const stripped = stripDuplicateContainerTags(source);
    if (stripped.length > 0 && stripped.length < source.length) {
      log.debug("parser", `[parseMarkupToolCalls] Detected ${openCount} unclosed <tool_calls> tag(s), stripped to ${stripped.length} chars, attempting parse`);

      // Pre-process: close unclosed argument tags before parsing
      const fixedStripped = closeUnclosedArgTags(stripped);

      // Try Strategy B: <tool_name>+<parameters> flat pairs (most common format inside containers)
      const names = findAllTagValues(fixedStripped, TOOL_NAME_PATTERNS);
      if (names.length) {
        const namedParams = parseParameterNameAttrs(fixedStripped);
        if (namedParams && Object.keys(namedParams).length > 0 && names.length === 1) {
          const toolName = names[0].trim();
          const argumentsText = JSON.stringify(namedParams);
          log.debug("parser", `[parseMarkupToolCalls] Unclosed container fallback (Strategy B named params): name="${toolName}", args="${argumentsText.slice(0, 100)}"`);
          return [buildParsedToolCall(toolName, argumentsText)];
        }

        const args = findAllArgsValues(fixedStripped, TOOL_ARGS_PATTERNS);
        if (args.length || names.length) {
          const results = [];
          for (let i = 0; i < names.length; i++) {
            const toolName = names[i].trim();
            const toolArgs = i < args.length ? args[i].trim() : "{}";
            if (!toolName) continue;

            let parsedArgs;
            const jsonArgs = parseJsonObject(toolArgs);
            if (jsonArgs) {
              parsedArgs = JSON.stringify(jsonArgs);
            } else {
              const markupArgs = parseMarkupInput(toolArgs);
              if (markupArgs && Object.keys(markupArgs).length > 0) {
                parsedArgs = JSON.stringify(markupArgs);
              } else {
                parsedArgs = toolArgs;
              }
            }
            results.push(buildParsedToolCall(toolName, parsedArgs));
          }
          if (results.length) {
            log.debug("parser", `[parseMarkupToolCalls] Unclosed container fallback (Strategy B): found ${results.length} call(s)`);
            return results;
          }
        }
      }

      // Try parsing the fixed content as standalone blocks
      const innerCalls = parseStandaloneSingularBlocks(fixedStripped, allowedToolNames);
      if (innerCalls.length) {
        log.debug("parser", `[parseMarkupToolCalls] Unclosed container fallback: found ${innerCalls.length} call(s)`);
        return innerCalls;
      }

      // Try other strategies on the fixed content
      const namedBlocks = findNamedToolBlocks(fixedStripped, allowedToolNames);
      if (namedBlocks.length) {
        log.debug("parser", `[parseMarkupToolCalls] Unclosed container fallback (named blocks): found ${namedBlocks.length} call(s)`);
        return namedBlocks.map(({ name, body }) => {
          const argumentsText = parseToolCallArguments(body);
          return buildParsedToolCall(name, argumentsText);
        });
      }

      const toolNameTags = findToolNameTags(fixedStripped, allowedToolNames);
      if (toolNameTags.length) {
        log.debug("parser", `[parseMarkupToolCalls] Unclosed container fallback (tag-name): found ${toolNameTags.length} call(s)`);
        return toolNameTags.map(({ name, body }) => {
          const argumentsText = parseToolCallArguments(body);
          return buildParsedToolCall(name, argumentsText);
        });
      }
    }
  }

  // Pre-process: fix mismatched wrapper closings (e.g. <tool_call ...>...</tool_calls>)
  const preprocessed = fixMismatchedWrapperClosings(source);

  const standaloneCalls = parseStandaloneSingularBlocks(preprocessed, allowedToolNames);
  log.debug("parser", `[parseMarkupToolCalls] Standalone mode: found ${standaloneCalls.length} call(s)`);
  return standaloneCalls;
}

/**
 * Strategy I: Detect raw diff/patch format in model output.
 * When the model outputs patch content directly (without XML tool call tags),
 * detect it and wrap as an ApplyPatch tool call.
 *
 * Supported formats:
 * 1. Claude-style patch: "*** Begin Patch" ... "*** End Patch"
 * 2. Unified diff: "--- a/file" + "+++ b/file" + "@@ ... @@"
 * 3. Simple diff markers: lines starting with "-" and "+" in a plausible patch context
 *
 * Returns a parsed tool call object if detected, or null otherwise.
 */
function tryParseRawPatchFormat(text, allowedToolNames) {
  if (!allowedToolNames?.length) return null;

  const allowed = new Set(allowedToolNames.map(n => toStringSafe(n).trim()).filter(Boolean));
  if (!allowed.has("ApplyPatch")) return null;

  const source = toStringSafe(text).trim();
  if (!source) return null;

  // Pattern 1: Claude-style patch format
  //   *** Begin Patch
  //   *** Modify File: src/foo.js
  //   @@ ... @@
  //   -old line
  //   +new line
  //   *** End Patch
  if (source.includes("*** Begin Patch") || source.includes("*** Add File") || source.includes("*** Delete File") || source.includes("*** Modify File")) {
    log.debug("parser", `[tryParseRawPatchFormat] Detected Claude-style patch format (*** Begin/Modify/Add/Delete)`);
    // The patch content is the entire source — the model outputted it as-is
    const patchContent = source;
    const argumentsText = JSON.stringify({ patch: patchContent });
    return buildParsedToolCall("ApplyPatch", argumentsText);
  }

  // Pattern 2: Unified diff format
  //   --- a/file.js
  //   +++ b/file.js
  //   @@ -10,5 +10,6 @@
  //    context line
  //   -old line
  //   +new line
  const hasDiffHeader = /^---\s+[ab]?\//m.test(source) && /^\+\+\+\s+[ab]?\//m.test(source);
  const hasHunkHeader = /^@@\s+-\d+/m.test(source);
  if (hasDiffHeader && hasHunkHeader) {
    log.debug("parser", `[tryParseRawPatchFormat] Detected unified diff format (---/+++ headers + @@ hunks)`);
    const argumentsText = JSON.stringify({ patch: source });
    return buildParsedToolCall("ApplyPatch", argumentsText);
  }

  // Pattern 3: Partial patch content with diff markers
  // Sometimes the model outputs the patch without proper headers, just
  // lines with -/+ prefixes in a code-block-like structure, preceded by
  // text mentioning "patch" or "edit"
  const hasPatchKeyword = /\b(patch|diff|edit file|modify file|apply patch)\b/i.test(source.slice(0, 500));
  const lines = source.split("\n");
  let removedLines = 0;
  let addedLines = 0;
  for (const line of lines) {
    if (/^-[^-]/.test(line)) removedLines++;
    if (/^\+[^+]/.test(line)) addedLines++;
  }
  // Heuristic: if there are both removals and additions, and at least 3 diff lines total,
  // and the text mentions "patch" or similar, treat it as a patch
  if (hasPatchKeyword && removedLines >= 1 && addedLines >= 1 && (removedLines + addedLines) >= 3) {
    log.debug("parser", `[tryParseRawPatchFormat] Detected informal diff format (${removedLines} removals, ${addedLines} additions, patch keyword present)`);
    const argumentsText = JSON.stringify({ patch: source });
    return buildParsedToolCall("ApplyPatch", argumentsText);
  }

  return null;
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
  let fixed = source.replace(MALFORMED_ATTR_EQUALS, '<$1 name="$2">');
  if (fixed !== source) {
    log.debug("parser", `Preprocessed malformed tags (e.g. <tool_name="..."> → <tool_name name="...">)`);
  }

  // Step 1.6: Fix <tool_name name="WebSearch</tool_name> — name attr value extends to closing tag
  // Extract the tool name and rewrite as <tool_name>WebSearch</tool_name>
  const beforeUnclosed = fixed;
  fixed = fixed.replace(MALFORMED_NAME_ATTR_UNCLOSED, '<$1>$2</$1>');
  if (fixed !== beforeUnclosed) {
    log.debug("parser", `Preprocessed unclosed name attr tags (e.g. <tool_name name="WebSearch</tool_name> → <tool_name>WebSearch</tool_name>)`);
  }

  // Step 1.7: Fix <tool_calls name="Glob"> — treat as tool call wrapper, not container
  // The model sometimes uses the plural container tag with a name= attr as a single tool call.
  // Rewrite <tool_calls name="Glob">...</tool_calls> → <tool_call name="Glob">...</tool_call_name>
  const beforeContainerFix = fixed;
  fixed = fixed.replace(
    /<((?:[a-z0-9_:-]+:)?tool_calls)\s+((?:name|function|tool)\s*=\s*"[^"]+")(?:[^>]*)>([\s\S]*?)<\/\1\s*>/gi,
    (full, tag, attrs, content) => {
      log.debug("parser", `[preprocess] Fixed <${tag} ${attrs}> (container with name attr) → <tool_call ${attrs}>`);
      return `<tool_call ${attrs}>${content}</tool_call_name>`;
    }
  );
  if (fixed !== beforeContainerFix) {
    log.debug("parser", `Preprocessed <tool_calls name="..."> → <tool_call name="...">`);
  }

  // Step 2: Quick check for any tool-related XML tags
  // Also matches <tool name="...">, <tool_name name="...">, <function name="...">
  // Also checks for tool-name-specific tags (e.g. <ApplyPatch>, <Shell>)
  const stripped = stripFencedCodeBlocks(fixed);
  const hasStandardToolTag = stripped.match(/<(?:tool_calls|tool_call|tool_name|tool|function_calls|function_call|function_name|function|invoke|tool_use)\b/i);
  const hasToolNameTag = allowedToolNames.length > 0 && allowedToolNames.some(name => {
    const escaped = name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    return stripped.match(new RegExp(`<${escaped}\\b`, "i"));
  });
  if (!hasStandardToolTag && !hasToolNameTag) {
    // Strategy I: Detect raw diff/patch format for ApplyPatch tool.
    // The model sometimes outputs patch content directly without XML tags:
    //   *** Begin Patch
    //   *** Modify File: src/foo.js
    //   @@ ... @@
    //   -old line
    //   +new line
    //   *** End Patch
    // Or standard unified diff format:
    //   --- a/file.js
    //   +++ b/file.js
    //   @@ ... @@
    const patchCall = tryParseRawPatchFormat(fixed, allowedToolNames);
    if (patchCall) {
      log.info("parser", `[Strategy I] Detected raw patch format, wrapping as ApplyPatch tool call (${fixed.length} chars)`);
      return [patchCall];
    }

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

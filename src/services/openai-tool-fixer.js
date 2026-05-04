import { existsSync, readFileSync } from "node:fs";

/**
 * openai-tool-fixer.js — 容错 JSON 解析 + 工具参数修复
 *
 * 移植自 cursor2api/src/converter.ts (tolerantParse) 和
 * cursor2api/src/tool-fixer.ts (replaceSmartQuotes, repairExactMatchToolArguments)
 */

// ── 智能引号 ──

const SMART_DOUBLE_QUOTES = new Set([
  "«", "“", "”", "❞",
  "‟", "„", "❝", "»",
]);

const SMART_SINGLE_QUOTES = new Set([
  "‘", "’", "‚", "‛",
]);

export function replaceSmartQuotes(text) {
  const chars = [...text];
  return chars.map(ch => {
    if (SMART_DOUBLE_QUOTES.has(ch)) return '"';
    if (SMART_SINGLE_QUOTES.has(ch)) return "'";
    return ch;
  }).join("");
}

// ── 容错 JSON 解析 ──

export function tolerantParse(jsonStr) {
  // L1: 直接解析
  try { return JSON.parse(jsonStr); } catch { /* continue */ }

  // L2: 字符级修复 — 字符串内裸控制字符转义、未闭合括号补齐
  let inString = false;
  let fixed = "";
  const bracketStack = [];

  for (let i = 0; i < jsonStr.length; i++) {
    const char = jsonStr[i];

    if (char === '"') {
      let backslashCount = 0;
      for (let j = i - 1; j >= 0 && jsonStr[j] === "\\"; j--) backslashCount++;
      if (backslashCount % 2 === 0) inString = !inString;
      fixed += char;
      continue;
    }

    if (inString) {
      if (char === "\n") fixed += "\\n";
      else if (char === "\r") fixed += "\\r";
      else if (char === "\t") fixed += "\\t";
      else fixed += char;
    } else {
      if (char === "{" || char === "[") bracketStack.push(char === "{" ? "}" : "]");
      else if (char === "}" || char === "]") { if (bracketStack.length) bracketStack.pop(); }
      fixed += char;
    }
  }

  if (inString) fixed += '"';
  while (bracketStack.length) fixed += bracketStack.pop();
  fixed = fixed.replace(/,\s*([}\]])/g, "$1");

  try { return JSON.parse(fixed); } catch { /* continue */ }

  // L3: 截断到最后一个完整的顶级对象
  const lastBrace = fixed.lastIndexOf("}");
  if (lastBrace > 0) {
    try { return JSON.parse(fixed.substring(0, lastBrace + 1)); } catch { /* continue */ }
  }

  // L4: 正则提取 tool + parameters（值中有未转义引号）
  try {
    const toolMatch = jsonStr.match(/"(?:tool|name)"\s*:\s*"([^"]+)"/);
    if (toolMatch) {
      const toolName = toolMatch[1];
      const paramsMatch = jsonStr.match(/"(?:parameters|arguments|input)"\s*:\s*(\{[\s\S]*)/);
      let params = {};
      if (paramsMatch) {
        const paramsStr = paramsMatch[1];
        let depth = 0, end = -1, pInString = false;
        for (let i = 0; i < paramsStr.length; i++) {
          const c = paramsStr[i];
          if (c === '"') {
            let bsc = 0;
            for (let j = i - 1; j >= 0 && paramsStr[j] === "\\"; j--) bsc++;
            if (bsc % 2 === 0) pInString = !pInString;
          }
          if (!pInString) {
            if (c === "{") depth++;
            if (c === "}") { depth--; if (depth === 0) { end = i; break; } }
          }
        }
        if (end > 0) {
          const rawParams = paramsStr.substring(0, end + 1);
          try {
            params = JSON.parse(rawParams);
          } catch {
            const fieldRegex = /"([^"]+)"\s*:\s*"((?:[^"\\]|\\.)*)"/g;
            let fm;
            while ((fm = fieldRegex.exec(rawParams)) !== null) {
              params[fm[1]] = fm[2].replace(/\\n/g, "\n").replace(/\\t/g, "\t");
            }
          }
        }
      }
      return { tool: toolName, parameters: params };
    }
  } catch { /* continue */ }

  // L5: 逆向贪婪大值提取（Write/Edit content 含未转义引号）
  try {
    const toolMatch2 = jsonStr.match(/["'](?:tool|name)["']\s*:\s*["']([^"']+)["']/);
    if (toolMatch2) {
      const toolName = toolMatch2[1];
      const params = {};

      const smallFieldRegex = /"(file_path|path|file|old_string|old_str|insert_line|mode|encoding|description|language|name)"\s*:\s*"((?:[^"\\]|\\.)*)"/g;
      let sfm;
      while ((sfm = smallFieldRegex.exec(jsonStr)) !== null) {
        params[sfm[1]] = sfm[2].replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\\\/g, "\\");
      }

      const bigValueFields = ["content", "command", "text", "new_string", "new_str", "file_text", "code"];
      for (const field of bigValueFields) {
        const fieldStart = jsonStr.indexOf(`"${field}"`);
        if (fieldStart === -1) continue;
        const colonPos = jsonStr.indexOf(":", fieldStart + field.length + 2);
        if (colonPos === -1) continue;
        const valueStart = jsonStr.indexOf('"', colonPos);
        if (valueStart === -1) continue;

        let valueEnd = jsonStr.length - 1;
        while (valueEnd > valueStart && /[}\]\s,]/.test(jsonStr[valueEnd])) valueEnd--;
        if (jsonStr[valueEnd] === '"' && valueEnd > valueStart + 1) {
          const rawValue = jsonStr.substring(valueStart + 1, valueEnd);
          try {
            params[field] = JSON.parse(`"${rawValue}"`);
          } catch {
            params[field] = rawValue
              .replace(/\\n/g, "\n").replace(/\\t/g, "\t").replace(/\\r/g, "\r")
              .replace(/\\\\/g, "\\").replace(/\\"/g, '"');
          }
        }
      }

      if (Object.keys(params).length > 0) return { tool: toolName, parameters: params };
    }
  } catch { /* continue */ }

  // 全失败，返回空对象
  return {};
}

// ── 工具参数修复 ──

function buildFuzzyPattern(text) {
  const parts = [];
  for (const ch of text) {
    if (SMART_DOUBLE_QUOTES.has(ch) || ch === '"') {
      parts.push('["«“”❞‟„❝»]');
    } else if (SMART_SINGLE_QUOTES.has(ch) || ch === "'") {
      parts.push("['‘’‚‛]");
    } else if (ch === " " || ch === "\t") {
      parts.push("\\s+");
    } else if (ch === "\\") {
      parts.push("\\\\{1,2}");
    } else {
      parts.push(ch.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
    }
  }
  return parts.join("");
}

export function repairExactMatchToolArguments(toolName, args) {
  if (!args || typeof args !== "object") return args;

  const lowerName = (toolName || "").toLowerCase();
  if (!lowerName.includes("str_replace") && !lowerName.includes("search_replace") && !lowerName.includes("strreplace")) {
    return args;
  }

  const oldString = args.old_string ?? args.old_str;
  if (!oldString) return args;

  const filePath = args.path ?? args.file_path;
  if (!filePath) return args;

  try {
    if (!existsSync(filePath)) return args;
    const content = readFileSync(filePath, "utf-8");

    if (content.includes(oldString)) return args;

    const pattern = buildFuzzyPattern(oldString);
    const regex = new RegExp(pattern, "g");
    const matches = [...content.matchAll(regex)];

    if (matches.length !== 1) return args;

    const matchedText = matches[0][0];

    if ("old_string" in args) args.old_string = matchedText;
    else if ("old_str" in args) args.old_str = matchedText;

    const newString = args.new_string ?? args.new_str;
    if (newString) {
      const fixed = replaceSmartQuotes(newString);
      if ("new_string" in args) args.new_string = fixed;
      else if ("new_str" in args) args.new_str = fixed;
    }
  } catch {
    // best-effort: 文件读取失败不阻塞请求
  }

  return args;
}

// ── 统一入口 ──

export function fixToolCallArguments(toolName, args) {
  if (!args || typeof args !== "object") return args;
  args = repairExactMatchToolArguments(toolName, args);
  return args;
}

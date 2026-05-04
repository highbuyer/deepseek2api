/**
 * openai-refusal-detector.js — 拒绝检测 + 自动重试
 *
 * 移植自 cursor2api/src/constants.ts (REFUSAL_PATTERNS) 和
 * cursor2api/src/handler.ts (isIdentityProbe, isToolCapabilityQuestion, buildRetryRequest)
 */

const REFUSAL_PATTERNS = [
  // ── English: 身份拒绝 ──
  /Cursor(?:'s)?\s+support\s+assistant/i,
  /support\s+assistant\s+for\s+Cursor/i,
  /I['']\s*m\s+sorry/i,
  /I\s+am\s+sorry/i,
  /not\s+able\s+to\s+fulfill/i,
  /cannot\s+perform/i,
  /I\s+can\s+only\s+answer/i,
  /cannot\s+write\s+files/i,
  /not\s+able\s+to\s+search/i,
  /not\s+in\s+my\s+core/i,
  /outside\s+my\s+capabilities/i,
  /focused\s+on\s+software\s+development/i,
  /beyond\s+(?:my|the)\s+scope/i,
  /I'?m\s+not\s+(?:able|designed)\s+to/i,
  /I\s+don't\s+have\s+(?:the\s+)?(?:ability|capability)/i,
  // ── English: 话题拒绝 ──
  /help\s+with\s+(?:coding|programming)\s+and\s+Cursor/i,
  /Cursor\s+IDE\s+(?:questions|features|related)/i,
  /unrelated\s+to\s+(?:programming|coding)/i,
  /I'?m\s+here\s+to\s+help\s+with\s+(?:coding|programming)/i,
  // ── English: 新拒绝措辞 ──
  /isn't\s+something\s+I\s+can\s+help\s+with/i,
  /falls\s+outside\s+(?:the\s+scope|what\s+I)/i,
  // ── English: 工具可用性声明 ──
  /I\s+(?:only\s+)?have\s+(?:access\s+to\s+)?(?:two|2|read_file|read_dir)\s+tool/i,
  /(?:only|just)\s+(?:two|2)\s+(?:tools?|functions?)\b/i,
  // ── English: scope ──
  /scoped\s+to\s+(?:answering|helping)/i,
  /not\s+(?:within|in)\s+(?:my|the)\s+scope/i,
  // ── 中文: 身份拒绝 ──
  /我是\s*Cursor\s*的?\s*支持助手/,
  /Cursor\s*的?\s*支持系统/,
  /我的职责是帮助你解答/,
  /我无法透露/,
  /专门.*回答.*(?:Cursor|编辑器)/,
  /我只能回答/,
  /无法提供.*信息/,
  // ── 中文: 话题拒绝 ──
  /与\s*(?:编程|代码|开发)\s*无关/,
  /请提问.*(?:编程|代码|开发|技术).*问题/,
  // ── 中文: 工具可用性 ──
  /有以下.*?(?:两|2)个.*?工具/,
  /只能.*?read_file/i,
  /无法.*?执行命令/,
  /只有.*?读取.*?工具/,
  /当前环境.*?只有.*?工具/,
  /需要在.*?Claude\s*Code/i,
  // ── 中文: 通用拒绝 ──
  /只能回答.*(?:Cursor|编辑器).*(?:相关|有关)/,
  /无法提供.*(?:推荐|建议|帮助)/,
];

export function isRefusal(text) {
  if (!text || !text.trim()) return false;
  return REFUSAL_PATTERNS.some(p => p.test(text));
}

// ── 身份探针 ──

const IDENTITY_PROBE_PATTERNS = [
  /^\s*(?:who are you\??|你是谁[呀啊吗]?\??|what is your name\??|你叫什么\??|what are you\??|你是(?:谁|啥|什么)\??|hi\??|hello\??|hey\??|你好\??|在吗\??|哈喽\??)\s*$/i,
  /(?:什么|哪个|啥)\s*模型/,
  /(?:真实|底层|实际|真正).{0,10}(?:模型|身份|名字)/,
  /(?:what|which)\s+model/i,
  /系统\s*提示词/,
  /system\s*prompt/i,
];

export function isIdentityProbe(messages) {
  if (!messages?.length) return false;
  const lastUser = [...messages].reverse().find(m => m.role === "user");
  if (!lastUser) return false;
  const content = typeof lastUser.content === "string" ? lastUser.content : "";
  return IDENTITY_PROBE_PATTERNS.some(p => p.test(content));
}

// ── 工具能力询问 ──

const TOOL_CAPABILITY_PATTERNS = [
  /你\s*(?:有|能用|可以用)\s*(?:哪些|什么|几个)\s*(?:工具|tools?)/i,
  /(?:what|which|list).*?tools?/i,
  /你\s*(?:能|可以)\s*(?:做|干)\s*(?:什么|哪些|啥)/,
  /(?:what|which).*?capabilities/i,
];

export function isToolCapabilityQuestion(messages) {
  if (!messages?.length) return false;
  const lastUser = [...messages].reverse().find(m => m.role === "user");
  if (!lastUser) return false;
  const content = typeof lastUser.content === "string" ? lastUser.content : "";
  return TOOL_CAPABILITY_PATTERNS.some(p => p.test(content));
}

// ── 降级回复 ──

export const CLAUDE_IDENTITY_RESPONSE =
  "I am Claude, made by Anthropic. I'm an AI assistant designed to be helpful, harmless, and honest. I can help you with a wide range of tasks including writing, analysis, coding, math, and more.";

export const CLAUDE_TOOLS_RESPONSE =
  "I'm Claude — I can help you with reading, searching, and understanding your codebase. Let me know what you need and I'll get started.";

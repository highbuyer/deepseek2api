# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Critical Rules

- **NEVER kill or interrupt running processes** without explicit user permission. Always ask before stopping any long-running process (server, test suite, data generation).
- **Stay focused on the immediate request.** Do not over-analyze, go on tangents, or suggest unrelated improvements unless explicitly asked. Execute exactly what was requested with zero extra commentary.
- **Verify assumptions before making recommendations.** Read configs to confirm settings — never assume defaults about the environment, model, or infrastructure.
- **Rank issues by architectural impact before multi-commit fixes.** Prioritize structural/fundamental fixes over surface patches (regex tweaks, rule adjustments).

## 项目概述

deepseek2api 是一个纯 Node.js 代理服务，提供 DeepSeek Web 控制台 + OpenAI 兼容 API 桥接。它将 Cursor IDE 发出的 OpenAI 格式请求转换为 DeepSeek Web API 调用，核心工作是把工具调用（tool calls）的 XML 输出解析回 OpenAI JSON 格式。

## 常用命令

```bash
npm start          # 启动服务，默认 http://127.0.0.1:3000
DEBUG=1 npm start  # 带调试日志启动，日志同时写入 data/debug.log
node --test test/openai-tool-parser.test.js  # 运行工具调用解析器测试（74 个 case）
```

## 工具调用数据流（核心架构）

`src/services/` 中的文件按此顺序构成完整管线：

1. **`openai-request.js`** — 解析 OpenAI 请求（model、tools、tool_choice），返回 `requestOptions`
2. **`openai-tool-policy.js`** — 解析 `tool_choice`（none/auto/required/forced），生成 `toolChoicePolicy`
3. **`openai-tool-prompt.js`** — 根据 `toolChoicePolicy` 注入工具 schema 到 system prompt
4. **`openai-completion-runner.js`** — 代理 DeepSeek SSE 流，注入 `<think>`/`</think>` 标签区分 thinking 和 response 内容，调用 `onText(text, kind)` 回调
5. **`openai-tool-sieve.js`** — **流式文本过滤器**，`createToolSieve(allowedToolNames)` 返回 `push(text, kind)` 方法。它在字节流中识别 `<tool_calls>` 等 XML 块的开始/结束，捕获完整 XML 并调用解析器，提取工具调用。非工具调用的文本以 `{ type: "text", text, kind }` 事件透传
6. **`openai-tool-parser.js`** — **离线解析器**，78 个函数、~1800 行。接收完整文本，输出 `[{ name, argumentsText, id }]` 数组。支持 8 种以上的预处理步骤处理 DeepSeek 的畸形 XML 输出
7. **`openai-bridge.js`** — 连接管线。`streamCompletionContent({ onText })` → sieve.push → 消费事件：工具调用走 `emitToolCalls`（写入 SSE `delta.tool_calls`），文本走 `emitTextEvent`（thinking → `reasoning_content`，response → `content`）

## DeepSeek SSE 与 thinking 标签

DeepSeek SSE 使用 `type` 字段区分内容类型：
- `type: "THINK"` → kind = `"thinking"` → 注入 `<think>内容</think>` → 桥接发给 Cursor 的 `reasoning_content`
- `type: "RESPONSE"` → kind = `"response"` → 注入 `</think>` 关闭思考标签 → 桥接发给 Cursor 的 `content`

`openai-completion-runner.js` 中的 `THINK_OPEN_TAG` 和 `THINK_CLOSE_TAG` 控制标签格式。不要改为 VSION（以前有过这个 typo）。

## 筛子（Sieve）捕获机制

`createToolSieve` 采用 Claude Code 的结构化事件模型（detect → buffer → parse）：
- 非捕获模式：扫描文本中的工具块开头（`<function_calls`、`<tool_calls`、`<invoke` 等），检测到后进入捕获
- 捕获模式：累积文本直到找到闭合标签，调用 `parseToolCallsFromText` 解析
- `<tool_result>` 块被检测到后直接丢弃（防止 prompt 回显泄漏到 SSE）
- 捕获限制 `50_000` 字符，超过后 flush 为文本
- chunk 边界检测（LOOKBEHIND=30）：保留尾部 `<` 片段防止标签被截断
- 解析失败时 emit `format_error` 事件，桥接层即时返回纠正指令（Claude Code `tool_use_error` 模式）
- **`formatErrorEmitted` 标记**：每个流只发一次 `format_error`，防止模型回显纠正文本导致无限循环

## 解析器（Parser）预处理步骤

`parseToolCallsFromText()` 先串行执行预处理（Step 1.1 ~ 1.68），再进入策略匹配：

- Step 1.1：替换智能引号（15 种 Unicode → ASCII），防 JSON.parse 失败
- Step 1.2：剥离 `<think>...</think>` 块（防止思考内容中的 XML 被误解析）
- Step 1.5：修复 `<tool_name="ReadFile">` → `<tool_name name="ReadFile">`
- Step 1.6：修复属性值未闭合：`<tool_name name="X</tool_name>`
- Step 1.55：修复 `</parameter>` 错误闭合 `<tool_name name="X">`
- Step 1.57：修复 `<tool_call_name="X">`、`<tool_params>`、`<tool_type>` 等模型自创标签
- Step 1.65：修复 JSON-in-XML：`<target_directory": "/path"</target_directory>`
- Step 1.68：修复 `<parameter name="key" value="val</parameter>`
- Step 1.7：修复 `<tool_calls name="Glob">` （容器带 name 属性 → 转为 tool_call）

预处理完成后，解析器按策略 A（嵌套 tool_call 块）、策略 B（tool_name + parameters）、策略 C（KV 标签匹配）等提取工具调用。参数 JSON 解析失败时走 `tolerantParse`（5 层容错）和 `fixToolCallArguments`（智能引号 + 模糊匹配）fallback。

## 新增模块

- **`openai-tool-fixer.js`** — 容错 JSON 解析（5 层 fallback）+ 智能引号替换 + 工具参数模糊匹配修复
- **`openai-refusal-detector.js`** — 拒绝检测（40+ 中英文正则）+ 身份探针检测 + 降级回复
- **`request-logger.js`** — 请求级 JSONL 日志，记录 TTFT、阶段耗时、工具调用数
- **`api-error.js`** — 统一 ApiError 类，替代散落的 Error + .code/.statusCode 模式
- **`utils/tool-truncation.js`** — 共享截断工具（预算计算、头尾裁切、预算替换），openai-prompt 和 anthropic-prompt 共用

## 三层防线（对齐 Claude Code）

1. **单条截断指令**：`truncateToolResult` — 超过预算时 DO NOT / Instead 指令，禁止整文件重读
2. **聚合预算替换**：70K per-message 硬上限 — 超出部分用预览替换，指示模型偏移量/Grep
3. **主动建议**：Read 结果占 prompt > 5% 时注入 `<system-reminder>` 提醒模型换策略

## 关键设计决策

- **工具调用的 text 事件不再写入 SSE**。桥接中的 `emitTextEvent` 仅当筛子显式返回 `{ type: "text" }` 事件时才写入，且会根据 `event.kind` 选择 `reasoning_content` 或 `content`
- **筛子暴露 `push(text, kind)`**，kind 参数传透到 text 事件中。桥接消费时可直接按 kind 写不同的 SSE 字段
- **工具调用 XML 只在筛子/解析器内部使用**。对外（Cursor）始终输出标准 OpenAI JSON 格式的 tool_calls
- 测试文件 `test/openai-tool-parser.test.js` 是 parser 和 sieve 的唯一测试，包含真实 DeepSeek 畸形输出的回归 case
- **遇难点先搜 Claude Code 源码**（`/home/langshen/Desktop/claude-code-source`，已 GitNexus 索引，52242 符号）。流式处理、工具解析、提示构建等实现问题，先参考 Claude Code 的做法再移植，不凭空设计

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **deepseek2api** (2096 symbols, 4716 relationships, 182 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## SSE Streaming Constraints

- **`emitToolCalls` MUST keep `id`/`type`/`function.name`/`function.arguments` in ONE delta chunk.** Splitting function fields across chunks causes Cursor's SSE merger to replace `{name}` with `{arguments}` (Cursor does shallow replace, not deep merge). The resulting empty `function.name` manifests as "unsupported local tool."
- **Fallback tool call detection MUST run BEFORE `finish_reason`.** If `parseToolCallsFromText` finds tool calls the sieve missed, they must be emitted before the final SSE chunk. Sending tool_calls after `finish_reason:"stop"` violates the protocol — clients stop processing after receiving finish_reason.
- **`getToolName` fallback chain**: `function.name` → `tool.name` → argument-based inference. Cursor clears `function.name` to `""` when reconstructing tool_calls from SSE; the inference layer recovers the tool name from parameter keys (`command`→Shell, `path`→ReadFile, `pattern`+`path`→rg, etc.).
- **`FORMAT_ERROR_MSG` is the single authoritative correction message** (in `openai-tool-sieve.js`). Do NOT add duplicate hardcoded `[ERROR:]` messages in the bridge — they create inconsistent UI rendering across clients.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/deepseek2api/context` | Codebase overview, check index freshness |
| `gitnexus://repo/deepseek2api/clusters` | All functional areas |
| `gitnexus://repo/deepseek2api/processes` | All execution flows |
| `gitnexus://repo/deepseek2api/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->

# DeepSeek2API

> 一个纯 Node.js 的 DeepSeek Web 控制台 + OpenAI 兼容桥接服务。

它把本地用户体系、DeepSeek 账号绑定、API Key 管理、DeepSeek 原生代理调试和 OpenAI 兼容接口放进同一个可直接运行的项目里。

## 功能概览

| 模块 | 能力 |
| --- | --- |
| 控制台 UI | 注册 / 登录、本地用户隔离、DeepSeek 账号绑定、API Key 管理 |
| OpenAI 兼容层 | `GET /v1/models`、`POST /v1/chat/completions`、`POST /v1/responses`、`GET /v1/responses/:id` |
| 工具调用 | 仅适配 chat / responses 协议；按 API Key 单独开关 |
| 原生代理层 | 提供 `/proxy/*` 白名单转发，便于调试和复用 DeepSeek Web 接口 |
| 管理后台 | 注册开关、邀请码、用户启用 / 禁用 / 删除、并发 / 速率限制 |
| 无痕模式 | 支持全局或用户级无痕，请求完成后自动清理会话 |
| 部署形态 | 无第三方运行时依赖，`npm start` 即可启动 |

## 项目特点

- 纯 Node.js 原生 HTTP 服务，无 Express、无数据库、无构建步骤
- 前后端都在同一个仓库里，静态资源由服务端直接托管
- 运行状态统一保存在 `data/app.json`
- DeepSeek token 失效时会自动重新登录并刷新
- 遇到 PoW 保护接口时会自动获取 wasm 并求解挑战
- OpenAI 兼容层同时支持流式和非流式响应
- `deepseek-reasoner-*` 模型会把思维内容包在 `<think>...</think>`
- API Key 请求会在当前用户可见账号之间轮询

## 运行要求

- Node.js 18+
- 服务端能够访问 [https://chat.deepseek.com](https://chat.deepseek.com)
- 浏览器在绑定 DeepSeek 账号时需要访问 [https://cdn.deepseek.com](https://cdn.deepseek.com)
- 如触发 PoW 校验，服务端还需要访问 [https://fe-static.deepseek.com](https://fe-static.deepseek.com)

## 快速开始

### 1. 启动服务

```bash
npm start
```

默认监听地址：

```text
http://127.0.0.1:3000
```

### 2. 可选：创建本地配置

仓库不自带 `.env`。如需启用管理员入口或修改端口，可参考 `.env.example` 手动创建：

```bash
cp .env.example .env
```

Windows PowerShell：

```powershell
Copy-Item .env.example .env
```

`.env.example` 内容：

```env
PORT=3000
APP_ADMIN_USERNAME=
APP_ADMIN_PASSWORD=
```

### 3. 打开控制台

浏览器访问 `http://127.0.0.1:3000`，然后按下面流程使用：

1. 注册本地用户，或使用管理员账号登录
2. 在“账号”页绑定 DeepSeek 账号
3. 在“密钥”页创建 API Key
4. 如需工具调用，为该 API Key 单独打开“工具调用”开关
5. 使用内置聊天工作区，或通过 OpenAI 兼容接口接入客户端

## 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `PORT` | `3000` | 服务监听端口 |
| `DEBUG` | `0` | 开启调试日志（设为 `1` 生效） |
| `APP_ADMIN_USERNAME` | 空 | 管理员用户名 |
| `APP_ADMIN_PASSWORD` | 空 | 管理员密码 |
| `MAX_PROMPT_CHARS` | `128000` | 字符级 fallback 上限（仅当 `MAX_PROMPT_TOKENS` 未设时启用） |
| `MAX_PROMPT_TOKENS` | `32000` | token 感知上限。V4 专家模式实测窗口 ~38K real tokens（三种变体共享），默认留 6K buffer。遇 `input_exceeds_limit` 报错调低此值 |
| `MAX_TOOL_DESC_CHARS` | `200` | 工具描述最大字符数 |
| `PROXY` | — | HTTP 代理地址（如 `http://127.0.0.1:10808`） |

只有同时设置 `APP_ADMIN_USERNAME` 和 `APP_ADMIN_PASSWORD` 时，管理员入口才会启用。

## 控制台能力

### 账号与密钥

- 绑定 / 删除 DeepSeek Web 账号
- 为当前用户创建多个 API Key
- API Key 可指定自定义明文，留空则自动生成
- API Key 可单独开启或关闭“工具调用”
- 创建 API Key 时可直接设置工具调用开关
- OpenAI 兼容请求会在当前用户可见账号之间轮询

### 管理后台

- 管理本地注册开关
- 控制是否必须使用邀请码注册
- 生成、删除、批量删除邀请码
- 禁用、启用、删除本地用户
- 为用户设置并发上限和每分钟请求上限

### 无痕模式

- 管理员可开启全局无痕
- 普通用户可只为自己开启无痕
- 开启后，请求完成后会自动清理相关 DeepSeek 会话

## OpenAI 兼容接口

### 支持的接口

- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/responses`
- `GET /v1/responses/:id`

### 模型说明

- 默认模型：`deepseek-chat-fast`
- 联网能力通过模型后缀 `-search` 控制
- 不支持 `web_search_options`，请改用 `*-search` 模型

支持的模型 ID（12 个）：

| 模型 | model_type | thinking |
|------|-----------|----------|
| `deepseek-chat-fast` | default | off |
| `deepseek-chat-fast-search` | default | off |
| `deepseek-reasoner-fast` | default | on |
| `deepseek-reasoner-fast-search` | default | on |
| `deepseek-chat-expert` | expert | off |
| `deepseek-chat-expert-search` | expert | off |
| `deepseek-reasoner-expert` | expert | on |
| `deepseek-reasoner-expert-search` | expert | on |
| `deepseek-vision` | vision | on |
| `deepseek-vision-search` | vision | on |
| `deepseek-vision-fast` | vision | off |
| `deepseek-vision-fast-search` | vision | off |

#### GPT 别名（MODEL_ALIASES）

`/v1/chat/completions` 和 `/v1/responses` 支持用 GPT 模型名，自动映射到 DeepSeek：

| GPT 别名 | DeepSeek 模型 |
|----------|-------------|
| `gpt-4` `gpt-4-turbo` `gpt-3.5-turbo` `gpt-4o-mini` | `deepseek-chat-fast` |
| `gpt-4o` `gpt-4.1` | `deepseek-chat-expert` |
| `o1` `o3` | `deepseek-reasoner-fast` |
| `gpt-5.5` `gpt-5` `o4-mini` `o3-mini` `o1-mini` | `deepseek-reasoner-expert` |
| `gpt-5.5-vision` | `deepseek-vision` |
| `gpt-5.5-vision-fast` | `deepseek-vision-fast` |

#### Claude 别名（ANTHROPIC_MODEL_ALIASES）

`/v1/messages` 支持用 Claude 模型名：

| Claude 别名 | DeepSeek 模型 |
|------------|-------------|
| `claude-haiku-4-5` `claude-3-5-haiku` | `deepseek-chat-fast` |
| `claude-sonnet-4-6` `claude-sonnet-4-5` `claude-3-5-sonnet` | `deepseek-chat-expert` |
| `claude-opus-4-5` `claude-opus-4` | `deepseek-reasoner-fast` |
| `claude-opus-4-7` | `deepseek-reasoner-expert` |
| `claude-opus-4-7-vision` | `deepseek-vision` |
| `claude-opus-4-7-vision-fast` | `deepseek-vision-fast` |

> `-search` 变体没有别名，用原生名直接调用。

### `chat/completions` 示例

```bash
curl http://127.0.0.1:3000/v1/chat/completions \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat-fast",
    "messages": [
      { "role": "user", "content": "hello" }
    ]
  }'
```

### `responses` 示例

```bash
curl http://127.0.0.1:3000/v1/responses \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-chat-fast",
    "input": "hello"
  }'
```

### 工具调用

- 工具调用仅适配 `chat/completions` 和 `responses`
- 协议入口始终存在；是否允许工具调用由 API Key 的开关决定
- API Key 未开启工具调用时：
  - 普通请求可正常使用
  - 带 `tools`、`tool_choice`、工具历史消息的请求会直接返回 `400`
- API Key 开启工具调用时：
  - 服务会把工具 schema 注入提示词
  - 再把模型输出中的工具 XML 解析回 OpenAI 兼容的工具调用结构

### 工具调用行为说明

- 当前实现本质上是“提示词注入 + 输出解析”，不是上游原生 tool calling
- 提示词允许模型在工具调用前、后或前后都输出普通文本
- 普通文本是否出现、出现在哪一侧，由模型自己决定，不做强制
- `chat/completions` 非流式：
  - 如果识别到工具调用，响应会同时返回 `message.tool_calls`
  - 如果模型在工具调用前后还输出了普通文本，文本会保留在 `message.content`
- `chat/completions` 流式：
  - 普通文本继续走 `delta.content`
  - 工具调用走 `delta.tool_calls`
  - 工具调用事件出现的位置不固定，取决于模型实际输出顺序
- `responses` 非流式：
  - 混合输出会按顺序拆成 `output` 数组中的多个 item
  - 典型形态是 `message -> function_call -> message`
- `responses` 流式：
  - 文本段会逐段生成独立的 message item
  - 工具调用会生成 function_call item
  - 事件顺序与模型实际输出顺序一致

### 当前限制

- 只识别 XML / Markup 风格的工具调用块
- 不识别把 `"tool_calls": [...]` 当普通文本吐出来的 JSON 片段
- `previous_response_id` 目前不支持
- 混合输出是否稳定出现，和所选模型强相关；`deepseek-reasoner-*` 通常比 `deepseek-chat-*` 更容易产出“文本 + 工具调用”的混合结果

## 原生代理接口

### 支持的接口

- `GET /proxy/...`
- `POST /proxy/...`

### 使用说明

- `/proxy/*` 走的是登录态会话，不是 API Key 鉴权
- 如果存在多个可用账号，可通过请求头 `x-proxy-account-id` 指定账号
- 只允许转发白名单路径，白名单定义在 `src/config.js`

当前白名单包含：

- `/api/v0/chat/completion`
- `/api/v0/chat/continue`
- `/api/v0/chat/create_pow_challenge`
- `/api/v0/chat/edit_message`
- `/api/v0/chat/history_messages`
- `/api/v0/chat/message_feedback`
- `/api/v0/chat/regenerate`
- `/api/v0/chat/resume_stream`
- `/api/v0/chat/stop_stream`
- `/api/v0/chat_session/create`
- `/api/v0/chat_session/delete`
- `/api/v0/chat_session/delete_all`
- `/api/v0/chat_session/fetch_page`
- `/api/v0/chat_session/update_pinned`
- `/api/v0/chat_session/update_title`
- `/api/v0/client/settings`
- `/api/v0/download_export_history`
- `/api/v0/export_all`
- `/api/v0/file/fetch_files`
- `/api/v0/file/preview`
- `/api/v0/file/upload_file`
- `/api/v0/share/content`
- `/api/v0/share/create`
- `/api/v0/share/delete`
- `/api/v0/share/fork`
- `/api/v0/share/list`
- `/api/v0/users/current`
- `/api/v0/users/settings`
- `/api/v0/users/update_settings`

## 本地接口总览

### 公共接口

| Method | Path | 鉴权 | 说明 |
| --- | --- | --- | --- |
| `GET` | `/api/me` | 无 | 返回当前会话；未登录时返回匿名态 |
| `GET` | `/api/discovery` | 无 | 返回允许的 `/proxy/*` 白名单路径 |
| `POST` | `/api/auth/login` | 无 | 本地用户登录 |
| `POST` | `/api/auth/register` | 无 | 本地用户注册 |
| `POST` | `/api/auth/logout` | 无 | 清理本地登录态 |

### 登录后接口

| Method | Path | 鉴权 | 说明 |
| --- | --- | --- | --- |
| `GET` | `/api/accounts` | Session | 列出当前用户可见账号 |
| `POST` | `/api/accounts` | Session | 创建并绑定 DeepSeek 账号 |
| `DELETE` | `/api/accounts/:id` | Session | 删除账号 |
| `POST` | `/api/incognito` | Session | 更新当前用户无痕配置 |
| `GET` | `/api/api-keys` | Session | 列出当前用户 API Key |
| `POST` | `/api/api-keys` | Session | 创建 API Key |
| `PATCH` | `/api/api-keys/:id` | Session | 更新 API Key 开关 |
| `DELETE` | `/api/api-keys/:id` | Session | 删除 API Key |

### 管理接口

| Method | Path | 鉴权 | 说明 |
| --- | --- | --- | --- |
| `POST` | `/api/admin/registration` | Admin | 更新注册开关和邀请码要求 |
| `POST` | `/api/admin/invites` | Admin | 创建邀请码 |
| `POST` | `/api/admin/invites/batch-delete` | Admin | 批量删除邀请码 |
| `DELETE` | `/api/admin/invites/:id` | Admin | 删除单个邀请码 |
| `PATCH` | `/api/admin/users/:id` | Admin | 更新用户状态和限流参数 |
| `DELETE` | `/api/admin/users/:id` | Admin | 删除单个用户 |
| `POST` | `/api/admin/users/batch-disable` | Admin | 批量启用或禁用用户 |
| `POST` | `/api/admin/users/batch-delete` | Admin | 批量删除用户 |

### OpenAI 兼容接口

| Method | Path | 鉴权 | 说明 |
| --- | --- | --- | --- |
| `GET` | `/v1/models` | API Key | 返回兼容模型列表 |
| `POST` | `/v1/messages` | API Key | Anthropic 风格 messages 接口 |
| `POST` | `/v1/chat/completions` | API Key | Chat Completions |
| `POST` | `/v1/embeddings` | API Key | Embeddings；缺少 API Key 时会尝试使用本地第一个 key 兜底 |
| `POST` | `/v1/responses` | API Key | Responses API |
| `GET` | `/v1/responses/:id` | API Key | 读取已缓存的 response |

### 原生代理接口

| Method | Path | 鉴权 | 说明 |
| --- | --- | --- | --- |
| `GET` | `/proxy/<allowed-path>` | Session | 将白名单路径转发到 DeepSeek Web 接口 |
| `POST` | `/proxy/<allowed-path>` | Session | 将白名单路径转发到 DeepSeek Web 接口 |
| `HEAD` | `/proxy/<allowed-path>` | Session | 代码路径支持透传；仅对白名单路径生效 |

补充说明：

- `/proxy/*` 使用登录态会话，不使用 API Key
- `/proxy/<allowed-path>` 中的 `<allowed-path>` 必须来自上面的白名单列表
- `POST /proxy/api/v0/chat/completion` 会在本地做额外处理，以兼容流式 / 非流式和无痕清理
- 全局 `OPTIONS` 预检请求直接返回 `204`

## 项目结构

```text
.
├─ data/                  # 运行时数据目录
├─ public/                # 前端控制台静态资源
├─ src/
│  ├─ routes/             # 公共 / 私有 / 管理 / OpenAI / 代理路由
│  ├─ services/           # 账号、用户、桥接、PoW、限流等核心逻辑
│  ├─ storage/            # JSON 文件存储
│  └─ utils/              # HTTP、SSE、ID、Prompt 等工具
├─ .env.example
├─ package.json
└─ README.md
```

## Codex 当前会话工具实测

以下内容仅适用于 **Codex 在本次会话中的实际运行环境**。
**不代表 Cursor、Claude Code 或其他客户端的工具能力与行为。**

### 已实测可调用并返回有效结果

- `functions.exec_command`
- `functions.write_stdin`
- `functions.list_mcp_resources`
- `functions.list_mcp_resource_templates`
- `functions.read_mcp_resource`
- `functions.update_plan`
- `functions.apply_patch`
- `functions.view_image`
- `functions.spawn_agent`
- `functions.send_input`
- `functions.resume_agent`
- `functions.wait_agent`
- `functions.close_agent`
- `functions.request_user_input`
- `multi_tool_use.parallel`

- `mcp__gitnexus__.list_repos`
- `mcp__gitnexus__.query`
- `mcp__gitnexus__.context`
- `mcp__gitnexus__.impact`
- `mcp__gitnexus__.detect_changes`
- `mcp__gitnexus__.cypher`
- `mcp__gitnexus__.rename`

- `web.run`
  本次会话中已实测通过的子操作：
  `search_query`、`open`、`click`、`find`、`screenshot`、`finance`、`weather`、`sports`、`time`

- `mcp__ChromeDevTools__.list_pages`
- `mcp__ChromeDevTools__.select_page`
- `mcp__ChromeDevTools__.new_page`
- `mcp__ChromeDevTools__.close_page`
- `mcp__ChromeDevTools__.navigate_page`
- `mcp__ChromeDevTools__.take_snapshot`
- `mcp__ChromeDevTools__.take_screenshot`
- `mcp__ChromeDevTools__.click`
- `mcp__ChromeDevTools__.hover`
- `mcp__ChromeDevTools__.drag`
- `mcp__ChromeDevTools__.fill`
- `mcp__ChromeDevTools__.fill_form`
- `mcp__ChromeDevTools__.type_text`
- `mcp__ChromeDevTools__.press_key`
- `mcp__ChromeDevTools__.upload_file`
- `mcp__ChromeDevTools__.wait_for`
- `mcp__ChromeDevTools__.evaluate_script`
- `mcp__ChromeDevTools__.handle_dialog`
- `mcp__ChromeDevTools__.list_console_messages`
- `mcp__ChromeDevTools__.get_console_message`
- `mcp__ChromeDevTools__.list_network_requests`
- `mcp__ChromeDevTools__.get_network_request`
- `mcp__ChromeDevTools__.resize_page`
- `mcp__ChromeDevTools__.emulate`
- `mcp__ChromeDevTools__.lighthouse_audit`
- `mcp__ChromeDevTools__.performance_start_trace`
- `mcp__ChromeDevTools__.performance_stop_trace`
- `mcp__ChromeDevTools__.performance_analyze_insight`
- `mcp__ChromeDevTools__.take_memory_snapshot`

### 可调用，但当前仓库或配置下无匹配数据

- `mcp__gitnexus__.route_map`
  当前 `deepseek2api` 索引中没有 `Route` 节点，返回空结果
- `mcp__gitnexus__.shape_check`
  当前索引中没有可做响应 shape 校验的路由数据，返回空结果
- `mcp__gitnexus__.tool_map`
  当前索引中没有 `Tool` 定义节点，返回空结果
- `mcp__gitnexus__.api_impact`
  工具本身可执行，但依赖 route 映射；当前仓库未索引出对应 route
- `mcp__gitnexus__.group_list`
  当前没有配置任何 GitNexus group，返回空结果
- `mcp__gitnexus__.group_sync`
  工具本身可调用，但当前无可同步 group；对不存在的 group 会返回 `Group not found`

### 工具入口存在，但当前后端不支持

- `web.run` 的 `image_query`
  后端返回：`Search type image is not supported`

### 说明

- 这里的“可用”是指 **Codex 当前会话里实际调用过并得到结果**
- 这里的结论 **不自动适用于 Cursor**
- 这里的结论 **也不自动适用于 Claude Code**
- `web.run` 是顶层工具名；`search_query`、`open` 等是其子操作，不是独立顶层工具
- `functions.request_user_input` 虽已实测成功，但是否适合使用，仍受 Codex 当前协作模式约束

## License

This project is licensed under the [MIT License](./LICENSE).

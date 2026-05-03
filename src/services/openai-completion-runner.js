import { createDeepseekDeltaDecoder, createSseParser } from "../utils/deepseek-sse.js";
import { createChatSession, deleteChatSession } from "./chat-session-service.js";
import { proxyDeepseekRequest } from "./deepseek-proxy.js";
import { log } from "../utils/log.js";

const THINK_OPEN_TAG = "<think>";
const THINK_CLOSE_TAG = "</think>";
const THINK_FORCED_CLOSE_TAG = "[proxy]</think>";

function startCompletion({ account, requestOptions, sessionId }) {
  return proxyDeepseekRequest({
    account,
    method: "POST",
    path: "/api/v0/chat/completion",
    body: Buffer.from(
      JSON.stringify({
        chat_session_id: sessionId,
        parent_message_id: null,
        model_type: requestOptions.model.modelType,
        prompt: requestOptions.prompt,
        ref_file_ids: [],
        thinking_enabled: requestOptions.model.thinkingEnabled,
        search_enabled: requestOptions.model.searchEnabled,
        preempt: false
      })
    ),
    headers: { "content-type": "application/json" }
  });
}

function createThinkingTagger() {
  let currentKind = null;

  return {
    get kind() {
      return currentKind;
    },
    flush() {
      if (currentKind !== "thinking") {
        return "";
      }

      currentKind = "response";
      return THINK_FORCED_CLOSE_TAG;
    },
    push(delta) {
      if (!delta) {
        return "";
      }

      let prefix = "";
      if (delta.kind !== currentKind) {
        if (currentKind === "thinking") {
          prefix += THINK_CLOSE_TAG;
        }
        if (delta.kind === "thinking") {
          prefix += THINK_OPEN_TAG;
        }
        currentKind = delta.kind;
      }

      if (!delta.text) {
        return prefix;
      }

      return prefix + delta.text;
    }
  };
}

/* Stream consumer.
 * The thinking-tagger emits text with `<think>`/`</think>` markers around
 * reasoning content.  All filtering of tool-call XML and partial-prefix
 * holding is delegated to the sieve via `onText(text, kind)`.  Callers
 * receive structured events from the sieve and decide how to write SSE. */
async function consumeTaggedStream(stream, onText) {
  if (!stream) {
    log.warn("runner", "No stream body received from upstream");
    return;
  }

  const decoder = new TextDecoder();
  const deltaDecoder = createDeepseekDeltaDecoder();
  const tagger = createThinkingTagger();
  let chunkCount = 0;

  const parser = createSseParser(({ data }) => {
    chunkCount++;
    const delta = deltaDecoder.consume(data);
    if (!delta) return;
    const text = tagger.push(delta);
    if (!text) return;

    onText(text, tagger.kind);
  });

  for await (const chunk of stream) {
    parser.push(decoder.decode(chunk, { stream: true }));
  }

  parser.flush();
  const suffix = tagger.flush();
  if (suffix) {
    onText(suffix, tagger.kind);
  }

  log.debug("runner", `Stream consumed: ${chunkCount} SSE events`);
}

async function withCompletionSession({ account, deleteAfterFinish, onComplete }) {
  const sessionId = await createChatSession(account);

  try {
    return await onComplete(sessionId);
  } finally {
    if (deleteAfterFinish) {
      await deleteChatSession(account, sessionId);
    }
  }
}

export async function collectCompletionContent({ account, deleteAfterFinish = false, requestOptions }) {
  return withCompletionSession({
    account,
    deleteAfterFinish,
    onComplete: async (sessionId) => {
      const { response } = await startCompletion({ account, requestOptions, sessionId });
      let content = "";

      await consumeTaggedStream(response.body, (text) => {
        content += text;
      });

      if (!content.length) {
        log.warn("runner", `Empty response from DeepSeek (model=${requestOptions.model.id}, prompt=${requestOptions.prompt.length} chars)`);
      }

      return { content };
    }
  });
}

export async function streamCompletionContent({ account, deleteAfterFinish = false, onText, requestOptions }) {
  return withCompletionSession({
    account,
    deleteAfterFinish,
    onComplete: async (sessionId) => {
      const { response } = await startCompletion({ account, requestOptions, sessionId });
      let hasContent = false;
      await consumeTaggedStream(response.body, (text, kind) => {
        hasContent = true;
        onText(text, kind);
      });

      if (!hasContent) {
        log.warn("runner", `Empty stream from DeepSeek (model=${requestOptions.model.id}, prompt=${requestOptions.prompt.length} chars)`);
      }
    }
  });
}

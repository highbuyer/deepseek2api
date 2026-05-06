// Bridge between deepseek2api API routes and Windsurf/Cursor language server.
// Wraps WindsurfClient + pool checkout/checkin into the standard
// collect / stream interface used by the route layer.

import { randomUUID } from 'node:crypto';
import { WindsurfClient, contentToString } from '../windsurf/client.js';
import { log } from '../utils/log.js';
import { config } from '../windsurf/config.js';
import { checkout, checkin, fingerprintBefore, fingerprintAfter } from '../windsurf/pool.js';
import { MODELS } from '../windsurf/models.js';

/* ── helpers ── */

function createMessageId() {
  return 'msg_' + randomUUID();
}

function resolveModel(body) {
  const modelId = body?.model || config.defaultModel || 'claude-4.5-sonnet-thinking';
  for (const [key, entry] of MODELS.entries()) {
    if (entry.label === modelId || entry.uid === modelId || key === modelId) {
      return { key, entry };
    }
  }
  const entry = MODELS.get(modelId);
  if (entry) return { key: modelId, entry };
  const first = MODELS.entries().next().value;
  if (first) return { key: first[0], entry: first[1] };
  throw new Error('No Windsurf models available');
}

function convertMessages(messages) {
  return (messages || []).map(m => ({
    role: m.role,
    content: typeof m.content === 'string' ? m.content : contentToString(m.content)
  }));
}

/* ── non-streaming (collect) ── */

export async function collectWindsurfMessage({
  account,
  body,
  deleteAfterFinish = false
}) {
  const model = resolveModel(body);
  const messages = convertMessages(body?.messages ?? []);
  const callerKey = account?.id || account?.apiKey || 'default';
  const fp = fingerprintBefore(messages, model.key, callerKey);

  log.info('bridge', `[windsurf-collect] model=${model.key} accountId=${callerKey}`);

  const entry = checkout(fp, callerKey);
  if (!entry) {
    throw new Error('No available Windsurf LS instance');
  }

  const client = new WindsurfClient(entry.apiKey, entry.port, entry.csrfToken);
  let resultText = '';
  let error = null;

  try {
    resultText = await new Promise((resolve, reject) => {
      client.cascadeChat(messages, model.entry.enum, model.entry.uid, {
        displayModel: body?.model,
        onChunk: () => {},
        onEnd: (text) => resolve(text),
        onError: (err) => reject(err)
      });
    });
  } catch (e) {
    error = e;
    log.error('bridge', `[windsurf-collect] error: ${e.message}`);
  } finally {
    const fpAfter = error ? fp : fingerprintAfter(messages, model.key, callerKey);
    checkin(fpAfter, entry, callerKey, deleteAfterFinish ? 0 : undefined);
  }

  if (error) throw error;

  return {
    id: createMessageId(),
    model: model.key,
    content: [{ type: 'text', text: resultText }],
    stop_reason: 'end_turn',
    usage: { input_tokens: 0, output_tokens: 0 }
  };
}

/* ── streaming (SSE) ── */

export async function streamWindsurfMessage({
  account,
  body,
  response,
  deleteAfterFinish = false
}) {
  const model = resolveModel(body);
  const messages = convertMessages(body?.messages ?? []);
  const callerKey = account?.id || account?.apiKey || 'default';
  const fp = fingerprintBefore(messages, model.key, callerKey);
  const msgId = createMessageId();

  log.info('bridge', `[windsurf-stream] model=${model.key} accountId=${callerKey}`);

  const entry = checkout(fp, callerKey);
  if (!entry) {
    response.writeHead(503, { 'Content-Type': 'application/json' });
    response.end(JSON.stringify({ error: 'No available Windsurf LS instance' }));
    return;
  }

  const client = new WindsurfClient(entry.apiKey, entry.port, entry.csrfToken);
  let finished = false;

  const cleanup = (err) => {
    if (finished) return;
    finished = true;
    const fpAfter = err ? fp : fingerprintAfter(messages, model.key, callerKey);
    checkin(fpAfter, entry, callerKey, deleteAfterFinish ? 0 : undefined);
  };

  response.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive'
  });

  response.write(`event: message_start\ndata: ${JSON.stringify({
    type: 'message_start',
    message: { id: msgId, model: model.key, content: [] }
  })}\n\n`);

  try {
    await new Promise((resolve, reject) => {
      client.cascadeChat(messages, model.entry.enum, model.entry.uid, {
        displayModel: body?.model,
        onChunk: (chunk) => {
          if (chunk) {
            response.write(`event: content_block_delta\ndata: ${JSON.stringify({
              type: 'content_block_delta',
              index: 0,
              delta: { type: 'text_delta', text: chunk }
            })}\n\n`);
          }
        },
        onEnd: (text) => {
          response.write(`event: message_delta\ndata: ${JSON.stringify({
            type: 'message_delta',
            delta: { stop_reason: 'end_turn', stop_sequence: null },
            usage: { output_tokens: 0 }
          })}\n\n`);
          response.write(`event: message_stop\ndata: ${JSON.stringify({
            type: 'message_stop'
          })}\n\n`);
          cleanup();
          resolve();
        },
        onError: (err) => {
          response.write(`event: error\ndata: ${JSON.stringify({
            type: 'error',
            error: { type: 'api_error', message: err.message }
          })}\n\n`);
          cleanup(err);
          reject(err);
        }
      });
    });
  } catch (e) {
    log.error('bridge', `[windsurf-stream] error: ${e.message}`);
  } finally {
    response.end();
  }
}

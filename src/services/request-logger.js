import { appendFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";

const LOG_DIR = "data/logs";

function ensureLogDir() {
  try { mkdirSync(LOG_DIR, { recursive: true }); } catch { /* ignore */ }
}

function formatDate() {
  return new Date().toISOString().slice(0, 10);
}

function appendJsonl(filename, obj) {
  ensureLogDir();
  try {
    appendFileSync(filename, JSON.stringify(obj) + "\n");
  } catch {
    // best-effort; don't block the request
  }
}

export function createRequestLogger({ requestId, model, toolNames = [] }) {
  const logFile = join(LOG_DIR, `requests-${formatDate()}.jsonl`);
  const phases = [];
  let currentPhase = null;
  let currentStart = 0;
  const startTime = Date.now();
  let ttft = null;
  let rawResponseLen = 0;
  let toolCallCount = 0;

  return Object.freeze({
    startPhase(phase, label) {
      if (currentPhase) this.endPhase();
      currentStart = performance.now();
      currentPhase = { phase, label };
    },

    endPhase() {
      if (!currentPhase) return;
      currentPhase.durationMs = Math.round(performance.now() - currentStart);
      phases.push(currentPhase);
      currentPhase = null;
    },

    recordTTFT() {
      if (ttft === null) {
        ttft = Date.now() - startTime;
      }
    },

    recordToolCalls(count) {
      toolCallCount += count;
    },

    recordResponseLength(length) {
      rawResponseLen = length;
    },

    complete({ status = "success", promptChars = 0, stopReason = "stop" } = {}) {
      if (currentPhase) this.endPhase();
      const entry = {
        ts: new Date().toISOString(),
        id: requestId,
        model,
        toolNames,
        promptChars,
        ttft,
        totalMs: Date.now() - startTime,
        responseChars: rawResponseLen,
        toolCallCount,
        stopReason,
        status,
        phases: phases.map(p => ({ phase: p.phase, label: p.label, ms: p.durationMs }))
      };
      appendJsonl(logFile, entry);
    },

    fail(error) {
      this.complete({ status: `error: ${error?.message ?? "unknown"}` });
    }
  });
}

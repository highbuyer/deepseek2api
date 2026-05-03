import { appendFileSync, mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { config } from "../config.js";

const LEVELS = Object.freeze({ debug: 0, info: 1, warn: 2, error: 3 });

function timestamp() {
  return new Date().toISOString();
}

const logDir = join(process.cwd(), "data");
const logFile = config.debug ? join(logDir, "debug.log") : null;
if (logFile) {
  mkdirSync(logDir, { recursive: true });
  writeFileSync(logFile, `=== deepseek2api started at ${timestamp()} ===\n`);
}

function formatLine(level, tag, args) {
  const prefix = `[${timestamp()}] [${level.toUpperCase()}] [${tag}]`;
  const body = args.map(a => typeof a === "string" ? a : JSON.stringify(a)).join(" ");
  return `${prefix} ${body}\n`;
}

function emit(level, tag, ...args) {
  const silenced = !config.debug && level === "debug";
  if (!silenced) {
    const logger = level === "error" ? console.error : level === "warn" ? console.warn : console.log;
    logger(`[${timestamp()}] [${level.toUpperCase()}] [${tag}]`, ...args);
  }

  if (logFile) {
    try {
      appendFileSync(logFile, formatLine(level, tag, args));
    } catch { /* best effort */ }
  }
}

export const log = Object.freeze({
  debug: (tag, ...args) => emit("debug", tag, ...args),
  info: (tag, ...args) => emit("info", tag, ...args),
  warn: (tag, ...args) => emit("warn", tag, ...args),
  error: (tag, ...args) => emit("error", tag, ...args)
});

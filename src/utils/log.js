import { config } from "../config.js";

const LEVELS = Object.freeze({ debug: 0, info: 1, warn: 2, error: 3 });

function timestamp() {
  return new Date().toISOString();
}

function emit(level, tag, ...args) {
  if (!config.debug && level === "debug") {
    return;
  }

  const prefix = `[${timestamp()}] [${level.toUpperCase()}] [${tag}]`;
  const logger = level === "error" ? console.error : level === "warn" ? console.warn : console.log;
  logger(prefix, ...args);
}

export const log = Object.freeze({
  debug: (tag, ...args) => emit("debug", tag, ...args),
  info: (tag, ...args) => emit("info", tag, ...args),
  warn: (tag, ...args) => emit("warn", tag, ...args),
  error: (tag, ...args) => emit("error", tag, ...args)
});

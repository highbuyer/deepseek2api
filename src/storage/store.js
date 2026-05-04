import { existsSync, readFileSync } from "node:fs";
import { writeFile } from "node:fs/promises";

import { config } from "../config.js";

function defaultState() {
  return {
    accounts: [],
    apiKeys: [],
    incognito: {
      globalEnabled: false,
      owners: {}
    },
    invites: [],
    registration: {
      inviteRequired: false
    },
    sessions: [],
    users: []
  };
}

function normalizeIncognito(value) {
  const owners = value?.owners;

  return {
    globalEnabled: Boolean(value?.globalEnabled),
    owners: owners && typeof owners === "object" ? owners : {}
  };
}

function normalizeInvites(value) {
  return Array.isArray(value) ? value : [];
}

function normalizeRegistration(value) {
  return {
    inviteRequired: Boolean(value?.inviteRequired)
  };
}

function normalizeUsers(value) {
  const normalizeLimit = (limit) => {
    const parsed = Number(limit);
    return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
  };

  return Array.isArray(value) ? value.map((user) => ({
    ...user,
    disabled: Boolean(user?.disabled),
    requestLimits: {
      maxConcurrency: normalizeLimit(user?.requestLimits?.maxConcurrency),
      maxRequestsPerMinute: normalizeLimit(user?.requestLimits?.maxRequestsPerMinute)
    }
  })) : [];
}

function normalizeApiKeys(value) {
  if (!Array.isArray(value)) {
    return [];
  }

  return value.map((record) => ({
    ...record,
    toolCallsEnabled: Boolean(record?.toolCallsEnabled)
  }));
}

function normalizeState(value) {
  return {
    accounts: Array.isArray(value?.accounts) ? value.accounts : [],
    apiKeys: normalizeApiKeys(value?.apiKeys),
    incognito: normalizeIncognito(value?.incognito),
    invites: normalizeInvites(value?.invites),
    registration: normalizeRegistration(value?.registration),
    sessions: Array.isArray(value?.sessions) ? value.sessions : [],
    users: normalizeUsers(value?.users)
  };
}

// ── In-memory cache + async disk flush ──
// updateStore writes to memory immediately (sync) and marks dirty.
// flushStore writes to disk asynchronously (called periodically).
// This eliminates synchronous writeFileSync blocking the event loop
// while keeping the updateStore API backward-compatible.

let _cache = null;
let _dirty = false;
let _writing = false;

function loadCache() {
  if (_cache) return _cache;
  if (!existsSync(config.dataFile)) {
    _cache = defaultState();
    _dirty = true;
    return _cache;
  }
  const raw = readFileSync(config.dataFile, "utf8");
  _cache = normalizeState(JSON.parse(raw));
  return _cache;
}

export function readStore() {
  return loadCache();
}

export function writeStore(state) {
  _cache = normalizeState(state ?? _cache);
  _dirty = true;
}

export function updateStore(updater) {
  loadCache();
  _cache = updater(_cache);
  _dirty = true;
  return _cache;
}

export async function flushStore() {
  if (!_dirty || _writing) return;
  _writing = true;
  const data = JSON.stringify(normalizeState(_cache), null, 2);
  try {
    await writeFile(config.dataFile, data);
    _dirty = false;
  } catch {
    // best-effort; next interval will retry
  } finally {
    _writing = false;
  }
}

import { getLocalUserFromOwnerId } from "./user-service.js";
import { createTaggedError } from "./api-error.js";

const RATE_WINDOW_MS = 60_000;
const activeRequests = new Map();
const requestHistory = new Map();

function pruneHistory(ownerId, now) {
  const current = requestHistory.get(ownerId) ?? [];
  const next = current.filter((timestamp) => now - timestamp < RATE_WINDOW_MS);
  requestHistory.set(ownerId, next);
  return next;
}

function releaseConcurrency(ownerId) {
  const current = activeRequests.get(ownerId) ?? 0;
  const next = Math.max(0, current - 1);

  if (next === 0) {
    activeRequests.delete(ownerId);
    return;
  }

  activeRequests.set(ownerId, next);
}

function assertUserAvailable(ownerId) {
  const user = getLocalUserFromOwnerId(ownerId);
  if (!user) {
    return null;
  }

  if (user.disabled) {
    throw createTaggedError("User is disabled", "USER_DISABLED");
  }

  return user;
}

function assertRequestLimits(ownerId, requestLimits) {
  const now = Date.now();
  const recentRequests = pruneHistory(ownerId, now);
  const maxConcurrency = requestLimits.maxConcurrency;
  const maxRequestsPerMinute = requestLimits.maxRequestsPerMinute;

  if (maxConcurrency !== null && (activeRequests.get(ownerId) ?? 0) >= maxConcurrency) {
    throw createTaggedError(`Concurrent request limit exceeded (${maxConcurrency})`, "REQUEST_LIMIT");
  }

  if (maxRequestsPerMinute !== null && recentRequests.length >= maxRequestsPerMinute) {
    throw createTaggedError(`Rate limit exceeded (${maxRequestsPerMinute}/minute)`, "REQUEST_LIMIT");
  }

  requestHistory.set(ownerId, [...recentRequests, now]);
}

export async function withOwnerRequestLimit(ownerId, action) {
  const user = assertUserAvailable(ownerId);
  if (!user) {
    return action();
  }

  const requestLimits = user.requestLimits ?? {};
  assertRequestLimits(ownerId, requestLimits);
  activeRequests.set(ownerId, (activeRequests.get(ownerId) ?? 0) + 1);

  try {
    return await action();
  } finally {
    releaseConcurrency(ownerId);
  }
}

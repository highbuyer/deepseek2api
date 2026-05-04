/**
 * Unified API error class — replaces scattered Error + .code/.statusCode patterns.
 */

export class ApiError extends Error {
  constructor(message, { statusCode = 500, code = "" } = {}) {
    super(message);
    this.name = "ApiError";
    this.statusCode = statusCode;
    this.code = code;
  }
}

/**
 * Create a tagged error for rate-limit / user-disabled rejections.
 * These are discriminated by .code at the route boundary.
 */
export function createTaggedError(message, code) {
  return new ApiError(message, { code });
}

/**
 * Create an OpenAI-compatible error with HTTP status.
 */
export function createOpenAiError(statusCode, message, code = "") {
  return new ApiError(message, { statusCode, code });
}

/**
 * Map .code to HTTP status for limit-related errors.
 * Shared between openai-routes and proxy-routes.
 */
export function resolveLimitStatus(error) {
  return error.code === "USER_DISABLED" ? 403 : 429;
}

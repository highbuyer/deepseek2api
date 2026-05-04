import { ApiError } from "./api-error.js";

export function createOpenAiError(statusCode, message, code = "") {
  return new ApiError(message, { statusCode, code });
}

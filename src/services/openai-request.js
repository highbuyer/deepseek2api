import { listEmbeddingModels } from "./openai-embeddings.js";

const DEFAULT_OPENAI_MODEL = "deepseek-chat-fast";
const SEARCH_MODEL_SUFFIX = "-search";

// Map OpenAI model names (sent by Warp client) to DeepSeek equivalents.
const MODEL_ALIASES = Object.freeze({
  "gpt-5.5": "deepseek-reasoner-expert",
  "gpt-5": "deepseek-reasoner-expert",
  "gpt-4o": "deepseek-chat-expert",
  "gpt-4.1": "deepseek-chat-expert",
  "gpt-4": "deepseek-chat-fast",
  "gpt-4-turbo": "deepseek-chat-fast",
  "gpt-3.5-turbo": "deepseek-chat-fast",
  "gpt-4o-mini": "deepseek-chat-fast",
  o1: "deepseek-chat-fast",
  o3: "deepseek-chat-fast",
  "o4-mini": "deepseek-reasoner-expert",
  "o3-mini": "deepseek-reasoner-expert",
  "o1-mini": "deepseek-reasoner-expert"
});

const BASE_OPENAI_MODELS = Object.freeze([
  Object.freeze({ id: "deepseek-chat-fast", modelType: "default", thinkingEnabled: false }),
  Object.freeze({ id: "deepseek-reasoner-fast", modelType: "default", thinkingEnabled: true }),
  Object.freeze({ id: "deepseek-chat-expert", modelType: "expert", thinkingEnabled: false }),
  Object.freeze({ id: "deepseek-reasoner-expert", modelType: "expert", thinkingEnabled: true })
]);

function createModelVariant(baseModel, searchEnabled) {
  return Object.freeze({
    ...baseModel,
    id: searchEnabled ? `${baseModel.id}${SEARCH_MODEL_SUFFIX}` : baseModel.id,
    searchEnabled
  });
}

const OPENAI_MODELS = Object.freeze(
  BASE_OPENAI_MODELS.flatMap((model) => [
    createModelVariant(model, false),
    createModelVariant(model, true)
  ])
);

const OPENAI_MODEL_MAP = Object.freeze(
  Object.fromEntries(OPENAI_MODELS.map((model) => [model.id, model]))
);

function createBadRequestError(message) {
  const error = new Error(message);
  error.statusCode = 400;
  return error;
}

export function listOpenAiModels() {
  const chatIds = new Set(OPENAI_MODELS.map(({ id }) => id));
  return [
    ...OPENAI_MODELS.map(({ id }) => ({ id, object: "model" })),
    ...listEmbeddingModels().filter((m) => !chatIds.has(m.id))
  ];
}

export function resolveOpenAiModel(model) {
  const modelId = model ?? DEFAULT_OPENAI_MODEL;

  // Check alias map first (e.g. "gpt-5.5" → "deepseek-chat-expert").
  const aliasedId = MODEL_ALIASES[modelId] ?? modelId;

  const resolvedModel = OPENAI_MODEL_MAP[aliasedId];

  if (!resolvedModel) {
    throw createBadRequestError(`Unsupported model: ${modelId}`);
  }

  return resolvedModel;
}

export function assertNoLegacySearchOptions(body) {
  if (Object.hasOwn(body ?? {}, "web_search_options")) {
    throw createBadRequestError(
      "Search is now controlled by model suffix '-search', not web_search_options"
    );
  }
}

export { DEFAULT_OPENAI_MODEL };

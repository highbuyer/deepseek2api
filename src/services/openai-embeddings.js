import { log } from "../utils/log.js";

const EMBEDDING_MODEL = "Xenova/all-MiniLM-L6-v2";
const EMBEDDING_DIMS = 384;

let _pipeline = null;
let _pipelineLoading = null;

async function loadPipeline() {
  const { pipeline } = await import("@xenova/transformers");
  log.info("embeddings", `Downloading local model: ${EMBEDDING_MODEL} ...`);
  const pipe = await pipeline("feature-extraction", EMBEDDING_MODEL, { quantized: true });
  log.info("embeddings", `Model ready: ${EMBEDDING_MODEL}`);
  return pipe;
}

export async function preloadEmbeddingModel() {
  if (!_pipeline) {
    _pipeline = await loadPipeline();
  }
}

async function getPipeline() {
  if (_pipeline) return _pipeline;
  if (_pipelineLoading) return _pipelineLoading;
  _pipelineLoading = loadPipeline().then((pipe) => { _pipeline = pipe; return pipe; });
  return _pipelineLoading;
}

export function listEmbeddingModels() {
  return [
    { id: "text-embedding-3-small", object: "model" },
    { id: "text-embedding-3-large", object: "model" },
    { id: "text-embedding-ada-002", object: "model" },
    { id: EMBEDDING_MODEL, object: "model" }
  ];
}

export async function createEmbeddings({ body }) {
  const input = body?.input;

  if (!input || (Array.isArray(input) && !input.length)) {
    const error = new Error("Missing or empty 'input' field");
    error.statusCode = 400;
    throw error;
  }

  const inputs = Array.isArray(input) ? input : [input];

  const extractor = await getPipeline();
  const output = await extractor(inputs, { pooling: "mean", normalize: true });

  const tensor = output.tolist ? output : output;
  const data = Array.from(tensor.data ?? tensor);
  const dims = tensor.dims ?? [data.length];
  const vecSize = dims.length > 1 ? dims[1] : dims[0];
  const count = dims.length > 1 ? dims[0] : 1;

  const vectors = [];
  for (let i = 0; i < count; i++) {
    vectors.push(data.slice(i * vecSize, (i + 1) * vecSize));
  }

  return {
    object: "list",
    data: vectors.map((embedding, index) => ({
      object: "embedding",
      index,
      embedding
    })),
    model: EMBEDDING_MODEL,
    usage: { prompt_tokens: inputs.join(" ").length, total_tokens: inputs.join(" ").length }
  };
}

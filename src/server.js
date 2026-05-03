import { readFileSync } from "node:fs";
import { createServer } from "node:http";
import { createServer as createHttpsServer } from "node:https";
import { setGlobalDispatcher, ProxyAgent } from "undici";

import { config } from "./config.js";
import { preloadEmbeddingModel } from "./services/openai-embeddings.js";

// Node.js fetch doesn't respect system proxy — wire it up manually
const proxyUrl = process.env.HTTPS_PROXY || process.env.https_proxy || process.env.HTTP_PROXY || process.env.http_proxy;
if (proxyUrl) {
  setGlobalDispatcher(new ProxyAgent(proxyUrl));
  console.log(`Proxy: ${proxyUrl}`);
}
import { handleApiRequest } from "./routes/api-routes.js";
import { handleOpenAiRequest } from "./routes/openai-routes.js";
import { handleProxyRequest } from "./routes/proxy-routes.js";
import { parseCookies, sendError, serveStaticFile } from "./utils/http.js";

function createRequestHandler() {
  return async (request, response) => {
    const url = new URL(request.url ?? "/", `http://${request.headers.host ?? "localhost"}`);
    request.cookies = parseCookies(request);

    response.setHeader("access-control-allow-origin", "*");
    response.setHeader("access-control-allow-headers", "*");
    response.setHeader("access-control-allow-methods", "GET,POST,PUT,PATCH,DELETE,OPTIONS");
    response.setHeader("access-control-allow-private-network", "true");

    if (request.method === "OPTIONS") {
      response.writeHead(204);
      response.end();
      return;
    }

    try {
      if (url.pathname.startsWith("/api/")) {
        const handled = await handleApiRequest(request, response, url);
        if (!handled) {
          sendError(response, 404, "API route not found");
        }
        return;
      }

      if (url.pathname.startsWith("/proxy/")) {
        await handleProxyRequest(request, response, url, config.allowedProxyPaths);
        return;
      }

      if (url.pathname.startsWith("/v1/")) {
        const handled = await handleOpenAiRequest(request, response, url);
        if (!handled) {
          sendError(response, 404, "OpenAI route not found");
        }
        return;
      }

      if (!serveStaticFile(request, response, url.pathname)) {
        sendError(response, 404, "Page not found");
      }
    } catch (error) {
      if (response.headersSent || response.writableEnded) {
        response.destroy(error);
        return;
      }

      sendError(response, 500, error.message);
    }
  };
}

function startHttpServer(handler) {
  createServer(handler).listen(config.port, () => {
    console.log(`HTTP  server listening on http://127.0.0.1:${config.port}`);
  });
}

function startHttpsServer(handler) {
  const options = {
    key: readFileSync(config.httpsKeyFile),
    cert: readFileSync(config.httpsCertFile)
  };
  createHttpsServer(options, handler).listen(config.port, () => {
    console.log(`HTTPS server listening on https://127.0.0.1:${config.port}`);
  });
}

const handler = createRequestHandler();

preloadEmbeddingModel().catch((err) => {
  console.warn("Embedding model preload failed:", err.message);
});

if (config.httpsEnabled) {
  startHttpsServer(handler);
  // Also serve HTTP on another port for local clients that can't verify the cert
  const httpPort = Number(process.env.HTTP_PORT || 3001);
  createServer(handler).listen(httpPort, () => {
    console.log(`HTTP  server listening on http://127.0.0.1:${httpPort}`);
  });
} else {
  startHttpServer(handler);
}

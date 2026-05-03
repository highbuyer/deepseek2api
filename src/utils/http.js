import { createReadStream, statSync } from "node:fs";
import { extname, join, normalize } from "node:path";

import { config } from "../config.js";

const textEncoder = new TextEncoder();
const publicDirectory = join(process.cwd(), "public");
const HTML_CACHE_CONTROL = "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0";
const DYNAMIC_ASSET_CACHE_CONTROL = "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0";
const STATIC_CACHE_CONTROL = "public, max-age=0, must-revalidate";
const CDN_NO_STORE_CACHE_CONTROL = "no-store, no-cache, max-age=0, s-maxage=0, must-revalidate";
const NO_STORE_EXTENSIONS = new Set([".css", ".html", ".js"]);

const contentTypes = Object.freeze({
  ".css": "text/css; charset=utf-8",
  ".html": "text/html; charset=utf-8",
  ".ico": "image/x-icon",
  ".js": "application/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".svg": "image/svg+xml"
});

export function json(value) {
  return textEncoder.encode(JSON.stringify(value));
}

export function sendJson(response, statusCode, payload) {
  const body = json(payload);
  response.writeHead(statusCode, {
    "content-length": body.byteLength,
    "content-type": "application/json; charset=utf-8"
  });
  response.end(body);
}

export function sendError(response, statusCode, message) {
  sendJson(response, statusCode, { error: message });
}

export function parseCookies(request) {
  const cookieHeader = request.headers.cookie ?? "";
  const result = {};
  for (const entry of cookieHeader.split(";")) {
    const [name, ...rest] = entry.trim().split("=");
    if (name) {
      result[name] = decodeURIComponent(rest.join("="));
    }
  }
  return result;
}

export function setCookie(response, name, value, maxAgeSeconds) {
  response.setHeader(
    "set-cookie",
    `${name}=${encodeURIComponent(value)}; HttpOnly; Path=/; Max-Age=${maxAgeSeconds}; SameSite=Lax`
  );
}

export function clearCookie(response, name) {
  response.setHeader(
    "set-cookie",
    `${name}=; HttpOnly; Path=/; Max-Age=0; SameSite=Lax`
  );
}

export async function readRequestBody(request) {
  const chunks = [];
  let total = 0;

  for await (const chunk of request) {
    total += chunk.byteLength;
    if (total > config.requestBodyLimitBytes) {
      throw new Error("Request body too large");
    }
    chunks.push(chunk);
  }

  return Buffer.concat(chunks);
}

export function parseJsonBody(buffer) {
  if (!buffer.byteLength) {
    return null;
  }

  return JSON.parse(buffer.toString("utf8"));
}

function resolveStaticFilePath(pathname) {
  const relativePath = pathname === "/" ? "/index.html" : pathname;
  const safePath = normalize(relativePath).replace(/^(\.\.[/\\])+/, "");
  const filePath = join(publicDirectory, safePath);

  if (!filePath.startsWith(publicDirectory)) {
    return null;
  }

  try {
    return statSync(filePath).isFile() ? filePath : null;
  } catch {
    return null;
  }
}

function buildEntityTag(fileStat) {
  return `"${fileStat.size.toString(16)}-${Math.trunc(fileStat.mtimeMs).toString(16)}"`;
}

function matchesIfNoneMatch(request, entityTag) {
  const headerValue = request.headers["if-none-match"];
  if (!headerValue) {
    return false;
  }

  return headerValue.split(",").some((value) => {
    const candidate = value.trim();
    return candidate === "*" || candidate === entityTag;
  });
}

function isNotModified(request, fileStat, entityTag) {
  if (matchesIfNoneMatch(request, entityTag)) {
    return true;
  }

  const headerValue = request.headers["if-modified-since"];
  if (!headerValue) {
    return false;
  }

  const modifiedSinceMs = Date.parse(headerValue);
  if (Number.isNaN(modifiedSinceMs)) {
    return false;
  }

  return Math.trunc(fileStat.mtimeMs / 1000) * 1000 <= modifiedSinceMs;
}

function resolveCachePolicy(extension) {
  if (!NO_STORE_EXTENSIONS.has(extension)) {
    return {
      allowNotModified: true,
      browserCacheControl: STATIC_CACHE_CONTROL,
      cdnCacheControl: STATIC_CACHE_CONTROL
    };
  }

  return {
    allowNotModified: false,
    browserCacheControl: extension === ".html" ? HTML_CACHE_CONTROL : DYNAMIC_ASSET_CACHE_CONTROL,
    cdnCacheControl: CDN_NO_STORE_CACHE_CONTROL
  };
}

function buildStaticHeaders(filePath, fileStat) {
  const extension = extname(filePath);
  const isHtml = extension === ".html";
  const cachePolicy = resolveCachePolicy(extension);

  return {
    allowNotModified: cachePolicy.allowNotModified,
    headers: {
      "cache-control": cachePolicy.browserCacheControl,
      "cdn-cache-control": cachePolicy.cdnCacheControl,
      "cloudflare-cdn-cache-control": cachePolicy.cdnCacheControl,
      "surrogate-control": cachePolicy.cdnCacheControl,
      "content-type": contentTypes[extension] ?? "application/octet-stream",
      etag: buildEntityTag(fileStat),
      "last-modified": fileStat.mtime.toUTCString(),
      ...(isHtml ? { "clear-site-data": "\"cache\"", expires: "0", pragma: "no-cache" } : {})
    }
  };
}

export function serveStaticFile(request, response, pathname) {
  const filePath = resolveStaticFilePath(pathname);
  if (!filePath) {
    return false;
  }

  const fileStat = statSync(filePath);
  const { allowNotModified, headers } = buildStaticHeaders(filePath, fileStat);

  if (allowNotModified && isNotModified(request, fileStat, headers.etag)) {
    response.writeHead(304, headers);
    response.end();
    return true;
  }

  response.writeHead(200, {
    ...headers,
    "content-length": fileStat.size
  });
  createReadStream(filePath).pipe(response);
  return true;
}

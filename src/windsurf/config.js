// Adapter config for WindsurfAPI modules in deepseek2api.
// Mirrors the shape expected by auth.js, langserver.js, grpc.js, etc.
// Values come from environment variables with sensible defaults.

import { homedir, platform, arch } from 'os';
import { join } from 'path';

const isMac = platform() === 'darwin';
const isArm = arch() === 'arm64';

const lsBinaryPath = process.env.LS_BINARY_PATH || (
  isMac
    ? `${homedir()}/.windsurf/language_server_macos_${isArm ? 'arm' : 'x64'}`
    : '/opt/windsurf/language_server_linux_x64'
);

export const config = {
  port: parseInt(process.env.PORT || '3003', 10),
  host: process.env.HOST || process.env.BIND_HOST || '0.0.0.0',
  apiKey: process.env.API_KEY || '',
  dataDir: process.env.DATA_DIR || join(homedir(), '.deepseek2api'),
  sharedDataDir: process.env.SHARED_DATA_DIR || process.env.DATA_DIR || join(homedir(), '.deepseek2api'),

  codeiumAuthToken: process.env.CODEIUM_AUTH_TOKEN || '',
  codeiumApiKey: process.env.CODEIUM_API_KEY || '',
  codeiumEmail: process.env.CODEIUM_EMAIL || '',
  codeiumPassword: process.env.CODEIUM_PASSWORD || '',

  codeiumApiUrl: process.env.CODEIUM_API_URL || 'https://server.self-serve.windsurf.com',
  defaultModel: process.env.DEFAULT_MODEL || 'claude-4.5-sonnet-thinking',
  maxTokens: parseInt(process.env.MAX_TOKENS || '8192', 10),
  logLevel: process.env.LOG_LEVEL || 'info',

  lsBinaryPath,
  lsPort: parseInt(process.env.LS_PORT || '42100', 10),

  dashboardPassword: process.env.DASHBOARD_PASSWORD || '',

  allowPrivateProxyHosts: process.env.ALLOW_PRIVATE_PROXY_HOSTS === '1',
};

/**
 * Atomic JSON file writer.
 *
 * The naive `writeFileSync(path, JSON.stringify(...))` truncates the
 * target file and then writes — if the process gets killed (SIGTERM
 * during docker stop, SIGKILL on OOM, panic-style crash) between the
 * truncate and the final write, the file is left empty or partial.
 * Next start, JSON.parse fails, the load() handler logs a warning and
 * silently falls back to defaults — user loses every persisted
 * setting, model-access list, proxy config, etc.
 *
 * Pattern: write the new contents to a sibling `${target}.tmp` first,
 * then `rename(2)` it onto the target. rename is atomic on POSIX and
 * replaces an existing target on Windows (per Node's documented
 * fs.renameSync behavior). A crash between writeFileSync(tmp) and
 * renameSync leaves the target intact; a crash after renameSync
 * leaves the new contents in place. Either way, no truncated JSON.
 *
 * Tmp file gets unlinked on write failure so repeated failures don't
 * leak garbage in DATA_DIR.
 *
 * Used by every dashboard config persister: model-access.json,
 * proxy.json, stats.json, runtime-config.json. accounts.json already
 * uses the same pattern hand-rolled in src/auth.js (kept inline there
 * because it has its own coalescing/_saveInFlight machinery).
 */

import { writeFileSync, renameSync, unlinkSync } from 'node:fs';

export function writeJsonAtomic(targetPath, value, { spaces = 2 } = {}) {
  const tmp = `${targetPath}.tmp`;
  try {
    writeFileSync(tmp, JSON.stringify(value, null, spaces));
    renameSync(tmp, targetPath);
  } catch (err) {
    try { unlinkSync(tmp); } catch {}
    throw err;
  }
}

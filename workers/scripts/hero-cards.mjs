// Runs scripts/hero-cards.ts: bundled for Node with esbuild (it comes with Wrangler), so the card
// code needs no TypeScript runtime. `node scripts/hero-cards.mjs [--check]`.
import { build } from "esbuild";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const workers = join(dirname(fileURLToPath(import.meta.url)), "..");
const dir = mkdtempSync(join(tmpdir(), "hero-cards-"));
const outfile = join(dir, "hero-cards.mjs");
try {
	await build({ entryPoints: [join(workers, "scripts", "hero-cards.ts")], bundle: true, platform: "node", format: "esm", target: "node22", outfile, logLevel: "warning" });
	process.env.WORKERS_DIR = workers;
	await import(pathToFileURL(outfile).href);
} finally {
	rmSync(dir, { recursive: true, force: true });
}

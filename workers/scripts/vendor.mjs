// Copies the renderer and fonts the card function needs out of node_modules into functions/_card,
// unmodified, so C's deploy only copies files (no install step). Run after `npm ci`:
//   node scripts/vendor.mjs
// resvg-wasm is MPL-2.0 and ships no licence file, so the MPL-2.0 text comes from lightningcss
// (the same licence, in web/node_modules). The fonts are OFL-1.1; each family keeps its own
// licence file. Fonts are renamed .ttf to .bin so Wrangler bundles them as binary data modules.
import { copyFileSync, mkdirSync, readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const nm = join(root, "node_modules");
const out = join(root, "functions", "_card");

const resvg = join(nm, "@resvg", "resvg-wasm");
const version = JSON.parse(readFileSync(join(resvg, "package.json"), "utf8")).version;
if (version !== "2.6.2") throw new Error(`expected @resvg/resvg-wasm 2.6.2, found ${version}`);
mkdirSync(join(out, "vendor", "resvg"), { recursive: true });
copyFileSync(join(resvg, "index.mjs"), join(out, "vendor", "resvg", "index.mjs"));
copyFileSync(join(resvg, "index.d.ts"), join(out, "vendor", "resvg", "index.d.mts"));
copyFileSync(join(resvg, "index_bg.wasm"), join(out, "vendor", "resvg", "index_bg.wasm"));
copyFileSync(join(root, "..", "web", "node_modules", "lightningcss", "LICENSE"), join(out, "vendor", "resvg", "LICENSE"));

const fonts = [
	["instrument-sans", "400Regular/InstrumentSans_400Regular.ttf", "InstrumentSans-400.bin"],
	["instrument-sans", "600SemiBold/InstrumentSans_600SemiBold.ttf", "InstrumentSans-600.bin"],
	["instrument-serif", "400Regular/InstrumentSerif_400Regular.ttf", "InstrumentSerif-400.bin"],
	["jetbrains-mono", "500Medium/JetBrainsMono_500Medium.ttf", "JetBrainsMono-500.bin"],
];
mkdirSync(join(out, "fonts"), { recursive: true });
for (const [pkg, file, name] of fonts) {
	const dir = join(nm, "@expo-google-fonts", pkg);
	copyFileSync(join(dir, file), join(out, "fonts", name));
	copyFileSync(join(dir, "LICENSE_FONT"), join(out, "fonts", `OFL-${pkg}.txt`));
}
console.log(`vendored resvg-wasm ${version} and ${fonts.length} fonts into functions/_card`);

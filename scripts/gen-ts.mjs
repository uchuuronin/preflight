// Generates schema/ts/contract.ts from schema/contract.schema.json.
// Run through `uv run python -m preflight.schema.generate`, not directly.
import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { compile } from "json-schema-to-typescript";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const check = process.argv.includes("--check");
const schema = JSON.parse(readFileSync(join(root, "schema/contract.schema.json"), "utf8"));

// Pydantic titles every field; json-schema-to-typescript turns each title into a
// named alias (Lens1, Kind2, ...). Keep titles only on the $defs models themselves.
function stripTitles(node, keep) {
	if (Array.isArray(node)) return node.forEach((n) => stripTitles(n, false));
	if (node === null || typeof node !== "object") return;
	if (!keep) delete node.title;
	for (const [k, v] of Object.entries(node)) {
		if (k === "$defs") for (const def of Object.values(v)) stripTitles(def, true);
		else if (k === "properties") for (const p of Object.values(v)) stripTitles(p, false);
		else stripTitles(v, false);
	}
}
stripTitles(schema, true);

const banner = `/* Generated from python/preflight/schema by \`uv run python -m preflight.schema.generate\`. Do not edit. */`;
let ts = await compile(schema, "PreflightContract", {
	bannerComment: banner,
	additionalProperties: false,
	unreachableDefinitions: true,
	strictIndexSignatures: true,
	style: { useTabs: true, semi: true, singleQuote: false, printWidth: 110 },
});
ts += `\nexport const SCHEMA_VERSION = ${JSON.stringify(schema.$id.split("/").at(-2))};\n`;

const out = join(root, "schema/ts/contract.ts");
if (check) {
	if (!existsSync(out) || readFileSync(out, "utf8") !== ts) {
		console.error("schema/ts/contract.ts is stale");
		process.exit(1);
	}
} else {
	writeFileSync(out, ts);
}

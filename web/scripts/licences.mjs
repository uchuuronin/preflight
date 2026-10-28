// Fail on copyleft dependencies (gap 22): GPL, AGPL or SSPL anywhere in web/node_modules.
// LGPL passes; an "MIT OR GPL-3.0" choice passes. Run from web/: node scripts/licences.mjs
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const BAD = /(^|[^L])\b(A?GPL|SSPL)\b/i;
const found = [];
const seen = new Set();

function licenceOf(pkg) {
	const l = pkg.license ?? pkg.licenses;
	if (!l) return "";
	if (typeof l === "string") return l;
	if (Array.isArray(l)) return l.map((x) => (typeof x === "string" ? x : x.type)).join(" OR ");
	return l.type ?? "";
}

function walk(dir) {
	if (!existsSync(dir)) return;
	for (const name of readdirSync(dir)) {
		if (name.startsWith(".")) continue;
		const path = join(dir, name);
		if (name.startsWith("@")) {
			walk(path);
			continue;
		}
		const file = join(path, "package.json");
		if (existsSync(file)) {
			const pkg = JSON.parse(readFileSync(file, "utf8"));
			const key = `${pkg.name}@${pkg.version}`;
			if (!seen.has(key)) {
				seen.add(key);
				const l = licenceOf(pkg);
				const options = l.replace(/[()]/g, "").split(/\s+OR\s+/i);
				if (options.length && options.every((o) => BAD.test(o) && !/LGPL/i.test(o))) found.push(`${key}: ${l}`);
			}
		}
		walk(join(path, "node_modules"));
	}
}

walk("node_modules");
for (const f of found) console.log(`::error title=copyleft dependency::${f}`);
console.log(`${seen.size} packages checked, ${found.length} copyleft`);
process.exit(found.length ? 1 : 0);

// Fail on copyleft dependencies (gap 22): GPL, AGPL or SSPL anywhere in ./node_modules.
// LGPL passes; an "MIT OR GPL-3.0" choice passes. A package that names no licence at all fails too,
// unless it's listed in NO_LICENCE_OK with the reason a person checked (D, Nov 3: it passed silently).
// Run from any package folder: node scripts/licences.mjs (web/), node ../web/scripts/licences.mjs
// (workers/), node web/scripts/licences.mjs (the root).
import { readdirSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

// No \b after the name: "GPLv3" is GPL too (D, Nov 3).
const BAD = /(^|[^L])\b(A?GPL|SSPL)/i;
// Packages that name no licence, each checked by hand: name -> why it's fine.
const NO_LICENCE_OK = {};
const unnamed = [];
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
				if (!l.trim() && !(pkg.name in NO_LICENCE_OK) && !existsSync(join(path, "LICENSE")) && !existsSync(join(path, "LICENSE.md")) && !existsSync(join(path, "LICENCE"))) unnamed.push(key);
				const options = l.replace(/[()]/g, "").split(/\s+OR\s+/i);
				if (options.length && options.every((o) => BAD.test(o) && !/LGPL/i.test(o))) found.push(`${key}: ${l}`);
			}
		}
		walk(join(path, "node_modules"));
	}
}

walk("node_modules");
for (const f of found) console.log(`::error title=copyleft dependency::${f}`);
for (const u of unnamed) console.log(`::error title=no licence named::${u} (check it by hand, then list it in NO_LICENCE_OK)`);
console.log(`${seen.size} packages checked, ${found.length} copyleft, ${unnamed.length} with no licence named`);
process.exit(found.length || unnamed.length ? 1 : 0);

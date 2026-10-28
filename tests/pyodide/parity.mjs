// Run B's engine in Pyodide on the golden records and print the results as JSON.
// Exit 3 when Pyodide's packages can't be fetched (no route to its CDN), so the test can skip honestly.
//
//	cd tests/pyodide && npm ci && node parity.mjs > pyodide.json
import path from "node:path";
import { fileURLToPath } from "node:url";
import { loadPyodide } from "pyodide";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../..");
const py = await loadPyodide();
const errors = [];
try {
	await py.loadPackage(["numpy", "pydantic"], { errorCallback: (msg) => errors.push(msg), messageCallback: () => {} });
} catch (err) {
	errors.push(String(err));
}
const missing = ["numpy", "pydantic"].filter((name) => !(name in py.loadedPackages));
if (missing.length) {
	// loadPackage logs a failed fetch instead of throwing: check what actually loaded
	console.error(`packages-unreachable: ${missing.join(", ")}: ${errors[0] ?? "not loaded"}`);
	process.exit(3);
}
py.mountNodeFS("/repo", root);
const out = py.runPython(`
import json, sys
sys.path[:0] = ["/repo/python", "/repo/tests/pyodide"]
import cases
json.dumps({"python": sys.version, "results": cases.run("/repo")})
`);
process.stdout.write(out);

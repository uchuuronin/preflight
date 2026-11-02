import { describe, expect, it } from "vitest";
import { fromFile, parsePackages, troubleFromFailure, troubleFromProbes, withPackages } from "./checkinHelp";

describe("where a check-in stopped, from what the runner says today (unseen runs, Nov 2)", () => {
	it("reads a repo over the read limit and code over the map's limit", () => {
		// lm-evaluation-harness, vLLM: refused while reading
		expect(troubleFromFailure("submission", "the source is larger than Preflight reads; point it at a smaller directory")).toEqual({ step: "read", limit: "repo" });
		// torchtune: the map's 2 MB, labelled a syntax error today
		expect(troubleFromFailure("syntax_error", "Python source is too large to map (over 2 MB)")).toEqual({ step: "read", limit: "map" });
		// and A's proposed codes
		expect(troubleFromFailure("submission.too_large", "")).toEqual({ step: "read", limit: "repo" });
		expect(troubleFromFailure("map.too_large", "")).toEqual({ step: "read", limit: "map" });
		expect(troubleFromFailure("oom", "out of memory")).toBeNull();
	});
	it("reads the missing imports when every probe stopped at one, with the package name Preflight installs", () => {
		const probes = [
			{ id: "p1", status: "error", error_code: "missing_package", error: "The script imports torchvision, which is not installed in the probe image" },
			{ id: "p2", status: "error", error_code: "missing_package", error: "The script imports torch_geometric.nn, which is not installed in the probe image" },
			{ id: "p3", status: "error", error_code: "missing_package", error: "The script imports torchvision, which is not installed in the probe image" },
		];
		expect(troubleFromProbes(probes, { torchvision: "torchvision", torch_geometric: "torch-geometric" })).toEqual({
			step: "install",
			missing: [
				{ module: "torchvision", pkg: "torchvision" },
				{ module: "torch_geometric.nn", pkg: "torch-geometric" },
			],
		});
		// A's proposed fields win over the message
		expect(troubleFromProbes([{ status: "error", error_code: "missing_package", missing_module: "smolagents", missing_package: "smolagents" }])).toEqual({ step: "install", missing: [{ module: "smolagents", pkg: "smolagents" }] });
		// an import off the list has no package name
		expect(troubleFromProbes([{ status: "error", error_code: "missing_package", error: "A library in the script needs mystery_lib, which is not installed in the probe image" }])).toEqual({ step: "install", missing: [{ module: "mystery_lib", pkg: null }] });
	});
	it("says nothing when a probe ran, or a probe stopped for another reason", () => {
		expect(troubleFromProbes([{ status: "ok" }, { status: "error", error_code: "missing_package", error: "The script imports x, which is not installed" }])).toBeNull();
		expect(troubleFromProbes([{ status: "oom", error_code: "oom" }])).toBeNull();
		expect(troubleFromProbes([])).toBeNull();
	});
});

describe("the ways past", () => {
	it("takes package names only: no versions, URLs or flags", () => {
		expect(parsePackages("torchvision, timm  torchvision")).toEqual({ names: ["torchvision", "timm"], bad: [] });
		expect(parsePackages("torch==2.1 --index-url=x https://e.com/p.whl").bad).toEqual(["torch==2.1", "--index-url=x", "https://e.com/p.whl"]);
	});
	it("sends the same check-in again with the packages, or reading one file", () => {
		const body = { source: { kind: "repo", uri: "https://github.com/a/b", commit: "c" }, gpu: "L4" } as never;
		expect(withPackages(body, ["torchvision"])).toMatchObject({ gpu: "L4", extra_packages: ["torchvision"], source: { uri: "https://github.com/a/b" } });
		expect(fromFile(body, "scripts/train.py")).toMatchObject({ source: { kind: "repo", entrypoint: "scripts/train.py", scope: "entrypoint" } });
	});
});

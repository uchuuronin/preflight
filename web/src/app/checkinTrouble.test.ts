import { describe, expect, it } from "vitest";
import { fromFile, parsePackages, troubleFromFailure, troubleFromProbes, withPackages } from "./checkinTrouble";
import { nothingMeasured } from "./checkinTrouble";

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
		// an import off the list (or a file of the visitor's own) has no package name
		expect(troubleFromProbes([{ status: "error", error_code: "missing_package", error: "The script imports utils, which is not installed in the probe image" }])).toEqual({ step: "install", missing: [{ module: "utils", pkg: null }] });
		// a library's own words aren't read as an import ("needs the protobuf library")
		expect(troubleFromProbes([{ status: "error", error_code: "missing_package", error: "A library in the script needs the, which is not installed in the probe image" }])).toBeNull();
		// no wheels on PyPI: no package to offer
		expect(troubleFromProbes([{ status: "error", error_code: "missing_package", error: "The script imports torch_sparse, which is not installed in the probe image" }], { torch_sparse: "torch-sparse" }, ["torch_sparse"])).toEqual({ step: "install", missing: [{ module: "torch_sparse", pkg: null, noWheels: true }] });
	});
	it("says nothing when a probe ran, or a probe stopped for another reason", () => {
		expect(troubleFromProbes([{ status: "ok" }, { status: "error", error_code: "missing_package", error: "The script imports x, which is not installed" }])).toBeNull();
		expect(troubleFromProbes([{ status: "oom", error_code: "oom" }])).toBeNull();
		// one point ran out of memory after measuring, another stopped at an import: something was measured
		expect(troubleFromProbes([{ status: "oom", error_code: "oom" }, { status: "error", error_code: "missing_package", error: "The script imports x, which is not installed" }])).toBeNull();
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

describe("nothingMeasured", () => {
	it("is the stop when no probe measured, with the code they share", () => {
		expect(nothingMeasured([{ status: "error", error_code: "gpu.unavailable" }, { status: "error", error_code: "gpu.unavailable" }])).toEqual({ step: "measure", code: "gpu.unavailable" });
		expect(nothingMeasured([{ status: "error", error_code: "notebook_only" }, { status: "timeout", error_code: "timeout" }])).toEqual({ step: "measure", code: null });
	});
	it("is nothing when any probe measured (or came from the cache), or there are no probes", () => {
		expect(nothingMeasured([{ status: "ok" }, { status: "error", error_code: "oom" }])).toBeNull();
		expect(nothingMeasured([{ status: "cached" }])).toBeNull();
		expect(nothingMeasured([])).toBeNull();
	});
	it("takes the job's own stop code when the runner sends one", () => {
		expect(nothingMeasured([{ status: "error", error_code: "a" }, { status: "error", error_code: "b" }], "gpu.unavailable")).toEqual({ step: "measure", code: "gpu.unavailable" });
	});
});

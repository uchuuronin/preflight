// @vitest-environment node
import { beforeAll, describe, expect, it } from "vitest";
import realRecord from "../../../../api/tests/fixtures/a-smoke-l4-main-4.record.json";
import golden from "../../../../schema/examples/finetune.json";
import agentGolden from "../../../../schema/examples/agent.json";
import type { FlightRecord } from "../../../../schema/ts/contract";
import { normaliseRecord, type KnobValue, type LoadedRecord } from "../../workspace/record/types";
import {
	CLI_INSTALL,
	changes,
	commandText,
	configText,
	editArgs,
	gpu,
	editsText,
	type ExportInput,
	pyLiteral,
	recheckText,
	SETUP_S,
	sbatchText,
	shq,
	slurmTime,
	wallTime,
} from "./build";

/**
 * The edits are shell scripts, so these tests run them with sh and perl. The web tsconfig has
 * DOM types only (no @types/node, C's config), so Node's modules load at run time behind the
 * few signatures used here.
 */
interface NodeApi {
	spawnSync(cmd: string, args: string[], opts: { cwd: string; encoding: "utf8" }): { status: number | null; stderr: string };
	execFileSync(cmd: string, args: string[], opts: { cwd: string }): unknown;
	mkdtempSync(prefix: string): string;
	readFileSync(path: string, enc: "utf8"): string;
	writeFileSync(path: string, text: string): void;
	tmpdir(): string;
	join(...parts: string[]): string;
}

let node: NodeApi;

beforeAll(async () => {
	const load = (name: string) => import(/* @vite-ignore */ `node:${name}`);
	const [cp, fs, os, path] = await Promise.all(["child_process", "fs", "os", "path"].map(load));
	node = { ...cp, ...fs, ...os, join: path.join } as NodeApi;
});

const real = normaliseRecord(realRecord as unknown as FlightRecord);
const gold = normaliseRecord(golden as unknown as FlightRecord);
const agent = normaliseRecord(agentGolden as unknown as FlightRecord);

function current(r: LoadedRecord): Record<string, KnobValue> {
	return Object.fromEntries(r.knobs.map((k) => [k.id, k.current as KnobValue]));
}

function input(r: LoadedRecord, set: Record<string, KnobValue> = {}, extra: Partial<ExportInput> = {}): ExportInput {
	return {
		record: r,
		knobs: { ...current(r), ...set },
		estimates: r.predictions.estimates,
		link: "https://example.test/#/cockpit/tok",
		illustrative: false,
		...extra,
	};
}

describe("changes", () => {
	it("lists nothing when the settings are the record's", () => {
		expect(changes(input(real))).toEqual([]);
	});

	it("makes exact edits on A's real record: int, float written as 1e-4, and a module constant", () => {
		const cs = changes(input(real, { batch_size: 8, lr: 0.0002, seq_len: 2048 }));
		const by = Object.fromEntries(cs.map((c) => [c.id, c]));
		expect(by.batch_size).toMatchObject({ how: "exact", line: 88, newExpr: "per_device_train_batch_size=8" });
		expect(by.lr).toMatchObject({ how: "exact", line: 91, expr: "learning_rate=1e-4", newExpr: "learning_rate=0.0002" });
		expect(by.seq_len).toMatchObject({ how: "exact", line: 32, newExpr: "MAX_LEN = 2048" });
	});

	it("leaves precision to the hand (bf16=True is not the value 'bf16')", () => {
		const [c] = changes(input(real, { precision: "fp16" }));
		expect(c).toMatchObject({ id: "precision", how: "by_hand", line: 94, newExpr: null });
	});

	it("routes settings that need code to Muse Code, and where-it-runs settings to the launch", () => {
		const by = Object.fromEntries(changes(input(real, { packing: true, gpu_type: "A100-SXM-80GB" })).map((c) => [c.id, c]));
		expect(by.packing).toMatchObject({ how: "code", template: "data.enable_packing" });
		expect(by.gpu_type).toMatchObject({ how: "launch" });
	});

	it("doesn't call a changed type an exact edit (an int knob given a string)", () => {
		const [c] = changes(input(real, { batch_size: "8" as unknown as number }));
		expect(c.how).toBe("by_hand");
	});
});

describe("editsText runs", () => {
	const script = () => node.readFileSync(new URL("../../../../zoo/finetune_lora_hf/train.py", import.meta.url).pathname, "utf8");

	function run(text: string, src: string): { status: number | null; out: string; err: string } {
		const dir = node.mkdtempSync(node.join(node.tmpdir(), "pf-edits-"));
		node.writeFileSync(node.join(dir, "train.py"), src);
		node.writeFileSync(node.join(dir, "edits.sh"), text);
		const r = node.spawnSync("sh", ["edits.sh"], { cwd: dir, encoding: "utf8" });
		return { status: r.status, out: node.readFileSync(node.join(dir, "train.py"), "utf8"), err: r.stderr };
	}

	it("the record's lines still match the zoo script it was read from", () => {
		const lines = script().split("\n");
		for (const k of real.job.script_map?.knobs ?? []) {
			if (!k.location?.expr || !k.location.line) continue;
			expect(lines[k.location.line - 1], k.id).toContain(k.location.expr);
		}
	});

	it("edits exactly the planned lines and nothing else", () => {
		const text = editsText(input(real, { batch_size: 8, lr: 0.0002, seq_len: 2048 }));
		const r = run(text, script());
		expect(r.status, r.err).toBe(0);
		const before = script().split("\n");
		const after = r.out.split("\n");
		const diff = before.flatMap((l: string, i: number) => (l === after[i] ? [] : [i + 1]));
		expect(diff).toEqual([32, 88, 91]);
		expect(after[87]).toContain("per_device_train_batch_size=8");
		expect(after[90]).toContain("learning_rate=0.0002");
		expect(after[31]).toContain("MAX_LEN = 2048");
	});

	it("refuses, and says why, when the line has moved", () => {
		const text = editsText(input(real, { batch_size: 8 }));
		const r = run(text, `# a new first line\n${script()}`);
		expect(r.status).not.toBe(0);
		expect(r.err).toContain("train.py line 88 no longer reads: per_device_train_batch_size=4");
		expect(r.out).toBe(`# a new first line\n${script()}`);
	});

	it("is valid shell even when nothing changed", () => {
		const r = run(editsText(input(real)), script());
		expect(r.status).toBe(0);
		expect(r.out).toBe(script());
	});
});

/** A copy of the real record whose only knob is this one, read at this file, line and expression. */
function withKnob(id: string, current: KnobValue, loc: { file: string; line: number; expr: string } | null, extra: Record<string, unknown> = {}): LoadedRecord {
	const knob = { ...real.knobs[0], id, current, needs_code: false, patch_template: null, ...extra };
	const sm = real.job.script_map;
	return {
		...real,
		knobs: [knob],
		job: { ...real.job, script_map: sm ? { ...sm, knobs: loc ? [{ id, label: id, kind: "int" as const, current, location: loc }] : [] } : sm },
	} as unknown as LoadedRecord;
}

function runFiles(text: string, files: Record<string, string>): { status: number | null; files: Record<string, string>; err: string; pwned: boolean } {
	const dir = node.mkdtempSync(node.join(node.tmpdir(), "pf-x-"));
	for (const [f, s] of Object.entries(files)) node.writeFileSync(node.join(dir, f), s);
	node.writeFileSync(node.join(dir, "edits.sh"), text);
	const r = node.spawnSync("sh", ["edits.sh"], { cwd: dir, encoding: "utf8" });
	const after = Object.fromEntries(Object.keys(files).map((f) => [f, node.readFileSync(node.join(dir, f), "utf8")]));
	const pwned = node.spawnSync("test", ["-e", node.join(dir, "pwned")], { cwd: dir, encoding: "utf8" }).status === 0;
	return { status: r.status, files: after, err: r.stderr, pwned };
}

describe("edit script safety", () => {
	it("changes the whole token only: lr, not warmup_lr", () => {
		const rec = withKnob("lr", 0.0001, { file: "t.py", line: 1, expr: "lr=1e-4" });
		const r = runFiles(editsText(input(rec, { lr: 0.0002 })), { "t.py": "f(warmup_lr=1e-4, lr=1e-4)\n" });
		expect(r.status, r.err).toBe(0);
		expect(r.files["t.py"]).toBe("f(warmup_lr=1e-4, lr=0.0002)\n");
	});

	it("refuses a line that now reads n = 40 where the record read n = 4", () => {
		const rec = withKnob("n", 4, { file: "t.py", line: 1, expr: "n = 4" });
		const r = runFiles(editsText(input(rec, { n: 8 })), { "t.py": "n = 40\n" });
		expect(r.status).not.toBe(0);
		expect(r.err).toContain("t.py line 1 no longer reads: n = 4");
		expect(r.files["t.py"]).toBe("n = 40\n");
	});

	it("changes nothing when a later edit no longer holds, though an earlier one would", () => {
		const src = node.readFileSync(new URL("../../../../zoo/finetune_lora_hf/train.py", import.meta.url).pathname, "utf8");
		const lines = src.split("\n");
		expect(lines[87]).toContain("per_device_train_batch_size=4");
		lines[90] = lines[90].replace("learning_rate=1e-4", "learning_rate=3e-4"); // only line 91 has moved on
		const moved = lines.join("\n");
		const r = runFiles(editsText(input(real, { batch_size: 8, lr: 0.0002 })), { "train.py": moved });
		expect(r.status).not.toBe(0);
		expect(r.err).toContain("train.py line 91 no longer reads: learning_rate=1e-4");
		expect(r.err).toContain("Nothing was changed.");
		expect(r.files["train.py"]).toBe(moved);
	});

	it("passes quotes, $, @, |, # and backslashes through as data, never as code", () => {
		const rec = withKnob("opt", "a$b|#@c", { file: "t.py", line: 1, expr: "opt = 'a$b|#@c'" }, { kind: "categorical" });
		const to = "it's $(touch pwned) `touch pwned` | # \\ ok";
		const [c] = changes(input(rec, { opt: to }));
		expect(c.how).toBe("exact");
		const r = runFiles(editsText(input(rec, { opt: to })), { "t.py": "opt = 'a$b|#@c'\n" });
		expect(r.status, r.err).toBe(0);
		expect(r.files["t.py"]).toBe(`opt = ${JSON.stringify(to)}\n`);
		expect(r.pwned).toBe(false);
	});

	it("keeps a line break in a value, an expression or a GPU name out of the script's code", () => {
		const evil = 'x"\n$(touch pwned)\n#';
		const byValue = withKnob("tag", evil, { file: "t.py", line: 1, expr: "TAG = 1" });
		const byExpr = withKnob("p", "bf16", { file: "t.py", line: 1, expr: 'P = ("bf16"\n"$(touch pwned)"\n"#")' });
		const byCode = withKnob("q", false, null, { needs_code: true, patch_template: "a\n$(touch pwned)\n#" });
		for (const [rec, set] of [
			[byValue, { tag: "y" }],
			[byExpr, { p: "fp16" }],
			[byCode, { q: true }],
		] as const) {
			const text = editsText(input(rec, set));
			expect(text.split("\n").filter((l) => l.includes("touch pwned")).every((l) => l.startsWith("#"))).toBe(true);
			const r = runFiles(text, { "t.py": "TAG = 1\n" });
			expect(r.status, r.err).toBe(0);
			expect(r.pwned).toBe(false);
		}
		const gpuRec = { ...real, env: { ...real.env, gpu: { ...real.env?.gpu, name: "L4\n$(touch pwned)" } } } as LoadedRecord;
		const sb = sbatchText(input(gpuRec, { gpu_type: "L4\n$(touch pwned)" }));
		expect(sb.split("\n").filter((l) => l.includes("touch pwned")).every((l) => l.startsWith("#"))).toBe(true);
	});

	it("handles a file name that starts with a dash", () => {
		const rec = withKnob("n", 4, { file: "-x.py", line: 1, expr: "n = 4" });
		const r = runFiles(editsText(input(rec, { n: 8 })), { "-x.py": "n = 4\n" });
		expect(r.status, r.err).toBe(0);
		expect(r.files["-x.py"]).toBe("n = 8\n");
	});

	it("keeps a float a float", () => {
		const rec = withKnob("lr", 1, { file: "t.py", line: 1, expr: "lr = 1.0" });
		expect(changes(input(rec, { lr: 2 }))[0].newExpr).toBe("lr = 2.0");
	});

	it("labels exact only what the script makes", () => {
		for (const c of changes(input(real, { batch_size: 8, lr: 0.0002, precision: "fp16", packing: true }))) {
			expect(editArgs(c) !== null, c.id).toBe(c.how === "exact");
		}
	});
});

describe("sbatch", () => {
	it("sizes the time from the slow end of the range plus setup, in whole 5-minute steps", () => {
		const est = gold.predictions.estimates["time.total"];
		const text = sbatchText(input(gold));
		expect(text).toContain(`#SBATCH --time=${slurmTime((est?.hi ?? 0) + SETUP_S)}`);
		expect(wallTime(est)).toBe(Math.ceil(((est?.hi ?? 0) + SETUP_S) / 300) * 300);
		expect((wallTime(est) ?? 0) % 300).toBe(0);
		expect(text).toContain("#SBATCH --gres=gpu:1");
		expect(text).toContain("Planned on 1 x L4");
		expect(text.trim().split("\n").at(-1)).toBe("python train.py");
	});

	it("follows the GPU chosen in the cockpit", () => {
		expect(sbatchText(input(gold, { gpu_type: "A100-SXM-80GB" }))).toContain("Planned on 1 x A100-SXM-80GB");
	});

	it("says so instead of guessing when there is no time estimate", () => {
		const text = sbatchText(input(real));
		expect(text).not.toContain("--time=");
		expect(text).toContain("set --time yourself");
	});

	it("asks for no GPU for an agent run", () => {
		expect(sbatchText(input(agent))).not.toContain("--gres");
	});

	it("asks for the cluster's limit, never more, and says the plan needs longer", () => {
		const capped = { ...gold, job: { ...gold.job, constraints: { ...gold.job.constraints, wall_time_cap_s: 3630 } } };
		const text = sbatchText(input(capped));
		expect(text).toContain("#SBATCH --time=01:00:00");
		expect(text).toContain("save checkpoints and resubmit");
	});

	it("tells the reader to run the edits first, and counts the changes it can't make", () => {
		const text = sbatchText(input(real, { batch_size: 8, packing: true }));
		expect(text).toContain("Run 'sh preflight-edits.sh' once before submitting");
		expect(text).toContain("# 1 more change to make by hand or with Muse Code");
	});

	it("puts GPUs per node in --gres", () => {
		const two = { ...gold, env: { ...gold.env, nodes: 2 } } as LoadedRecord;
		expect(sbatchText(input(two, { gpu_count: 8 }))).toContain("#SBATCH --gres=gpu:4");
		expect(gpu(input(two, { gpu_count: 8 })).count).toBe(8);
	});
});

describe("config", () => {
	it("carries every setting, what changed, and only the readouts that are predicted", () => {
		const plan = JSON.parse(configText(input(gold, { batch_size: 8 })));
		expect(plan.settings.batch_size).toBe(8);
		expect(plan.changed).toEqual({ batch_size: { from: 4, to: 8 } });
		expect(plan.gpu).toEqual({ type: "L4", count: 1 });
		expect(plan.predicted["time.total"]).toMatchObject({ unit: "s", level: 0.9 });
		expect(plan.link).toBe("https://example.test/#/cockpit/tok");
		expect(JSON.stringify(plan)).not.toContain(gold.record_id);
	});

	it("lists unpredicted readouts by name rather than inventing them", () => {
		const est = { ...gold.predictions.estimates } as Record<string, NonNullable<(typeof gold.predictions.estimates)[string]>>;
		est["memory.peak"] = { ...est["memory.peak"], status: "abstain" };
		delete est["risk.oom"];
		const plan = JSON.parse(configText(input(gold, {}, { estimates: est })));
		expect(plan.not_predicted).toEqual(["memory.peak", "risk.oom"]);
		expect(Object.keys(plan.predicted)).toEqual(["time.total", "cost.total"]);
	});

	it("marks a sample's numbers as illustrative", () => {
		expect(JSON.parse(configText(input(gold, {}, { illustrative: true }))).note).toMatch(/illustrative/);
	});
});

describe("helpers", () => {
	it("writes Python literals", () => {
		expect([pyLiteral(true), pyLiteral(4), pyLiteral(1e-4), pyLiteral("a'b"), pyLiteral(Number.NaN)]).toEqual([
			"True",
			"4",
			"0.0001",
			'"a\'b"',
			null,
		]);
	});

	it("quotes for the shell only when needed", () => {
		expect(shq("train.py")).toBe("train.py");
		expect(shq("it's here")).toBe(`'it'\\''s here'`);
	});

	it("formats Slurm time", () => {
		expect([slurmTime(1), slurmTime(3600), slurmTime(90061)]).toEqual(["00:05:00", "01:00:00", "1-01:05:00"]);
	});

	it("writes the CLI re-check with the record's entry point and arguments", () => {
		const repo = (r: LoadedRecord, source: Record<string, unknown>) => ({ ...r, job: { ...r.job, source: { ...r.job.source, kind: "repo", ...source } } }) as LoadedRecord;
		const text = recheckText({ record: repo(gold, { entrypoint: "src/my train.py", args: ["--name", "my run"] }) }) ?? "";
		expect(text).toContain(`${CLI_INSTALL}\n`);
		expect(text.trim().split("\n").at(-1)).toBe("preflight run . --entrypoint='src/my train.py' --out .preflight/records -- --name 'my run'");
		expect(recheckText({ record: repo(real, { entrypoint: "-x.py" }) })?.trim().split("\n").at(-1)).toBe("preflight run . --entrypoint=-x.py --out .preflight/records");
	});

	it("offers no re-check for samples, notebooks or API runs", () => {
		const as = (kind: string) => ({ ...real, job: { ...real.job, source: { ...real.job.source, kind } } }) as LoadedRecord;
		expect(recheckText({ record: as("sample") })).toBeNull();
		expect(recheckText({ record: as("notebook") })).toBeNull();
		expect(recheckText({ record: { ...as("repo"), env: { ...real.env, backend: "api" } } as LoadedRecord })).toBeNull();
		expect(recheckText({ record: as("upload") })).not.toBeNull();
	});

	it("leaves out arguments that hold control characters, and says so", () => {
		const r = { ...real, job: { ...real.job, source: { ...real.job.source, kind: "repo", args: ["--name", "ok\r$(evil)"] } } } as LoadedRecord;
		for (const text of [commandText({ record: r }), recheckText({ record: r }) ?? ""]) {
			expect(text).not.toContain("evil");
			expect(text).toContain("left out: add them yourself");
		}
	});

	it("quotes an entry point with a space that A's launch line leaves bare", () => {
		const spaced = { ...real, job: { ...real.job, source: { ...real.job.source, entrypoint: "my train.py" }, script_map: { ...real.job.script_map, launch: "python my train.py" } } } as LoadedRecord;
		expect(commandText({ record: spaced })).toBe("python 'my train.py'");
	});

	it("builds the command from the record, quoting arguments", () => {
		const withArgs = { ...gold, job: { ...gold.job, source: { ...gold.job.source, args: ["--name", "my run"] } } };
		expect(commandText({ record: withArgs })).toBe("python train.py --name 'my run'");
	});
});

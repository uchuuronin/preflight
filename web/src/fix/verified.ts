/**
 * What verify found for a fix, shown on its lane (PM, Nov 3: D films the race's verdicts). Verify is
 * a GPU job (A's job store, mode "verify", with the patch) that the live site doesn't run on a race:
 * a lane's verdict comes from the race's own `verify` when the API sends one, else from the verify
 * runs Preflight made of the same fix on the same sample, dated.
 */
import type { RaceCandidate } from "./race";

export interface Verdict {
	passed: boolean;
	/** Why it was rejected: the re-run still ran out of memory. */
	reason?: "oom";
	/** The fixed side's evaluation peak, bytes. */
	evalPeakBytes?: number;
	/** The loss's change at the same seed, as a fraction (0.0011 is 0.11%). */
	lossDelta?: number;
	/** When Preflight checked an earlier patch for the same fix (a verify run before this race), YYYY-MM-DD. */
	checkedOn?: string;
	/** The GPU that earlier check ran on. */
	gpu?: string;
}

/**
 * Source: measured, C's fix race on the LoRA sample's eval out-of-memory, Nov 2 (GitHub run
 * 41482457185, `api/fix_app.py::race_stack` on an A10; docs/status/c.md D8). Eval without gradients
 * alone: verify failed, the evaluation ran out of memory with and without it (record
 * 01M6EHR42KZR8F3MWZJKAQBWNW). The stack at eval batch 8: verify passed every check, eval peak
 * 11,153,155,072 bytes, loss within 0.11% at the same seed (record 01M6EHYYCM3JRHXKJ9XM5EPJ41).
 */
const CHECKED: ReadonlyArray<{ sample: string; templateId: string; value: unknown; verdict: Verdict }> = [
	{ sample: "lora-qwen", templateId: "eval.wrap_no_grad", value: null, verdict: { passed: false, reason: "oom", checkedOn: "2026-11-02", gpu: "A10" } },
	{ sample: "lora-qwen", templateId: "eval.no_grad_and_batch", value: 8, verdict: { passed: true, evalPeakBytes: 11_153_155_072, lossDelta: 0.0011, checkedOn: "2026-11-02", gpu: "A10" } },
];

type RaceVerify = { passed?: boolean; reason?: string | null; eval_peak_bytes?: number | null; loss_delta?: number | null };

/** A lane's verdict: the race's own when it has one, else a dated check of the same fix on this sample. */
export function verdictFor(recordRef: string, c: Pick<RaceCandidate, "template_id" | "value"> & { verify?: RaceVerify | null }): Verdict | null {
	const v = c.verify;
	if (v && typeof v.passed === "boolean") {
		return {
			passed: v.passed,
			...(v.reason === "oom" ? { reason: "oom" as const } : {}),
			...(typeof v.eval_peak_bytes === "number" ? { evalPeakBytes: v.eval_peak_bytes } : {}),
			...(typeof v.loss_delta === "number" ? { lossDelta: v.loss_delta } : {}),
		};
	}
	const same = CHECKED.find((x) => x.sample === recordRef && x.templateId === c.template_id && (x.value ?? null) === (c.value ?? null));
	return same?.verdict ?? null;
}

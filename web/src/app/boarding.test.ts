import { describe, expect, it } from "vitest";
import { boardingState, type BoardingEvent, STAGES } from "./boarding";

const ev = (seq: number, stage: BoardingEvent["stage"], more: Partial<BoardingEvent> = {}): BoardingEvent => ({
	job_id: "01JOB",
	seq,
	at: "2026-10-28T12:00:00Z",
	stage,
	...more,
});

describe("boardingState", () => {
	it("opens the record by its share token, which arrives with the skeleton, not on done", () => {
		const s = boardingState([ev(0, "queued"), ev(1, "skeleton", { share_token: "tok_abcdefghijklmnop", record_rev: 1 }), ev(2, "probing"), ev(3, "done", { record_id: "01REC" })]);
		expect(s.done).toBe(true);
		expect(s.open).toBe("tok_abcdefghijklmnop");
	});

	it("never opens the cockpit on a record id", () => {
		const s = boardingState([ev(0, "done", { record_id: "01REC" })]);
		expect(s.done).toBe(true);
		expect(s.open).toBeNull();
	});

	it("shows a failure by its error code, with A's own words as the hint", () => {
		const s = boardingState([ev(0, "queued"), ev(1, "failed", { error_code: "submission.knob_value", message: "batch_size must be one of 1, 2, 4, 8", retryable: false })]);
		expect(s.failed).toBe(true);
		expect(s.error?.code).toBe("submission.knob_value");
		expect(s.error?.message).toMatch(/outside/);
		expect(s.error?.hint).toBe("batch_size must be one of 1, 2, 4, 8");
	});

	it("falls back to the code's family, then to the plain message", () => {
		expect(boardingState([ev(0, "failed", { error_code: "internal.something_new" })]).error?.code).toBe("internal");
		const plain = boardingState([ev(0, "failed", { message: "the queue was drained" })]).error;
		expect(plain?.message).toBe("the queue was drained");
	});

	it("doesn't repeat the copy's own text as its hint", () => {
		const s = boardingState([ev(0, "failed", { error_code: "budget", message: "error.budget" })]);
		expect(s.error?.hint).not.toBe(s.error?.message);
	});

	it("reads stage and queue position from the latest event", () => {
		const s = boardingState([ev(0, "queued", { queue_position: 2 }), ev(1, "admitted", { queue_position: 0 })]);
		expect(s.last?.stage).toBe("admitted");
		expect(s.done).toBe(false);
		expect(s.failed).toBe(false);
	});

	it("treats cancelled as over, without an error alert for a plain cancel", () => {
		const s = boardingState([ev(0, "cancelled")]);
		expect(s.failed).toBe(true);
		expect(s.cancelled).toBe(true);
		expect(s.error).toBeNull();
		expect(boardingState([ev(0, "cancelled", { error_code: "budget" })]).error?.code).toBe("budget");
	});
});

describe("where a job stopped", () => {
	const ev = (seq: number, stage: string, extra: Record<string, unknown> = {}) =>
		({ job_id: "j", seq, at: "2026-10-31T00:00:00Z", stage, ...extra }) as never;

	it("marks the leg of the last stage reached when the job fails, not none", () => {
		const s = boardingState([ev(0, "queued"), ev(1, "fetching"), ev(2, "harnessing"), ev(3, "probing"), ev(4, "failed", { error_code: "oom" })]);
		expect(s.failed).toBe(true);
		expect(STAGES[s.at]?.key).toBe("boarding.stage.probe");
	});

	it("puts a failure before any stage on the queue leg, and a cancel where it happened", () => {
		expect(boardingState([ev(0, "failed", { error_code: "submission.invalid" })]).at).toBe(0);
		const c = boardingState([ev(0, "queued"), ev(1, "mapping"), ev(2, "cancelled")]);
		expect(c.cancelled).toBe(true);
		expect(STAGES[c.at]?.key).toBe("boarding.stage.read");
	});

	it("follows a running job leg by leg, and ends on the last when it's done", () => {
		expect(STAGES[boardingState([ev(0, "queued"), ev(1, "triage")]).at]?.key).toBe("boarding.stage.triage");
		expect(boardingState([ev(0, "predicting"), ev(1, "done")]).at).toBe(STAGES.length - 1);
		expect(boardingState([]).at).toBe(-1);
	});
});

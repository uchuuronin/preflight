import { describe, expect, it } from "vitest";
import { boardingState, type BoardingEvent } from "./boarding";

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

	it("treats cancelled as failed", () => {
		expect(boardingState([ev(0, "cancelled")]).failed).toBe(true);
	});
});

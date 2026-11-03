import { describe, expect, it } from "vitest";
import { errorCopy, eventText } from "./errors";
import { t } from "./i18n";

const CODES = ["oom", "oom.eval", "oom.host", "missing_package", "download_blocked", "gated_model", "file_not_found", "cuda_error", "nccl", "dtype_unsupported", "shape_mismatch", "dataloader_worker", "syntax_error", "script_exit", "too_few_steps", "timeout", "killed", "recorder", "user_error", "nan_divergence"];

describe("error copy for A's probe error codes", () => {
	it("has a message and a hint for every family A lists", () => {
		for (const code of CODES) {
			const e = errorCopy(code);
			expect(e.code).toBe(code);
			expect(e.message).not.toBe(`error.${code}`);
			expect(e.message.length).toBeGreaterThan(10);
			expect(e.hint.length).toBeGreaterThan(0);
		}
	});

	it("falls back to the family, then to a generic line, never a bare code", () => {
		expect(errorCopy("oom.fsdp_shard").message).toBe(t("error.oom"));
		expect(errorCopy("brand_new_code", "Traceback: boom").message).toBe("Traceback: boom");
		expect(errorCopy(null).message).toBe(t("error.generic"));
	});

	it("translates job event messages that are keys and passes log lines through", () => {
		expect(eventText("error.gated_model")?.text).toBe(t("error.gated_model"));
		expect(eventText("pip install failed")?.text).toBe("pip install failed");
		expect(eventText(null)).toBeNull();
	});
});

describe("the repo's own module (A, Nov 3: missing_module.local)", () => {
	it("says to check in the whole repository, not to add it to requirements.txt", () => {
		const c = errorCopy("missing_module.local", "No module named 'helpers'");
		expect(c.code).toBe("missing_module.local");
		expect(c.hint).toMatch(/whole repository/);
		expect(c.hint).toMatch(/won't help/);
		expect(c.message).not.toMatch(/install/i);
	});
});

describe("A's new stops (Nov 3)", () => {
	it("names the notebook-only call and the too-old torch, with what to do", () => {
		expect(errorCopy("notebook_only").hint).toMatch(/login cell/);
		const torch = errorCopy("torch.too_old");
		expect(torch.code).toBe("torch.too_old");
		expect(torch.hint).toMatch(/torch>=2\.6/);
	});
});

describe("an entry point that runs nothing (A, Nov 3)", () => {
	it("says so and what to name instead", () => {
		const c = errorCopy("entry.runs_nothing", "raw");
		expect(c.message).toMatch(/runs nothing/);
		expect(c.hint).toMatch(/python/);
	});
	it("shows A's own words for a sub-code it doesn't know yet", () => {
		expect(errorCopy("entry.something_else", "The file defines no code to run.").message).toBe("The file defines no code to run.");
	});
});

describe("a check-in refusal this page doesn't know yet", () => {
	it("keeps A's own words, under the family's hint", () => {
		const c = errorCopy("submission.no_entry_code", "The entry point defines no code that runs.");
		expect(c.message).toBe("The entry point defines no code that runs.");
		expect(c.hint).toBe(t("error.submission.hint"));
	});
});

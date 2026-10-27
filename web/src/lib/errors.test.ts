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

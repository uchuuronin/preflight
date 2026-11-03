import { describe, expect, it } from "vitest";
import { commandWords } from "./command";

describe("a typed command as its words", () => {
	it("splits as a shell would: quotes keep spaces, a backslash escapes", () => {
		expect(commandWords(`lm_eval --tasks 'arc easy' --x "a \\"b\\"" c\\ d`)).toEqual({ ok: true, words: ["lm_eval", "--tasks", "arc easy", "--x", 'a "b"', "c d"] });
	});
	it("refuses what only a shell runs, but takes a quoted pipe as a word", () => {
		expect(commandWords("lm_eval --tasks x | tee out")).toMatchObject({ ok: false, reason: "shell", word: "|" });
		expect(commandWords("lm_eval --tasks x >out.txt")).toMatchObject({ ok: false, reason: "shell", word: ">" });
		expect(commandWords("lm_eval --tasks x 2>/dev/null")).toMatchObject({ ok: false, reason: "shell", word: ">" });
		expect(commandWords("lm_eval --model_args pretrained=$MODEL")).toMatchObject({ ok: false, reason: "shell", word: "$" });
		expect(commandWords(`lm_eval --x "$HOME"`)).toMatchObject({ ok: false, reason: "shell", word: "$" });
		expect(commandWords("lm_eval --x 'a>b'")).toEqual({ ok: true, words: ["lm_eval", "--x", "a>b"] });
		expect(commandWords("lm_eval --sep '|'")).toEqual({ ok: true, words: ["lm_eval", "--sep", "|"] });
	});
	it("names an open quote, an empty line and a command off the list", () => {
		expect(commandWords(`lm_eval "x`)).toMatchObject({ ok: false, reason: "quote" });
		expect(commandWords("   ")).toMatchObject({ ok: false, reason: "empty" });
		expect(commandWords("python train.py")).toMatchObject({ ok: false, reason: "unknown", word: "python" });
	});
	it("keeps to A's limits", () => {
		expect(commandWords(`lm_eval ${"a ".repeat(64)}`)).toMatchObject({ ok: false, reason: "too_long" });
		expect(commandWords(`lm_eval ${"a".repeat(8200)}`)).toMatchObject({ ok: false, reason: "too_long" });
	});
});

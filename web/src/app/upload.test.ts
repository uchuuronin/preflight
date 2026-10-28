import { describe, expect, it } from "vitest";
import { MAX_TOTAL_BYTES, uploadSource, type PickedFile } from "./upload";

const f = (name: string, text = "print(1)\n", size?: number): PickedFile => ({ name, text, size: size ?? new TextEncoder().encode(text).length });

describe("uploadSource", () => {
	it("sends a script and its requirements as an upload, entry point by A's own order", () => {
		const r = uploadSource([f("requirements.txt", "peft\n"), f("train.py"), f("utils.py")]);
		expect(r.ok).toBe(true);
		if (!r.ok) return;
		expect(r.source).toEqual({ kind: "upload", files: { "requirements.txt": "peft\n", "train.py": "print(1)\n", "utils.py": "print(1)\n" }, entrypoint: "train.py" });
	});

	it("takes the only .py file as the entry point", () => {
		const r = uploadSource([f("finetune_lora.py")]);
		expect(r.ok && r.source.kind === "upload" && r.source.entrypoint).toBe("finetune_lora.py");
	});

	it("asks which file to run when several could be the entry point", () => {
		const r = uploadSource([f("a.py"), f("b.py")]);
		expect(r).toEqual({ ok: false, reason: "upload.which", candidates: ["a.py", "b.py"] });
		const chosen = uploadSource([f("a.py"), f("b.py")], "b.py");
		expect(chosen.ok && chosen.source.kind === "upload" && chosen.source.entrypoint).toBe("b.py");
	});

	it("sends one notebook raw as kind notebook, for A's converter", () => {
		const nb = JSON.stringify({ cells: [] });
		const r = uploadSource([f("train.ipynb", nb)]);
		expect(r.ok && r.source).toEqual({ kind: "notebook", files: { "train.ipynb": nb } });
	});

	it("refuses a notebook mixed with other files, binary files, dotfiles and odd names", () => {
		expect(uploadSource([f("train.ipynb", "{}"), f("x.py")])).toMatchObject({ ok: false, reason: "upload.notebook_alone" });
		expect(uploadSource([f("w.py", "a\u0000b")])).toMatchObject({ ok: false, reason: "upload.binary" });
		expect(uploadSource([f(".env", "K=1")])).toMatchObject({ ok: false, reason: "upload.name" });
		expect(uploadSource([f("we!rd.py")])).toMatchObject({ ok: false, reason: "upload.name" });
		expect(uploadSource([f("model.safetensors", "x")])).toMatchObject({ ok: false, reason: "upload.type" });
	});

	it("keeps the whole check-in under the API's limit", () => {
		expect(uploadSource([f("train.py", "x", MAX_TOTAL_BYTES + 1)])).toMatchObject({ ok: false, reason: "upload.too_big" });
	});

	it("needs a file and something to run", () => {
		expect(uploadSource([])).toMatchObject({ ok: false, reason: "upload.empty" });
		expect(uploadSource([f("requirements.txt", "peft\n")])).toMatchObject({ ok: false, reason: "upload.no_script" });
	});

	it("refuses two files with the same name", () => {
		expect(uploadSource([f("train.py"), f("train.py")])).toMatchObject({ ok: false, reason: "upload.duplicate" });
	});
});

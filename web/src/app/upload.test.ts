import { describe, expect, it } from "vitest";
import { MAX_FILES, MAX_TOTAL_BYTES, precheck, uploadSource, type PickedFile } from "./upload";

const f = (name: string, text = "print(1)\n", size?: number): PickedFile => ({ name, text, size: size ?? new TextEncoder().encode(text).length });

describe("uploadSource", () => {
	it("sends a script and its requirements as an upload, entry point by A's own order", () => {
		const r = uploadSource([f("requirements.txt", "peft\n"), f("train.py"), f("utils.py")]);
		expect(r.ok).toBe(true);
		if (!r.ok) return;
		expect(r.source).toEqual({ kind: "upload", files: { "requirements.txt": "peft\n", "train.py": "print(1)\n", "utils.py": "print(1)\n" }, entrypoint: "train.py" });
		expect(r.scripts).toEqual(["train.py", "utils.py"]);
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

	it("keeps the whole check-in under the API's limit, counted as the JSON it is sent as", () => {
		expect(uploadSource([f("train.py", "x", MAX_TOTAL_BYTES + 1)])).toMatchObject({ ok: false, reason: "upload.too_big" });
		// Under the limit on disk, over it once quotes and newlines are escaped.
		const nb = '"\n'.repeat(Math.ceil(MAX_TOTAL_BYTES / 2 / 2));
		expect(uploadSource([f("train.ipynb", nb, MAX_TOTAL_BYTES - 1)])).toMatchObject({ ok: false, reason: "upload.too_big" });
	});

	it("says no from names and sizes before reading a huge file", () => {
		expect(precheck([{ name: "model.safetensors", size: 4e9 }])).toBe("upload.type");
		expect(precheck([{ name: "train.py", size: 4e9 }])).toBe("upload.too_big");
		expect(precheck(Array.from({ length: MAX_FILES + 1 }, (_, i) => ({ name: `f${i}.py`, size: 1 })))).toBe("upload.too_many");
		expect(precheck([{ name: "train.py", size: 10 }])).toBeNull();
	});

	it("takes small text data along, as A allows", () => {
		expect(uploadSource([f("train.py"), f("tickets.csv", "a,b\n"), f("rows.jsonl", "{}\n")]).ok).toBe(true);
	});

	it("treats Train.PY as a script, whatever the case", () => {
		const r = uploadSource([f("Train.PY")]);
		expect(r.ok && r.source.kind === "upload" && r.source.entrypoint).toBe("Train.PY");
	});

	it("needs a file and something to run", () => {
		expect(uploadSource([])).toMatchObject({ ok: false, reason: "upload.empty" });
		expect(uploadSource([f("requirements.txt", "peft\n")])).toMatchObject({ ok: false, reason: "upload.no_script" });
	});

	it("refuses two files with the same name", () => {
		expect(uploadSource([f("train.py"), f("train.py")])).toMatchObject({ ok: false, reason: "upload.duplicate" });
	});
});

import { describe, expect, it } from "vitest";
import names from "./fixtures/names.json";
import random from "./fixtures/random.expected.json";
import sent from "./fixtures/sent.expected.json";
import { MAX_FILE_BYTES, notebookRequirements, notebookToScript, prepareNotebook, safeNotebookName } from "./convert";

/**
 * Parity with A's converter (runner/sources.py `_notebook_to_script`, `notebook_requirements` and
 * `fetch` for kind "notebook"): each fixture's expected output was produced by A's code, and
 * tests/test_d_notebook_parity.py checks A's code still produces it, so drift on either side fails.
 */
const NOTEBOOKS = import.meta.glob<string>("./fixtures/*.ipynb", { eager: true, query: "?raw", import: "default" });
const EXPECTED = import.meta.glob<{ script: string; requirements: string[]; files: Record<string, string>; entrypoint: string }>(
	"./fixtures/*.expected.json",
	{ eager: true, import: "default" },
);

const cases = Object.entries(NOTEBOOKS).map(([path, text]) => {
	const name = path.split("/").pop() as string;
	return { name, text, expected: EXPECTED[path.replace(".ipynb", ".expected.json")] };
});

describe("notebook conversion, same as A's", () => {
	it("has a fixture with expected output for every notebook", () => {
		expect(cases.length).toBeGreaterThanOrEqual(4);
		for (const c of cases) expect(c.expected, c.name).toBeDefined();
	});

	for (const c of cases) {
		it(`turns ${c.name} into the script A runs`, () => {
			expect(notebookToScript(c.text)).toBe(c.expected.script);
		});

		it(`reads ${c.name}'s pip lines as A does`, () => {
			expect(notebookRequirements(notebookToScript(c.text))).toEqual(c.expected.requirements);
		});
	}

	it("matches A on 400 seeded random notebooks (every Python space and line break, NaN, odd cells)", () => {
		const diffs = random.cases.flatMap((c, i) => {
			if ("error" in c) {
				try {
					notebookToScript(c.text);
					return [`${i}: A refuses, D converts`];
				} catch {
					return [];
				}
			}
			const script = notebookToScript(c.text);
			const reqs = notebookRequirements(script);
			return script === c.script && JSON.stringify(reqs) === JSON.stringify(c.requirements) ? [] : [`${i}`];
		});
		expect(random.cases).toHaveLength(400);
		expect(diffs).toEqual([]);
	});

	it("sends a notebook that converts to the same script, for every random notebook D accepts", () => {
		const diffs = random.cases.flatMap((c, i) => {
			if ("error" in c) return [];
			const out = prepareNotebook("r.ipynb", c.text);
			if (!out.ok) return out.error === "no_code" || out.error === "bad_text" ? [] : [`${i}: ${out.error}`];
			return notebookToScript(out.source.files["r.ipynb"]) === c.script ? [] : [`${i}`];
		});
		expect(diffs).toEqual([]);
	});

	it("still sends exactly what A's side was checked against (fixtures/sent.expected.json)", () => {
		const diffs = sent.cases.flatMap((c) => {
			const text = c.from.startsWith("random#") ? random.cases[Number(c.from.slice(7))].text : NOTEBOOKS[`./fixtures/${c.from}`];
			const out = prepareNotebook(c.from.startsWith("random#") ? "r.ipynb" : c.from, text);
			return out.ok && out.source.files[c.name] === c.sent ? [] : [c.from];
		});
		expect(sent.cases.length).toBeGreaterThan(200);
		expect(diffs).toEqual([]);
	});

	it("previews the exact files and entry point A will run", () => {
		for (const c of cases.filter((x) => x.expected.script.trim())) {
			const out = prepareNotebook(c.name, c.text);
			expect(out.ok, c.name).toBe(true);
			if (!out.ok) continue;
			expect(out.preview.files).toEqual(c.expected.files);
			expect(out.preview.entrypoint).toBe(c.expected.entrypoint);
		}
	});
});

describe("what D sends", () => {
	const colab = cases.find((c) => c.name === "colab_lora.ipynb") as (typeof cases)[number];

	it("sends kind notebook with one .ipynb file", () => {
		const out = prepareNotebook("colab_lora.ipynb", colab.text);
		if (!out.ok) throw new Error(out.error);
		expect(out.source.kind).toBe("notebook");
		expect(Object.keys(out.source.files)).toEqual(["colab_lora.ipynb"]);
	});

	it("leaves out saved outputs, which A never reads, and the script is unchanged", () => {
		const out = prepareNotebook("colab_lora.ipynb", colab.text);
		if (!out.ok) throw new Error(out.error);
		const sent = out.source.files["colab_lora.ipynb"];
		expect(sent).not.toContain("image/png");
		expect(sent.length).toBeLessThan(colab.text.length / 2);
		expect(notebookToScript(sent)).toBe(notebookToScript(colab.text));
		expect(out.preview.removedBytes).toBeGreaterThan(4000);
	});

	it("counts what the conversion keeps and drops", () => {
		const out = prepareNotebook("colab_lora.ipynb", colab.text);
		if (!out.ok) throw new Error(out.error);
		expect(out.preview).toMatchObject({ codeCells: 3, otherCells: 2, commentedLines: 4 });
	});
});

describe("notebook names A accepts", () => {
	it("keeps names A already accepts", () => {
		expect(safeNotebookName("Untitled1.ipynb")).toBe("Untitled1.ipynb");
		expect(safeNotebookName("my notebook.ipynb")).toBe("my notebook.ipynb");
	});

	it("cleans names a browser download gives, which A rejects", () => {
		for (const [given, sent] of Object.entries(names.names)) expect(safeNotebookName(given), given).toBe(sent);
	});

	it("sends and previews under the cleaned name", () => {
		const out = prepareNotebook("train (1).ipynb", cases[0].text);
		if (!out.ok) throw new Error(out.error);
		expect(Object.keys(out.source.files)).toEqual(["train_1.ipynb"]);
		expect(out.preview.entrypoint).toBe("train_1.py");
		expect(out.preview.renamedFrom).toBe("train (1).ipynb");
	});
});

describe("notebooks A accepts that a browser parser wouldn't", () => {
	it("reads a notebook whose metadata isn't an object, as A does", () => {
		const nb = '{"metadata":"x","cells":[{"cell_type":"code","source":"print(1)"}]}';
		const out = prepareNotebook("m.ipynb", nb);
		expect(out.ok).toBe(true);
		if (out.ok) expect(notebookToScript(out.source.files["m.ipynb"])).toBe("print(1)\n");
	});

	it("reads NaN and Infinity that Jupyter writes in outputs, as Python's json does", () => {
		const nb = '{"cells":[{"cell_type":"code","source":"x = \\"NaN\\"","outputs":[{"data":{"a":NaN,"b":-Infinity,"c":[Infinity]}}]}]}';
		expect(notebookToScript(nb)).toBe('x = "NaN"\n');
		const out = prepareNotebook("n.ipynb", nb);
		expect(out.ok).toBe(true);
		if (out.ok) expect(() => JSON.parse(out.source.files["n.ipynb"])).not.toThrow();
	});

	it("reads a 20 MB notebook whose outputs are full of NaN without stalling the page", () => {
		const row = `"${"x ".repeat(40)}NaN \\" NaN"`;
		const output = `{"output_type":"stream","text":[${Array(2000).fill(row).join(",")}],"v":[${Array(200).fill("NaN").join(",")}]}`;
		const cell = `{"cell_type":"code","source":"y = 1","outputs":[${output}]}`;
		const nb = `{"cells":[${Array(110).fill(cell).join(",")}]}`;
		expect(nb.length).toBeGreaterThan(20_000_000);
		const t0 = performance.now();
		const out = prepareNotebook("big.ipynb", nb);
		const ms = performance.now() - t0;
		expect(out.ok).toBe(true);
		expect(ms).toBeLessThan(1000);
	});

	it("keeps -NaN invalid, as Python's json does", () => {
		expect(prepareNotebook("n.ipynb", '{"cells":[{"cell_type":"code","source":"x","outputs":[-NaN]}]}')).toEqual({ ok: false, error: "not_json" });
	});

	it("leaves NaN inside strings alone", () => {
		expect(notebookToScript('{"cells":[{"cell_type":"code","source":["a = \\"NaN Infinity\\"\\n"]}]}')).toBe('a = "NaN Infinity"\n\n');
	});
});

describe("what is sent stays small and valid", () => {
	it("sends markdown and raw cells without their text, which A never reads", () => {
		const img = "x".repeat(50_000);
		const nb = JSON.stringify({ cells: [{ cell_type: "markdown", source: `![p](data:image/png;base64,${img})` }, { cell_type: "code", source: "y = 2" }] });
		const out = prepareNotebook("md.ipynb", nb);
		if (!out.ok) throw new Error(out.error);
		expect(out.source.files["md.ipynb"].length).toBeLessThan(500);
		expect(notebookToScript(out.source.files["md.ipynb"])).toBe(notebookToScript(nb));
	});

	it("stops text that isn't valid Unicode, which A can't store", () => {
		const nb = JSON.stringify({ cells: [{ cell_type: "code", source: "x = '\ud800'" }] });
		expect(prepareNotebook("s.ipynb", nb)).toEqual({ ok: false, error: "bad_text" });
	});

	it("keeps long names within what A accepts", () => {
		const name = `${"a".repeat(300)}.ipynb`;
		expect(safeNotebookName(name).length).toBeLessThanOrEqual(126);
	});
});

describe("notebooks that can't run", () => {
	it("says when the file isn't a notebook", () => {
		expect(prepareNotebook("a.ipynb", "not json")).toEqual({ ok: false, error: "not_json" });
		expect(prepareNotebook("a.ipynb", '{"nbformat": 4}')).toEqual({ ok: false, error: "no_cells" });
		expect(prepareNotebook("a.py", "{}")).toEqual({ ok: false, error: "not_ipynb" });
	});

	it("says when a cell's source isn't text", () => {
		const bad = JSON.stringify({ cells: [{ cell_type: "code", source: [1, 2] }] });
		expect(prepareNotebook("a.ipynb", bad)).toEqual({ ok: false, error: "cell_not_text" });
	});

	it("stops a notebook with no code before it costs a probe", () => {
		const md = cases.find((c) => c.name === "markdown_only.ipynb") as (typeof cases)[number];
		expect(prepareNotebook(md.name, md.text)).toEqual({ ok: false, error: "no_code" });
	});

	it("stops a script over A's per-file cap", () => {
		const big = JSON.stringify({ cells: [{ cell_type: "code", source: "x = 1\n".repeat(MAX_FILE_BYTES / 6 + 10) }] });
		expect(prepareNotebook("big.ipynb", big)).toEqual({ ok: false, error: "too_big" });
	});
});

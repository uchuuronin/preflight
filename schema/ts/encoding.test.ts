// node --test schema/ts/encoding.test.ts   (Node 22.18+ strips the types itself)
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import { decodeF16, decodeMask, cellIndex } from "./encoding.ts";

const root = new URL("../../", import.meta.url).pathname;

test("decodes the same float16 values Python wrote", () => {
	const rec = JSON.parse(readFileSync(root + "schema/examples/finetune.json", "utf8"));
	const grid = rec.predictions.grids[0];
	const n = grid.axes.reduce((a: number, ax: { values: unknown[] }) => a * ax.values.length, 1);
	const ts = Array.from(decodeF16(grid.point, n, grid.scale));
	const py = JSON.parse(
		execFileSync("uv", ["run", "--quiet", "python", "-c",
			"import json,sys;from preflight.schema.encoding import decode_f16;" +
			"g=json.load(open('schema/examples/finetune.json'))['predictions']['grids'][0];" +
			"print(json.dumps((decode_f16(g['point'])*g['scale']).tolist()))"], { cwd: root, encoding: "utf8" }),
	);
	assert.deepEqual(ts, py);
	// bytes need a scale: a 1.5B model's peak is gigabytes, far past float16's 65504
	assert.ok(grid.scale > 1);
	assert.ok(ts.every(Number.isFinite));
});

test("decodes the abstain mask bit-for-bit", () => {
	const rec = JSON.parse(readFileSync(root + "schema/examples/finetune.json", "utf8"));
	const grid = rec.predictions.grids[0];
	const n = grid.axes.reduce((a: number, ax: { values: unknown[] }) => a * ax.values.length, 1);
	const mask = decodeMask(grid.abstain, n);
	// batch_size 32 x seq_len 4096 is far past 2x the largest probe, so it abstains
	const shape = grid.axes.map((ax: { values: unknown[] }) => ax.values.length);
	assert.equal(mask[cellIndex(shape, [5, 4])], 1);
	// batch_size 1 x seq_len 256 sits inside it
	assert.equal(mask[cellIndex(shape, [0, 0])], 0);
});

test("half floats: specials", () => {
	// 0x7c00 = +inf, 0x3c00 = 1.0, 0x0001 = smallest subnormal
	const b64 = Buffer.from(new Uint8Array([0x00, 0x7c, 0x00, 0x3c, 0x01, 0x00])).toString("base64");
	const out = decodeF16(b64, 3);
	assert.equal(out[0], Infinity);
	assert.equal(out[1], 1);
	assert.equal(out[2], 2 ** -24);
});

import { describe, expect, it } from "vitest";
import { focusView, focusZoom } from "./Orbit";
import type { OrbitNode } from "./orbitLayout";

const node = (id: string, x: number, y: number, size = 110): OrbitNode => ({ id, label: id, graphId: id, phase: null, x, y, size, shape: "circle", share: 0.1, repeat: 1, branch: false });
// A loop like the GNN sample's: the opened node at the top, its two neighbours either side, one
// node far out to the left (the graph store) and one at the bottom.
const layout = { nodes: [node("top", 300, 100), node("left", 180, 140), node("right", 420, 140), node("far", 40, 300), node("bottom", 300, 460)] };
const W = 578;
const H = 540;

function onScreen(v: NonNullable<ReturnType<typeof focusView>>, n: OrbitNode) {
	const sx = (n.x - v.x) * v.zoom + W / 2;
	const sy = (n.y - v.y) * v.zoom + H / 2;
	const r = (n.size / 2) * v.zoom;
	return { l: sx - r, r: sx + r, t: sy - r, b: sy + r };
}

describe("where a focused node's zoom lands (D's audit S2)", () => {
	it("shows the opened node and its two nearest neighbours whole, inside the soft edge", () => {
		const v = focusView(layout, "top", W, H)!;
		for (const id of ["top", "left", "right"]) {
			const s = onScreen(v, layout.nodes.find((n) => n.id === id)!);
			expect(s.l).toBeGreaterThanOrEqual(32);
			expect(s.r).toBeLessThanOrEqual(W - 32);
			expect(s.t).toBeGreaterThanOrEqual(32);
			expect(s.b).toBeLessThanOrEqual(H - 32);
		}
		expect(v.off.has("left") || v.off.has("right") || v.off.has("top")).toBe(false);
	});
	it("still zooms in, never past the focus zoom, and never out past the whole loop", () => {
		const v = focusView(layout, "top", W, H)!;
		const fitted = Math.min(W / 600, H / 560) / 1.08;
		expect(v.zoom).toBeGreaterThanOrEqual(fitted);
		expect(v.zoom).toBeLessThanOrEqual(focusZoom(W, H));
	});
	it("hides a node that would sit cut in half at the edge, and keeps the opened one", () => {
		const v = focusView(layout, "top", W, H)!;
		expect(v.off.has("far")).toBe(true);
		expect(v.off.has("top")).toBe(false);
	});
	it("knows no view for a node that isn't on the map", () => {
		expect(focusView(layout, "nope", W, H)).toBeNull();
	});
});

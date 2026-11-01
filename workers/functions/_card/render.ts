/**
 * SVG to PNG with resvg (WebAssembly, vendored unmodified under MPL-2.0 in vendor/resvg). The
 * wasm and the fonts are passed in, so this runs the same in a Pages Function (assets.ts imports
 * them as modules) and in Node tests (read from disk). resvg is initialised once per isolate.
 */
import { initWasm, Resvg } from "./vendor/resvg/index.mjs";

let ready: Promise<void> | null = null;

export function init(wasm: WebAssembly.Module | BufferSource): Promise<void> {
	const started =
		ready ??
		initWasm(wasm).catch((e: unknown) => {
			ready = null;
			throw e;
		});
	ready = started;
	return started;
}

export async function renderPng(svg: string, wasm: WebAssembly.Module | BufferSource, fonts: Uint8Array[], width: number): Promise<Uint8Array<ArrayBuffer>> {
	await init(wasm);
	const r = new Resvg(svg, {
		fitTo: { mode: "width", value: width },
		font: { fontBuffers: fonts, loadSystemFonts: false, defaultFontFamily: "Instrument Sans" },
	});
	const img = r.render();
	const png = new Uint8Array(img.asPng());
	img.free();
	r.free();
	return png;
}

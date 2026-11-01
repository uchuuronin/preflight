/**
 * The renderer's binary assets as Workers modules: Wrangler bundles .wasm as a compiled
 * WebAssembly.Module and .bin as an ArrayBuffer (its default module rules), so nothing is
 * fetched or compiled at request time.
 */
import resvgWasm from "./vendor/resvg/index_bg.wasm";
import sans400 from "./fonts/InstrumentSans-400.bin";
import sans600 from "./fonts/InstrumentSans-600.bin";
import serif400 from "./fonts/InstrumentSerif-400.bin";
import mono500 from "./fonts/JetBrainsMono-500.bin";

export const WASM: WebAssembly.Module = resvgWasm;
export const FONTS: Uint8Array[] = [sans400, sans600, serif400, mono500].map((b) => new Uint8Array(b));

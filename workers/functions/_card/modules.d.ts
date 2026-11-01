/** Wrangler's default module rules: .wasm imports as a compiled module, .bin as raw bytes. */
declare module "*.wasm" {
	const module: WebAssembly.Module;
	export default module;
}
declare module "*.bin" {
	const bytes: ArrayBuffer;
	export default bytes;
}

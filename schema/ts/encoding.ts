// Decoders for grid payloads. Mirrors python/preflight/schema/encoding.py:
// base64 little-endian IEEE float16 (real value = decoded x grid.scale), and LSB-first bit masks.
// Hand-written (not generated); the parity test in schema/ts/encoding.test.ts keeps it honest.

export const MAX_GRID_CELLS = 16_384;

function b64ToBytes(data: string): Uint8Array {
	if (typeof atob === "function") {
		const bin = atob(data);
		const out = new Uint8Array(bin.length);
		for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
		return out;
	}
	// Node without atob
	return new Uint8Array((globalThis as any).Buffer.from(data, "base64"));
}

function halfToFloat(h: number): number {
	const sign = h & 0x8000 ? -1 : 1;
	const exp = (h >> 10) & 0x1f;
	const frac = h & 0x3ff;
	if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
	if (exp === 31) return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
	return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

/** Decoded float16 values multiplied by the grid's scale (a power of two). */
export function decodeF16(data: string, n?: number, scale = 1): Float64Array {
	const bytes = b64ToBytes(data);
	const count = bytes.length >> 1;
	if (n !== undefined && count !== n) throw new Error(`expected ${n} float16 values, got ${count}`);
	const out = new Float64Array(count);
	for (let i = 0; i < count; i++) out[i] = halfToFloat(bytes[2 * i] | (bytes[2 * i + 1] << 8)) * scale;
	return out;
}

export function decodeMask(data: string, n: number): Uint8Array {
	const bytes = b64ToBytes(data);
	if (bytes.length !== (n + 7) >> 3) throw new Error(`mask holds ${bytes.length * 8} bits, expected ${n}`);
	const out = new Uint8Array(n);
	for (let i = 0; i < n; i++) out[i] = (bytes[i >> 3] >> (i & 7)) & 1;
	return out;
}

/** Row-major flat index for one value index per axis. */
export function cellIndex(shape: readonly number[], idx: readonly number[]): number {
	let flat = 0;
	for (let a = 0; a < shape.length; a++) flat = flat * shape[a] + idx[a];
	return flat;
}

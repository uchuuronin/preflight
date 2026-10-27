/** Decoding for the record's base64 float16 grids and uint8 masks. */

export function base64ToBytes(b64: string): Uint8Array {
	const bin = atob(b64);
	const out = new Uint8Array(bin.length);
	for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
	return out;
}

export function halfToFloat(h: number): number {
	const sign = h & 0x8000 ? -1 : 1;
	const exp = (h >> 10) & 0x1f;
	const frac = h & 0x03ff;
	if (exp === 0) return sign * 2 ** -14 * (frac / 1024);
	if (exp === 0x1f) return frac ? Number.NaN : sign * Number.POSITIVE_INFINITY;
	return sign * 2 ** (exp - 15) * (1 + frac / 1024);
}

/** Little-endian float16 array, as numpy's astype("<f2").tobytes() writes it. */
export function decodeFloat16(b64: string): Float32Array {
	const bytes = base64ToBytes(b64);
	const n = bytes.length >> 1;
	const out = new Float32Array(n);
	for (let i = 0; i < n; i++) {
		const lo = bytes[2 * i] ?? 0;
		const hi = bytes[2 * i + 1] ?? 0;
		out[i] = halfToFloat(lo | (hi << 8));
	}
	return out;
}

export function decodeMask(b64: string): Uint8Array {
	return base64ToBytes(b64);
}

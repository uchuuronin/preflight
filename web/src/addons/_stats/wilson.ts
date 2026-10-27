/**
 * Wilson score interval for a proportion (Wilson 1927), the contract's rule for reporting
 * proportions. z = 1.959964 for 95% [derived: the normal quantile at 0.975].
 */
export const Z95 = 1.959964;

export interface Proportion {
	k: number;
	n: number;
	p: number;
	lo: number;
	hi: number;
}

export function wilson(k: number, n: number, z = Z95): Proportion {
	if (!(n > 0) || k < 0 || k > n) throw new RangeError(`wilson(${k}, ${n}): need 0 <= k <= n and n > 0`);
	const p = k / n;
	const d = 1 + (z * z) / n;
	const centre = (p + (z * z) / (2 * n)) / d;
	const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / d;
	return { k, n, p, lo: Math.max(0, centre - half), hi: Math.min(1, centre + half) };
}

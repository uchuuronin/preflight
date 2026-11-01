/**
 * Reads a flight record by its share token from the API, the way C's /v1 proxy does
 * (web/functions/_lib/proxy.ts): straight to PREFLIGHT_API_ORIGIN with the edge secret, so the
 * API's own checks and quotas apply. Only GET, only a well-formed token.
 */
import type { RecordLike } from "./headline";

export interface CardEnv {
	PREFLIGHT_API_ORIGIN?: string;
	PREFLIGHT_EDGE_SECRET?: string;
}

/** Share tokens are URL-safe base64 (22 characters today); anything else is refused before a fetch. */
export const TOKEN = /^[A-Za-z0-9_-]{8,64}$/;

export type Fetched = { ok: true; record: RecordLike } | { ok: false; status: number };

export async function fetchRecord(token: string, request: Request, env: CardEnv, fetchImpl: typeof fetch = fetch): Promise<Fetched> {
	if (!TOKEN.test(token)) return { ok: false, status: 404 };
	if (!env.PREFLIGHT_API_ORIGIN || !env.PREFLIGHT_EDGE_SECRET) return { ok: false, status: 503 };
	const url = new URL(`/v1/records/${encodeURIComponent(token)}`, env.PREFLIGHT_API_ORIGIN);
	const res = await fetchImpl(url.toString(), {
		headers: {
			accept: "application/json",
			"X-Preflight-Edge": env.PREFLIGHT_EDGE_SECRET,
			"X-Preflight-Client-IP": request.headers.get("CF-Connecting-IP") ?? "",
		},
	});
	if (!res.ok) return { ok: false, status: res.status === 404 ? 404 : 502 };
	try {
		return { ok: true, record: (await res.json()) as RecordLike };
	} catch {
		return { ok: false, status: 502 };
	}
}

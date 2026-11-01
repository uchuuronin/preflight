/**
 * The fix race over /v1/patches (api/preflight_api/routes/patches.py): start, follow, cancel.
 * Every number shown comes from the API's answer: the cap it reserved, what Muse Code reported.
 */
import type { Candidate } from "./candidates";

export type CandidateStatus = "pending" | "patched" | "no_change" | "refused" | "timeout" | "over_budget" | "failed" | "leaked";

export interface RaceCandidate {
	id: string;
	template_id: string;
	value: unknown;
	status: CandidateStatus;
	diff: string;
	files: string[];
	final: string | null;
	seconds: number;
	problem: string | null;
	blocked: string[];
}

export interface RaceView {
	patch_id: string;
	status: "running" | "done" | "failed";
	candidates: RaceCandidate[];
	spent_usd?: number | null;
	error?: string;
	cap_usd?: number;
	candidates_asked?: Candidate[];
}

export type RaceStart = { ok: true; race: RaceView } | { ok: false; code: string; message: string };

// Source: derived, a candidate takes about 20 to 45 s (the two smoke runs, docs/status/c.md);
// a look every 2 s shows each finish within a couple of seconds without crowding the API.
export const POLL_MS = 2000;

async function json(res: Response): Promise<Record<string, unknown> | null> {
	return (await res.json().catch(() => null)) as Record<string, unknown> | null;
}

export async function startRace(record: string, finding: string | null, candidates: Candidate[], post: typeof fetch = fetch): Promise<RaceStart> {
	try {
		const res = await post("/v1/patches", {
			method: "POST",
			credentials: "same-origin",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ record, candidates, ...(finding ? { finding } : {}) }),
		});
		const body = await json(res);
		if (!res.ok || !body) {
			const err = (body?.error ?? {}) as { code?: string; message?: string };
			return { ok: false, code: err.code ?? `http_${res.status}`, message: err.message ?? "" };
		}
		return { ok: true, race: body as unknown as RaceView };
	} catch {
		return { ok: false, code: "unreachable", message: "" };
	}
}

export async function getRace(id: string, get: typeof fetch = fetch): Promise<RaceView | null> {
	try {
		const res = await get(`/v1/patches/${encodeURIComponent(id)}`, { credentials: "same-origin" });
		return res.ok ? ((await json(res)) as unknown as RaceView) : null;
	} catch {
		return null;
	}
}

export async function cancelRace(id: string, post: typeof fetch = fetch): Promise<RaceView | null> {
	try {
		const res = await post(`/v1/patches/${encodeURIComponent(id)}/cancel`, { method: "POST", credentials: "same-origin" });
		return res.ok ? ((await json(res)) as unknown as RaceView) : null;
	} catch {
		return null;
	}
}

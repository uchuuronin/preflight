/**
 * Where a record comes from: ?share=<token> reads it from the API; otherwise the
 * development fixture loads (lazily, so it never sits in the main bundle).
 */
import { t } from "../../lib/i18n";
import type { FlightRecord } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export async function loadRecord(params: URLSearchParams): Promise<FlightRecord> {
	// Share links carry only the random share token; the record's ULID never appears in a
	// URL because it leaks creation time. The API resolves the token to the record.
	const share = params.get("share");
	if (share) {
		const res = await fetch(`${API_BASE}/v1/records/${encodeURIComponent(share)}`);
		if (!res.ok) {
			const body = (await res.json().catch(() => null)) as { error?: { message?: string } } | null;
			throw new Error(body?.error?.message ?? t("app.record_unavailable"));
		}
		return (await res.json()) as FlightRecord;
	}
	const mod = await import("../fixtures/handmade-finetune.record.json");
	return mod.default as unknown as FlightRecord;
}

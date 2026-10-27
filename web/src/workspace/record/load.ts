/**
 * Where records come from. Samples ship with the app (the golden records and C's dev record)
 * and cost nothing to open; anything else is read from the API by its share token. A share
 * link never carries the record's ULID, because a ULID leaks its creation time.
 */
import { t } from "../../lib/i18n";
import type { FlightRecord, Lens, LoadedRecord } from "./types";
import { normaliseRecord } from "./types";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";

export interface Sample {
	id: string;
	code: string;
	titleKey: string;
	lens: Lens;
	gpu: string;
	load: () => Promise<FlightRecord>;
}

export const SAMPLES: Sample[] = [
	{
		id: "lora-7b",
		code: "PF101",
		titleKey: "sample.lora_7b",
		lens: "finetune",
		gpu: "A100",
		load: () => import("../fixtures/dev-finetune.record.json").then((m) => m.default as unknown as FlightRecord),
	},
	{
		id: "lora-qwen",
		code: "PF102",
		titleKey: "sample.lora_qwen",
		lens: "finetune",
		gpu: "L4",
		load: () => import("../../../../schema/examples/finetune.json").then((m) => m.default as unknown as FlightRecord),
	},
	{
		id: "pretrain",
		code: "PF412",
		titleKey: "sample.pretrain",
		lens: "pretrain",
		gpu: "L4",
		load: () => import("../../../../schema/examples/pretrain.json").then((m) => m.default as unknown as FlightRecord),
	},
	{
		id: "agent",
		code: "PF330",
		titleKey: "sample.agent",
		lens: "agent",
		gpu: "API",
		load: () => import("../../../../schema/examples/agent.json").then((m) => m.default as unknown as FlightRecord),
	},
];

export const DEFAULT_SAMPLE = "lora-7b";

export async function loadRecord(ref: string): Promise<LoadedRecord> {
	const sample = SAMPLES.find((s) => s.id === ref);
	if (sample) return normaliseRecord(await sample.load());
	const res = await fetch(`${API_BASE}/v1/records/${encodeURIComponent(ref)}`, { credentials: "include" });
	if (!res.ok) {
		const body = (await res.json().catch(() => null)) as { error?: { message?: string } } | null;
		throw new Error(body?.error?.message ?? t("app.record_unavailable"));
	}
	return normaliseRecord((await res.json()) as FlightRecord);
}

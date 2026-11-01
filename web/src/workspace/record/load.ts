/**
 * Where records come from. Samples ship with the app (the golden records and C's dev record)
 * and cost nothing to open; anything else is read from the API by its share token. A share
 * link never carries the record's ULID, because a ULID leaks its creation time.
 */
import { t } from "../../lib/i18n";
import type { FlightRecord, Lens, LoadedRecord } from "./types";
import { normaliseRecord } from "./types";

/** Same origin: /v1 is served by the web app's own Pages Function (functions/v1). */
const API_BASE = "";
/** Source: derived, the API asks for 2 s after a busy record store (routes/records.py); 5 s caps a larger ask. */
const RETRY_CAP_MS = 5_000;

const sleep = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export interface Sample {
	id: string;
	code: string;
	titleKey: string;
	lens: Lens;
	gpu: string;
	/** What the sample's numbers are: a fixture's illustrative ones, or a real run's (tested against isFixture). */
	numbers: "illustrative" | "measured";
	load: () => Promise<FlightRecord>;
}

export const SAMPLES: Sample[] = [
	{
		id: "lora-7b",
		code: "PF101",
		titleKey: "sample.lora_7b",
		lens: "finetune",
		gpu: "A100",
		numbers: "illustrative",
		load: () => import("../fixtures/dev-finetune.record.json").then((m) => m.default as unknown as FlightRecord),
	},
	{
		id: "lora-qwen",
		code: "PF102",
		titleKey: "sample.lora_qwen",
		lens: "finetune",
		gpu: "L4",
		numbers: "illustrative",
		load: () => import("../../../../schema/examples/finetune.json").then((m) => m.default as unknown as FlightRecord),
	},
	{
		id: "pretrain",
		code: "PF412",
		titleKey: "sample.pretrain",
		lens: "pretrain",
		gpu: "L4",
		numbers: "illustrative",
		load: () => import("../../../../schema/examples/pretrain.json").then((m) => m.default as unknown as FlightRecord),
	},
	{
		id: "agent",
		code: "PF330",
		titleKey: "sample.agent",
		lens: "agent",
		gpu: "API",
		numbers: "illustrative",
		load: () => import("../../../../schema/examples/agent.json").then((m) => m.default as unknown as FlightRecord),
	},
	{
		// A's real run of zoo/agent_eval_spark on Muse Spark (Oct 31, 30 of 500 tasks sampled), with
		// its share token removed: measured numbers, not illustrative ones.
		id: "agent-eval",
		code: "PF331",
		titleKey: "sample.agent_eval",
		lens: "agent",
		gpu: "API",
		numbers: "measured",
		load: () => import("../fixtures/agent-eval-oct31.record.json").then((m) => m.default as unknown as FlightRecord),
	},
	{
		// The pretrain golden record with a campaign block from B's planner on a 12-run research
		// grid (web/scripts/make_campaign_sample.py): illustrative numbers, every one B's.
		id: "campaign-grid",
		code: "PF420",
		titleKey: "sample.campaign_grid",
		lens: "pretrain",
		gpu: "A100",
		numbers: "illustrative",
		load: () => import("../fixtures/campaign-grid.record.json").then((m) => m.default as unknown as FlightRecord),
	},
	{
		// A's L4 smoke run of zoo/grpo_trl (Nov 1, GitHub run 41260610897) as a record: A's map and
		// measured probe, B's predictions (web/scripts/make_grpo_sample.py). Measured, not illustrative.
		id: "grpo",
		code: "PF510",
		titleKey: "sample.grpo_l4",
		lens: "rl",
		gpu: "L4",
		numbers: "measured",
		load: () => import("../fixtures/grpo-l4.record.json").then((m) => m.default as unknown as FlightRecord),
	},
];

export const DEFAULT_SAMPLE = "lora-7b";

export async function loadRecord(ref: string): Promise<LoadedRecord> {
	const sample = SAMPLES.find((s) => s.id === ref);
	if (sample) return normaliseRecord(await sample.load());
	const url = `${API_BASE}/v1/records/${encodeURIComponent(ref)}`;
	let res = await fetch(url, { credentials: "same-origin" });
	if (res.status === 503) {
		// The record store was mid-reload (a retryable 503): wait as told, at most RETRY_CAP_MS, once.
		const after = Number(res.headers.get("Retry-After"));
		await sleep(Math.min(RETRY_CAP_MS, Number.isFinite(after) && after > 0 ? after * 1000 : 1000));
		res = await fetch(url, { credentials: "same-origin" });
	}
	if (!res.ok) {
		const body = (await res.json().catch(() => null)) as { error?: { message?: string } } | null;
		throw new Error(body?.error?.message ?? t("app.record_unavailable"));
	}
	return normaliseRecord((await res.json()) as FlightRecord);
}

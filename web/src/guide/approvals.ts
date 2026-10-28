/**
 * Spending tools the visitor approved, for the rule-based tour (the live guide approves inside
 * its own conversation, see liveGuide.ts). POST /v1/guide/tools runs the tool on the server
 * (api/preflight_api/routes/guide.py): a probe, a verify run or more agent tasks go to the job
 * store; the answer says what actually happened.
 */
import type { KnobValue } from "../workspace/record/types";

export type SpendTool = "run_probe" | "run_verify" | "sample_more" | "propose_patch";

export interface ToolRun {
	status: "submitted" | "unavailable" | "queue_full" | "refused" | "failed";
	job_id?: string;
	queue_position?: number | null;
	reason?: string;
	retry_after_s?: number | null;
}

export async function runApprovedTool(
	record: string,
	tool: SpendTool,
	args: { knobs?: Record<string, KnobValue>; knob_id?: string; reason?: string },
	setting: Record<string, KnobValue>,
	post: typeof fetch = fetch,
): Promise<ToolRun> {
	try {
		const res = await post("/v1/guide/tools", {
			method: "POST",
			credentials: "same-origin",
			headers: { "Content-Type": "application/json" },
			body: JSON.stringify({ record, tool, setting, ...args }),
		});
		const body = (await res.json().catch(() => null)) as (ToolRun & { error?: { message?: string } }) | null;
		if (!res.ok || !body?.status) return { status: "failed", reason: body?.error?.message ?? `request failed (${res.status})` };
		return body;
	} catch {
		return { status: "unavailable", reason: "server_unreachable" };
	}
}

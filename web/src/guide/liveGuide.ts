/**
 * The live guide loop. The browser holds the conversation, sends it with this turn's tool
 * definitions to /v1/guide/turns, runs the function calls that come back (UI tools here,
 * money-spending tools only queued for the person's click), and sends the results until
 * the model stops calling tools. A 503 with fallback=rule_based hands over to the
 * rule-based tour.
 */
import { buildTools, executeTool, type ToolContext } from "./tools";

const API_BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? "";
const MAX_ROUNDS = 6;

export type InputItem = Record<string, unknown>;

export interface TurnResponse {
	text: string;
	calls: Array<{ call_id: string; name: string; arguments: Record<string, unknown> }>;
	provenance: Record<string, unknown>;
}

export class GuideFallback extends Error {
	constructor(
		message: string,
		readonly reason: string,
	) {
		super(message);
	}
}

export type Poster = (body: unknown) => Promise<Response>;

const defaultPost: Poster = (body) =>
	fetch(`${API_BASE}/v1/guide/turns`, {
		method: "POST",
		credentials: "include",
		headers: { "Content-Type": "application/json" },
		body: JSON.stringify(body),
	});

export interface LiveTurnResult {
	text: string;
	history: InputItem[];
	provenance: Array<Record<string, unknown>>;
}

export async function runLiveTurn(
	history: InputItem[],
	userText: string,
	ctx: () => ToolContext,
	opts: { recordRef?: string; post?: Poster; onText?: (t: string) => void } = {},
): Promise<LiveTurnResult> {
	const post = opts.post ?? defaultPost;
	const items: InputItem[] = [...history, { role: "user", content: userText }];
	const provenance: Array<Record<string, unknown>> = [];
	let text = "";
	for (let round = 0; round < MAX_ROUNDS; round++) {
		const c = ctx();
		// Rebuilt every round: a pin made mid-turn removes that knob from set_knob at once.
		const tools = buildTools(c.record, c.pinned);
		const res = await post({ input: items, tools, record_ref: opts.recordRef, effort: "low" });
		if (!res.ok) {
			const body = (await res.json().catch(() => null)) as { error?: { message?: string; fallback?: string; reason?: string } } | null;
			if (body?.error?.fallback === "rule_based") throw new GuideFallback(body.error.message ?? "", body.error.reason ?? "unknown");
			throw new GuideFallback(body?.error?.message ?? `Guide request failed (${res.status}).`, "http_error");
		}
		const turn = (await res.json()) as TurnResponse;
		provenance.push(turn.provenance);
		if (turn.text) {
			text = turn.text;
			opts.onText?.(text);
			items.push({ role: "assistant", content: turn.text });
		}
		if (!turn.calls.length) break;
		for (const call of turn.calls) {
			const result = executeTool(call.name, call.arguments, ctx());
			items.push({ type: "function_call", call_id: call.call_id, name: call.name, arguments: JSON.stringify(call.arguments) });
			items.push({ type: "function_call_output", call_id: call.call_id, output: JSON.stringify(result) });
		}
	}
	return { text, history: items, provenance };
}

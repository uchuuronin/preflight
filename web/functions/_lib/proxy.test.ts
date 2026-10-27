import { describe, expect, it, vi } from "vitest";
import { proxy } from "./proxy";

const env = { PREFLIGHT_API_ORIGIN: "https://api.example.modal.run", PREFLIGHT_EDGE_SECRET: "s".repeat(40) };

function capture() {
	const calls: Array<{ url: string; init: RequestInit }> = [];
	const fetchImpl = vi.fn(async (url: string, init: RequestInit) => {
		calls.push({ url, init });
		return new Response('{"ok":true}', { status: 200, headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": "*" } });
	}) as unknown as typeof fetch;
	return { calls, fetchImpl };
}

describe("same-site API proxy", () => {
	it("forwards /v1 with the edge secret and the visitor's IP, and strips forged headers", async () => {
		const { calls, fetchImpl } = capture();
		const req = new Request("https://preflight.pages.dev/v1/guide/turns?x=1", {
			method: "POST",
			body: "{}",
			headers: { "CF-Connecting-IP": "203.0.113.9", "X-Preflight-Edge": "forged", "X-Preflight-Client-IP": "1.2.3.4", Cookie: "__Host-pf_session=v1.a.b" },
		});
		const res = await proxy(req, env, fetchImpl);
		expect(res.status).toBe(200);
		expect(res.headers.get("access-control-allow-origin")).toBeNull();
		expect(calls[0]?.url).toBe("https://api.example.modal.run/v1/guide/turns?x=1");
		const h = new Headers(calls[0]?.init.headers);
		expect(h.get("X-Preflight-Edge")).toBe(env.PREFLIGHT_EDGE_SECRET);
		expect(h.get("X-Preflight-Client-IP")).toBe("203.0.113.9");
		expect(h.get("Cookie")).toBe("__Host-pf_session=v1.a.b");
	});

	it("never forwards outside /v1 and says so when the API isn't configured", async () => {
		const { fetchImpl } = capture();
		expect((await proxy(new Request("https://x.dev/admin"), env, fetchImpl)).status).toBe(404);
		const off = await proxy(new Request("https://x.dev/v1/status"), { PREFLIGHT_API_ORIGIN: "", PREFLIGHT_EDGE_SECRET: "" }, fetchImpl);
		expect(off.status).toBe(503);
	});
});

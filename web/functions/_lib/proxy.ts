/**
 * Same-site API: the web app's origin serves /v1/* by forwarding to the API on Modal, so the
 * session cookie is first-party (`__Host-`, SameSite=Lax) and there is no cross-site CORS.
 *
 * The edge adds what the API trusts and strips what a visitor could forge:
 * - X-Preflight-Edge: the shared secret (PREFLIGHT_EDGE_SECRET), so the Modal URL refuses
 *   direct calls that would skip this hop;
 * - X-Preflight-Client-IP: Cloudflare's CF-Connecting-IP, for the API's IP bucket.
 * Bodies and responses stream through, so the job events stream (SSE) works unchanged.
 */
export interface ProxyEnv {
	PREFLIGHT_API_ORIGIN: string;
	PREFLIGHT_EDGE_SECRET: string;
}

const HOP_BY_HOP = ["host", "connection", "keep-alive", "transfer-encoding", "upgrade", "proxy-authorization", "te", "trailer"];
const FORGEABLE = ["x-preflight-edge", "x-preflight-client-ip", "x-forwarded-for", "x-real-ip", "cf-connecting-ip-forged"];

export async function proxy(request: Request, env: ProxyEnv, fetchImpl: typeof fetch = fetch): Promise<Response> {
	if (!env.PREFLIGHT_API_ORIGIN || !env.PREFLIGHT_EDGE_SECRET) {
		return Response.json(
			{ error: { code: "unavailable", message: "Preflight's server isn't connected yet. Samples still work.", retryable: true } },
			{ status: 503 },
		);
	}
	const url = new URL(request.url);
	if (!url.pathname.startsWith("/v1/")) return new Response("Not found", { status: 404 });
	const target = new URL(url.pathname + url.search, env.PREFLIGHT_API_ORIGIN);
	const headers = new Headers(request.headers);
	for (const h of [...HOP_BY_HOP, ...FORGEABLE]) headers.delete(h);
	headers.set("X-Preflight-Edge", env.PREFLIGHT_EDGE_SECRET);
	headers.set("X-Preflight-Client-IP", request.headers.get("CF-Connecting-IP") ?? "");
	const hasBody = !["GET", "HEAD"].includes(request.method);
	const res = await fetchImpl(target.toString(), {
		method: request.method,
		headers,
		body: hasBody ? request.body : undefined,
		redirect: "manual",
		// @ts-expect-error: streaming request bodies need half-duplex in fetch implementations that check it
		duplex: hasBody ? "half" : undefined,
	});
	const out = new Headers(res.headers);
	out.delete("access-control-allow-origin");
	out.delete("access-control-allow-credentials");
	return new Response(res.body, { status: res.status, statusText: res.statusText, headers: out });
}

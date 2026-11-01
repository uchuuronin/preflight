/**
 * The three share routes, as plain functions over (request, env, token) so tests can call them
 * without the Pages runtime. The files under s/, card/ and badge/ are thin Pages wrappers.
 *
 *   GET /s/<token>      a page with link-preview tags (title, description, the card image) that
 *                       sends a person on to the cockpit, #/cockpit/<token>
 *   GET /card/<token>   the 1200 x 630 PNG card
 *   GET /badge/<token>  the README badge (SVG)
 *
 * A running job revises its record, so responses are cached for two minutes only, and an API outage
 * is never cached (no-store), so no link preview keeps a generic card.
 */
import { headline, type Headline } from "./headline";
import { HERO_CARDS } from "./hero";
import { type CardEnv, type Fetched, fetchRecord, TOKEN } from "./record";
import { badgeSvg, CARD_W, cardSvg, esc } from "./svg";

export interface Renderer {
	png(svg: string): Promise<Uint8Array<ArrayBuffer>>;
}

const CACHE = "public, max-age=120";

/** The response status and caching for what the API answered: a record or a 404 is cached for two
 * minutes; an API outage (502, 503) is never cached, so no link preview keeps a generic card. */
function outcome(got: Fetched): { status: number; cache: string } {
	if (got.ok) return { status: 200, cache: CACHE };
	if (got.status === 404) return { status: 404, cache: CACHE };
	return { status: got.status, cache: "no-store" };
}

const NONE: Headline = { title: "A Preflight flight record", subtitle: "", lines: [], illustrative: false };

async function load(request: Request, env: CardEnv, token: string, fetchImpl?: typeof fetch): Promise<{ h: Headline; got: Fetched }> {
	const got = await fetchRecord(token, request, env, fetchImpl);
	return { h: got.ok ? headline(got.record) : NONE, got };
}

function describe(h: Headline): string {
	const parts = h.lines.map((l) => `${l.label}: ${l.value} (${l.status.toLowerCase()})`);
	return parts.length ? parts.join(" · ") : "A flight record from Preflight, the flight simulator for ML jobs.";
}

export async function sharePage(request: Request, env: CardEnv, token: string, fetchImpl?: typeof fetch): Promise<Response> {
	const origin = new URL(request.url).origin;
	const safe = TOKEN.test(token) ? token : "";
	const { h, got } = await load(request, env, token, fetchImpl);
	const cockpit = safe ? `${origin}/#/cockpit/${safe}` : `${origin}/`;
	const missing = !got.ok && got.status === 404;
	const out = outcome(got);
	const title = missing ? "No flight record matches this link" : `${h.title} · Preflight`;
	const desc = missing ? "Check the link, or open a sample in Preflight." : describe(h);
	// A hero record has a card pre-rendered as a plain file (scripts/hero-cards.ts), which previews
	// with an image on the Free plan; others get the rendered card only where the deploy allows it.
	const image = !safe ? "" : HERO_CARDS.has(safe) ? `${origin}/cards/${safe}.png` : env.PREFLIGHT_SHARE_CARD_IMAGES === "on" ? `${origin}/card/${safe}` : "";
	const html = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(title)}</title>
<meta name="description" content="${esc(desc)}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="Preflight">
<meta property="og:title" content="${esc(title)}">
<meta property="og:description" content="${esc(desc)}">
<meta property="og:url" content="${esc(`${origin}/s/${safe}`)}">
${image && got.ok ? `<meta property="og:image" content="${esc(image)}">\n<meta property="og:image:width" content="1200">\n<meta property="og:image:height" content="630">\n<meta property="og:image:alt" content="${esc(desc)}">\n<meta name="twitter:card" content="summary_large_image">` : '<meta name="twitter:card" content="summary">'}
<meta http-equiv="refresh" content="0; url=${esc(cockpit)}">
<style>body{margin:0;min-height:100vh;display:grid;place-items:center;background:#0E2634;color:#FFF5D1;font:16px/1.5 system-ui,sans-serif;padding:16px}a{color:#FFF5D1}</style>
</head>
<body>
<p>${missing ? "No flight record matches this link. " : ""}<a href="${esc(cockpit)}">${missing ? "Open Preflight" : "Open this flight in Preflight"}</a></p>
</body>
</html>`;
	return new Response(html, {
		status: out.status,
		headers: { "content-type": "text/html; charset=utf-8", "cache-control": out.cache, "x-robots-tag": got.ok ? "all" : "noindex" },
	});
}

export async function cardPng(request: Request, env: CardEnv, token: string, renderer: Renderer, fetchImpl?: typeof fetch): Promise<Response> {
	// Rendering per request is off unless the deploy turns it on (it passes the Free plan's CPU
	// limit): a hero record goes to its pre-rendered file, anything else is not served.
	if (env.PREFLIGHT_SHARE_CARD_IMAGES !== "on") {
		if (TOKEN.test(token) && HERO_CARDS.has(token)) return Response.redirect(new URL(`/cards/${token}.png`, request.url).toString(), 302);
		return new Response("Not found", { status: 404, headers: { "cache-control": CACHE } });
	}
	const { h, got } = await load(request, env, token, fetchImpl);
	const png = await renderer.png(cardSvg(h, new URL(request.url).host));
	const out = outcome(got);
	return new Response(png, { status: out.status, headers: { "content-type": "image/png", "cache-control": out.cache } });
}

export async function badge(request: Request, env: CardEnv, token: string, fetchImpl?: typeof fetch): Promise<Response> {
	const { h, got } = await load(request, env, token, fetchImpl);
	const out = outcome(got);
	return new Response(badgeSvg(h), { status: out.status, headers: { "content-type": "image/svg+xml; charset=utf-8", "cache-control": out.cache } });
}

export { CARD_W };

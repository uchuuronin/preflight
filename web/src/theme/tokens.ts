/**
 * Theme tokens. Every colour, space, radius, type size and easing in the app comes from here,
 * read as CSS variables (var(--pf-<name>)), so switching sets never re-renders React.
 *
 * The look follows D's "Preflight glass workspace" design: glass panels over one tall sky
 * gradient that climbs as you move through the screens, in the Limoncello palette (Rhino,
 * Barley, Naples, Sky, Neptune, Light olive, Coral). Two sets: "dusk" (dark, the default)
 * and "day" (light). D may add or override sets with registerTokenSet from an add-on.
 *
 * Reserved meanings: `fail` is the only red in the app (red at dusk, maroon by day, with a
 * maroon tint behind failing parts at dusk), so anything red-toned means something will
 * break; it always ships with an icon and a word. ok / tight are fit status, also with icon
 * and word. Memory-term colours (term-1..5) follow D's mockup; term-6 (generation memory:
 * KV cache and the inference engine, RL lens) is a lavender outside the mockup.
 */

const shared = {
	"font-serif": "'Instrument Serif', Georgia, 'Times New Roman', serif",
	"font-sans": "'Instrument Sans', system-ui, -apple-system, 'Segoe UI', sans-serif",
	"font-mono": "'JetBrains Mono', ui-monospace, Menlo, monospace",

	"text-xs": "11px",
	"text-sm": "12.5px",
	"text-md": "14.5px",
	"text-lg": "17px",
	"text-xl": "22px",
	"text-2xl": "30px",
	"text-3xl": "46px",

	"space-1": "4px",
	"space-2": "8px",
	"space-3": "12px",
	"space-4": "16px",
	"space-5": "22px",
	"space-6": "32px",
	"space-7": "48px",

	"radius-sm": "12px",
	"radius-md": "18px",
	"radius-lg": "28px",
	"radius-pill": "999px",

	blur: "16px",
	"blur-node": "10px",
	ease: "cubic-bezier(.23, 1, .32, 1)",
	"ease-move": "cubic-bezier(.77, 0, .175, 1)",
	"dur-press": "160ms",
	"dur-quick": "250ms",
	"dur-value": "600ms",
	"dur-sheet": "450ms",
	"dur-screen": "700ms",
	"dur-sky": "1400ms",
	// The map's glide (nodes, rings, arcs) and the core number travelling with it (lib/tween.ts).
	"dur-move": "800ms",
	// How long a move's wake stays on the readouts it changed (Dials.tsx).
	"dur-wake": "2600ms",
} as const;

/**
 * Dusk (the dark sky), from Chiya's reference (Nov 1): a clear sunset over cloud, indigo overhead
 * through violet and pink to peach at the horizon (#563A88, #8F468A, #DF6E94, #F5BAAF, #070507).
 * Clear, not murky: saturated stops, an indigo glass a little more solid than before so text never
 * sits on haze, and red-orange for alerts. The cockpit sees the sky from about 12% to 64% of this
 * gradient and the home screen from 47% (Shell.tsx SKY_Y), so the vivid pink and peach are kept to
 * the last few percent, under the home screen's last line: text on the open sky stays on violet
 * (fg 5.7:1 on #8F468A, where the home screen's lede sits). Contrast on the glass over the lowest
 * sky the cockpit shows (#7A4389): fg 11.7:1, fg-dim 7.4:1, fail 4.9:1, ok 8.3:1, tight 9.6:1
 * (WCAG formula, alpha composited).
 */
export const duskTokens = {
	...shared,
	"color-scheme": "dark",
	"sky-base": "#120C2E",
	sky: "linear-gradient(180deg, #0A0720 0%, #171140 14%, #261C5C 26%, #3E2C78 38%, #563A88 50%, #6E4089 62%, #7A4389 72%, #8F468A 82%, #9E4B8C 88%, #C35A91 92%, #DF6E94 95%, #F5BAAF 98%, #F7A076 100%)",
	fg: "#FFF4EE",
	"fg-dim": "rgba(255, 244, 238, 0.76)",
	glass: "rgba(22, 15, 52, 0.62)",
	"glass-hi": "rgba(255, 236, 230, 0.14)",
	"glass-solid": "#1E1645",
	stroke: "rgba(255, 236, 230, 0.18)",
	"stroke-strong": "rgba(255, 236, 230, 0.40)",
	node: "rgba(22, 15, 52, 0.70)",
	accent: "#FFB98F",
	"accent-glow": "rgba(255, 185, 143, 0.42)",
	ok: "#7EEBC6",
	tight: "#FFD66E",
	fail: "#FF6B4A",
	// Surface tints for squares and fills (not text): brighter than the text colours so a fitting
	// setting reads as lit, not grey, on the indigo glass (Chiya, Nov 1: "a bit brighter").
	"ok-tint": "#5FE8B8",
	"tight-tint": "#FFC64F",
	"fail-tint": "#FF6B4A",
	"fail-soft": "rgba(255, 107, 74, 0.22)",
	track: "rgba(255, 244, 238, 0.16)",
	shadow: "0 18px 44px rgba(6, 3, 20, 0.42)",
	// Clouds: a lit edge and a shaded body, as in the reference's sunset clouds.
	"cloud-a": "rgba(239, 128, 160, 0.55)",
	"cloud-b": "rgba(86, 58, 136, 0.62)",
	"pill-bg": "#FFB98F",
	"pill-fg": "#1A1033",
	"term-1": "#A9B8FF",
	"term-2": "#F5BAAF",
	"term-3": "#7EEBC6",
	"term-4": "#E7D3F5",
	"term-5": "#FFC46B",
	"term-6": "#E57AA3",
	focus: "#FFB98F",
} as const;

/**
 * Dawn (the light sky), from Chiya's reference (Nov 1): a painted sunrise, lupine blue overhead
 * through lavender and apple-blossom to sienna and June-day yellow (SW 6810 Lupine, 6838 Vigorous
 * Violet, 76 Appleblossom, 9009 Subdued Sienna, 6682 June Day). Deep violet-navy ink, deep maroon for
 * alerts. Contrast on the sky's top stop: fg 8.5:1, fg-dim 5.5:1, fail 6.3:1; on the glass over it:
 * fg 11.8:1, fg-dim 7.6:1, ok 5.9:1, tight 4.7:1, accent 4.7:1.
 */
export const dayTokens = {
	...shared,
	"color-scheme": "light",
	"sky-base": "#C3CCEA",
	sky: "linear-gradient(180deg, #A8BDE6 0%, #B9C3EA 18%, #C9BFE3 34%, #D9BFD6 48%, #E9C3C0 62%, #F2C2A4 76%, #F6CF8E 88%, #F8DE86 100%)",
	fg: "#241B3D",
	"fg-dim": "#463A5E",
	glass: "rgba(255, 252, 250, 0.50)",
	"glass-hi": "rgba(255, 255, 255, 0.70)",
	"glass-solid": "#F7F3F6",
	stroke: "rgba(36, 27, 61, 0.14)",
	"stroke-strong": "rgba(36, 27, 61, 0.32)",
	node: "rgba(255, 252, 250, 0.74)",
	accent: "#7B4F7A",
	"accent-glow": "rgba(241, 206, 107, 0.72)",
	ok: "#1F5A3D",
	tight: "#8A5200",
	fail: "#6E1027",
	"ok-tint": "#4FBF8A",
	"tight-tint": "#F0B23A",
	"fail-tint": "#A8304F",
	"fail-soft": "rgba(110, 16, 39, 0.16)",
	track: "rgba(36, 27, 61, 0.12)",
	shadow: "0 18px 44px rgba(70, 50, 110, 0.16)",
	"cloud-a": "rgba(255, 246, 240, 0.86)",
	"cloud-b": "rgba(150, 134, 196, 0.42)",
	"pill-bg": "#F1CE6B",
	"pill-fg": "#241B3D",
	"term-1": "#4A5F9C",
	"term-2": "#B8714F",
	"term-3": "#5E8C74",
	"term-4": "#7B5674",
	"term-5": "#C99A1E",
	"term-6": "#B5577A",
	focus: "#241B3D",
} as const;

export type TokenName = keyof typeof duskTokens;
export type TokenSet = Record<TokenName, string>;

const sets = new Map<string, Partial<TokenSet>>([
	["dusk", duskTokens],
	["day", dayTokens],
]);

export const DEFAULT_SET = "dusk";

/** Add or extend a token set from an add-on. Missing tokens fall back to dusk. */
export function registerTokenSet(name: string, set: Partial<TokenSet>): void {
	sets.set(name, { ...(sets.get(name) ?? {}), ...set });
}

export function tokenSetNames(): string[] {
	return [...sets.keys()];
}

export function tokensToCss(name = DEFAULT_SET): string {
	const set = { ...duskTokens, ...(sets.get(name) ?? {}) };
	return Object.entries(set)
		.map(([k, v]) => (k === "color-scheme" ? `color-scheme: ${v};` : `--pf-${k}: ${v};`))
		.join("\n");
}

/** Apply a set to the page. Every set is emitted once so switching is a class flip. */
export function applyTokenSet(name = DEFAULT_SET, root: HTMLElement = document.documentElement): string {
	const chosen = sets.has(name) ? name : DEFAULT_SET;
	let style = document.getElementById("pf-tokens") as HTMLStyleElement | null;
	if (!style) {
		style = document.createElement("style");
		style.id = "pf-tokens";
		document.head.appendChild(style);
	}
	style.textContent = [...sets.keys()]
		.map((n) => `:root[data-pf-theme="${n}"] {\n${tokensToCss(n)}\n}`)
		.join("\n");
	root.dataset.pfTheme = chosen;
	return chosen;
}

export function currentTokenSet(): string {
	return document.documentElement.dataset.pfTheme ?? DEFAULT_SET;
}

/** Token reference for inline styles: tok("accent") -> "var(--pf-accent)". */
export function tok(name: TokenName): string {
	return `var(--pf-${name})`;
}

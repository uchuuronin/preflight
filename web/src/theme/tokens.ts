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

	blur: "18px",
	"blur-node": "10px",
	ease: "cubic-bezier(.23, 1, .32, 1)",
	"ease-move": "cubic-bezier(.77, 0, .175, 1)",
	"dur-press": "160ms",
	"dur-quick": "250ms",
	"dur-value": "600ms",
	"dur-sheet": "450ms",
	"dur-screen": "700ms",
	"dur-sky": "1400ms",
} as const;

export const duskTokens = {
	...shared,
	"color-scheme": "dark",
	"sky-base": "#0E2A38",
	sky: "linear-gradient(180deg, #0A2230 0%, #123747 22%, #1A475C 44%, #3F6276 66%, #9C7A44 86%, #E7B44C 100%)",
	fg: "#FFF5D1",
	"fg-dim": "rgba(255, 245, 209, 0.74)",
	glass: "rgba(14, 38, 52, 0.46)",
	"glass-hi": "rgba(255, 245, 209, 0.10)",
	"glass-solid": "#123747",
	stroke: "rgba(255, 245, 209, 0.18)",
	"stroke-strong": "rgba(255, 245, 209, 0.38)",
	node: "rgba(14, 38, 52, 0.62)",
	accent: "#FAD563",
	"accent-glow": "rgba(250, 213, 99, 0.42)",
	ok: "#8EBD9D",
	tight: "#F2B45A",
	fail: "#FF6B6B",
	"fail-soft": "rgba(139, 30, 45, 0.45)",
	track: "rgba(255, 245, 209, 0.16)",
	shadow: "0 20px 50px rgba(4, 18, 26, 0.40)",
	"cloud-a": "rgba(158, 186, 224, 0.30)",
	"cloud-b": "rgba(250, 213, 99, 0.22)",
	"pill-bg": "#FAD563",
	"pill-fg": "#173E50",
	"term-1": "#9EBAE0",
	"term-2": "#B4BD62",
	"term-3": "#8EBD9D",
	"term-4": "#E8DCB0",
	"term-5": "#FAD563",
	"term-6": "#C3B1E1",
	focus: "#FAD563",
} as const;

export const dayTokens = {
	...shared,
	"color-scheme": "light",
	"sky-base": "#9EBAE0",
	sky: "linear-gradient(180deg, #5F86BC 0%, #7FA3D2 22%, #9EBAE0 44%, #CFE0F2 70%, #FFF5D1 100%)",
	fg: "#173E50",
	"fg-dim": "#3B5F70",
	glass: "rgba(255, 255, 255, 0.46)",
	"glass-hi": "rgba(255, 255, 255, 0.62)",
	"glass-solid": "#F4F7FB",
	stroke: "rgba(23, 62, 80, 0.15)",
	"stroke-strong": "rgba(23, 62, 80, 0.34)",
	node: "rgba(255, 255, 255, 0.72)",
	accent: "#A86F00",
	"accent-glow": "rgba(250, 213, 99, 0.75)",
	ok: "#2E7652",
	tight: "#9A5A00",
	fail: "#8B1E2D",
	"fail-soft": "rgba(139, 30, 45, 0.16)",
	track: "rgba(23, 62, 80, 0.13)",
	shadow: "0 20px 50px rgba(40, 70, 100, 0.18)",
	"cloud-a": "rgba(255, 255, 255, 0.85)",
	"cloud-b": "rgba(255, 224, 196, 0.75)",
	"pill-bg": "#FAD563",
	"pill-fg": "#173E50",
	"term-1": "#70772E",
	"term-2": "#B4BD62",
	"term-3": "#8EBD9D",
	"term-4": "#4E6A7D",
	"term-5": "#D49810",
	"term-6": "#6C5A9C",
	focus: "#173E50",
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

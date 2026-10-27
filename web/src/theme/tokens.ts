/**
 * Theme tokens. Every colour, space, radius and type size in the app comes from here.
 *
 * C owns the light set. D adds other sets (dark) by calling registerTokenSet from an
 * add-on, never by editing this file. Components read tokens as CSS variables
 * (var(--pf-<name>)), so switching sets never re-renders React.
 *
 * Palette: aeronautical sectional chart. Cool chart paper, navy chart ink, and
 * sectional magenta as the single accent, reserved for "what the guide is talking
 * about right now". Status colours are reserved for fits / tight / fails and always
 * ship with an icon and a word.
 */

export type TokenSet = Record<TokenName, string>;

export const lightTokens = {
	// Surfaces
	"surface-page": "#EEF2F1",
	"surface-panel": "#FBFCFC",
	"surface-raised": "#FFFFFF",
	"surface-sunken": "#E3E9E8",
	"surface-map": "#F4F7F6",

	// Ink
	"ink-primary": "#16222E",
	"ink-secondary": "#45535F",
	"ink-muted": "#6E7B86",
	"ink-inverse": "#FFFFFF",

	// Lines
	"line-hairline": "#D3DBDD",
	"line-strong": "#9AA7AE",
	"line-focus": "#1F5A8C",

	// Chart ink: structure that was measured or predicted
	"chart-ink": "#1F5A8C",
	"chart-ink-soft": "#C9D9E7",
	"chart-band": "rgba(31, 90, 140, 0.16)",
	"chart-hatch": "#9AA7AE",

	// The one accent
	"accent": "#A3216F",
	"accent-soft": "#F3DDEA",
	"accent-ink": "#7C1553",

	// Status (reserved; never a series colour)
	// Validated on #FBFCFC: CVD adjacent dE 9.6, normal-vision 18.3, all >= 3:1.
	"status-fits": "#2B7A4B",
	"status-fits-soft": "#DCEFE3",
	"status-tight": "#B98200",
	"status-tight-soft": "#F6EBCC",
	"status-fails": "#C62D2D",
	"status-fails-soft": "#F8DCDC",

	// Categorical series in fixed order (memory-stack terms, model sweep). Validated on
	// #FBFCFC: CVD adjacent dE 9.1, normal-vision 22.9. Slots 3, 4, 6 sit under 3:1, so
	// every use ships direct labels or a table view.
	"series-1": "#2A78D6",
	"series-2": "#EB6834",
	"series-3": "#1BAF7A",
	"series-4": "#EDA100",
	"series-5": "#6250D6",
	"series-6": "#E87BA4",

	// Space (4 px base)
	"space-1": "4px",
	"space-2": "8px",
	"space-3": "12px",
	"space-4": "16px",
	"space-5": "24px",
	"space-6": "32px",
	"space-7": "48px",

	// Radius: small for controls, larger for panels, full for pills
	"radius-control": "6px",
	"radius-panel": "10px",
	"radius-pill": "999px",

	// Type
	"font-ui": "'Instrument Sans', system-ui, sans-serif",
	"font-instrument": "'IBM Plex Mono', ui-monospace, monospace",
	"text-xs": "11px",
	"text-sm": "12.5px",
	"text-md": "14px",
	"text-lg": "17px",
	"text-xl": "22px",
	"text-xxl": "28px",

	// Motion
	"motion-map": "300ms",
	"motion-quick": "120ms",
	"ease-out": "cubic-bezier(0.2, 0.8, 0.2, 1)",

	// Elevation
	"shadow-panel": "0 1px 0 rgba(22, 34, 46, 0.04), 0 1px 3px rgba(22, 34, 46, 0.06)",
	"shadow-pop": "0 6px 24px rgba(22, 34, 46, 0.14)",
} as const;

export type TokenName = keyof typeof lightTokens;

const sets = new Map<string, Partial<TokenSet>>([["light", lightTokens]]);

/** Add a token set (D's dark set, a high-contrast set). Missing tokens fall back to light. */
export function registerTokenSet(name: string, set: Partial<TokenSet>): void {
	sets.set(name, set);
}

export function tokenSetNames(): string[] {
	return [...sets.keys()];
}

/** CSS custom-property text for one set, for a :root rule or a scoped element. */
export function tokensToCss(name = "light"): string {
	const set = { ...lightTokens, ...(sets.get(name) ?? {}) };
	return Object.entries(set)
		.map(([k, v]) => `--pf-${k}: ${v};`)
		.join("\n");
}

/** Apply a set to the document. Returns the applied set name. */
export function applyTokenSet(name = "light", root: HTMLElement = document.documentElement): string {
	const chosen = sets.has(name) ? name : "light";
	let style = document.getElementById("pf-tokens") as HTMLStyleElement | null;
	if (!style) {
		style = document.createElement("style");
		style.id = "pf-tokens";
		document.head.appendChild(style);
	}
	style.textContent = `:root[data-pf-theme="${chosen}"] {\n${tokensToCss(chosen)}\n}`;
	root.dataset.pfTheme = chosen;
	return chosen;
}

/** Token reference for inline styles: tok("accent") -> "var(--pf-accent)". */
export function tok(name: TokenName): string {
	return `var(--pf-${name})`;
}

/** Raw value, for the few places that need a literal (canvas, SVG pattern ids in tests). */
export function tokenValue(name: TokenName, set = "light"): string {
	return (sets.get(set) as Partial<TokenSet> | undefined)?.[name] ?? lightTokens[name];
}

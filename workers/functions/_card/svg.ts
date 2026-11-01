/**
 * The share card (1200 x 630, the size link previews use) and the README badge, as SVG.
 * The card is drawn in the app's dusk colours with its own fonts (Instrument Serif, Instrument
 * Sans, JetBrains Mono), which render.ts hands to the rasteriser; the badge uses the system font
 * stack, as shields-style badges do, because GitHub shows it as SVG.
 */
import type { Headline, StatusWord } from "./headline";

export const CARD_W = 1200;
export const CARD_H = 630;

const FG = "#FFF5D1";
const DIM = "#B9C6C4";
const BG_TOP = "#0E2634";
const BG_BOTTOM = "#1F4A5E";

const CHIP: Record<StatusWord, string> = {
	Measured: "#8FD3A5",
	Predicted: "#FFF5D1",
	Extrapolated: "#F2C66D",
	"Not trusted": "#B9C6C4",
	Illustrative: "#C9B6F2",
};

export function esc(s: string): string {
	return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

/** Greedy word wrap by an average character width: good enough to keep a title inside the card. */
export function wrap(text: string, maxChars: number, maxLines: number): string[] {
	const words = text.split(/\s+/).filter(Boolean);
	const lines: string[] = [];
	let cur = "";
	for (const w of words) {
		const next = cur ? `${cur} ${w}` : w;
		if (next.length <= maxChars) cur = next;
		else {
			if (cur) lines.push(cur);
			cur = w.length > maxChars ? `${w.slice(0, maxChars - 3)}...` : w;
		}
	}
	if (cur) lines.push(cur);
	if (lines.length > maxLines) {
		const kept = lines.slice(0, maxLines);
		const last = kept[maxLines - 1] ?? "";
		kept[maxLines - 1] = last.length > maxChars - 3 ? `${last.slice(0, maxChars - 3)}...` : `${last}...`;
		return kept;
	}
	return lines;
}

export function cardSvg(h: Headline, host: string): string {
	// Vertical rhythm sized so a two-line title and four rows still clear the footer (y 596).
	const title = wrap(h.title, 36, 2);
	const titleY = 142;
	const titleLead = 58;
	const titleSize = 54;
	const subY = titleY + (title.length - 1) * titleLead + 44;
	const rowsTop = subY + 58;
	const rowH = 64;
	const rows = h.lines
		.map((l, i) => {
			const y = rowsTop + i * rowH;
			const chipW = 24 + l.status.length * 11.5;
			const colour = CHIP[l.status];
			return [
				`<text x="80" y="${y}" font-family="Instrument Sans" font-weight="400" font-size="22" fill="${DIM}">${esc(l.label)}</text>`,
				`<text x="80" y="${y + 33}" font-family="JetBrains Mono" font-weight="500" font-size="30" fill="${FG}">${esc(l.value)}</text>`,
				`<rect x="${CARD_W - 80 - chipW}" y="${y + 2}" width="${chipW}" height="36" rx="18" fill="none" stroke="${colour}" stroke-width="2"/>`,
				`<text x="${CARD_W - 80 - chipW / 2}" y="${y + 27}" text-anchor="middle" font-family="Instrument Sans" font-weight="600" font-size="20" fill="${colour}">${esc(l.status)}</text>`,
			].join("");
		})
		.join("");
	const empty = h.lines.length
		? ""
		: `<text x="80" y="${rowsTop + 20}" font-family="Instrument Sans" font-size="28" fill="${DIM}">No numbers yet: open the flight to see where it stands.</text>`;
	const foot = h.illustrative
		? "Illustrative numbers from a fixture, not measured."
		: "Every number says how it was made: measured, predicted or not trusted.";
	return [
		`<svg xmlns="http://www.w3.org/2000/svg" width="${CARD_W}" height="${CARD_H}" viewBox="0 0 ${CARD_W} ${CARD_H}">`,
		`<defs><linearGradient id="sky" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="${BG_TOP}"/><stop offset="1" stop-color="${BG_BOTTOM}"/></linearGradient></defs>`,
		`<rect width="${CARD_W}" height="${CARD_H}" fill="url(#sky)"/>`,
		`<text x="80" y="78" font-family="JetBrains Mono" font-weight="500" font-size="20" letter-spacing="4" fill="${DIM}">PREFLIGHT · FLIGHT RECORD</text>`,
		...title.map((t, i) => `<text x="80" y="${titleY + i * titleLead}" font-family="Instrument Serif" font-size="${titleSize}" fill="${FG}">${esc(t)}</text>`),
		h.subtitle ? `<text x="80" y="${subY}" font-family="Instrument Sans" font-size="26" fill="${DIM}">${esc(h.subtitle)}</text>` : "",
		rows,
		empty,
		`<text x="80" y="${CARD_H - 34}" font-family="Instrument Sans" font-size="22" fill="${DIM}">${esc(foot)}</text>`,
		`<text x="${CARD_W - 80}" y="${CARD_H - 34}" text-anchor="end" font-family="JetBrains Mono" font-weight="500" font-size="20" fill="${DIM}">${esc(host)}</text>`,
		"</svg>",
	].join("");
}

/** A shields-style badge: "preflight" on the left, the first headline number and its status on the right. */
export function badgeSvg(h: Headline): string {
	const first = h.lines[0];
	const right = first ? `${first.label.toLowerCase()} ${first.value} · ${first.status.toLowerCase()}` : "flight record";
	const left = "preflight";
	const w = (s: string) => Math.round(s.length * 6.6 + 12);
	const lw = w(left);
	const rw = w(right);
	const colour = first ? (first.status === "Measured" ? "#2E7D4F" : first.status === "Not trusted" ? "#6B7B7A" : "#1F4A5E") : "#1F4A5E";
	const label = `${left}: ${right}`;
	return [
		`<svg xmlns="http://www.w3.org/2000/svg" width="${lw + rw}" height="20" role="img" aria-label="${esc(label)}">`,
		`<title>${esc(label)}</title>`,
		`<rect width="${lw}" height="20" fill="#0E2634"/>`,
		`<rect x="${lw}" width="${rw}" height="20" fill="${colour}"/>`,
		`<g fill="#fff" font-family="Verdana,Geneva,DejaVu Sans,sans-serif" font-size="11">`,
		`<text x="${lw / 2}" y="14" text-anchor="middle">${esc(left)}</text>`,
		`<text x="${lw + rw / 2}" y="14" text-anchor="middle">${esc(right)}</text>`,
		"</g></svg>",
	].join("");
}

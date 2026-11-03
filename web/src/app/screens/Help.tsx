/**
 * Help: docs/guide/knobs.md (B and D), as a plain technical document in the site's colours and fonts,
 * with a table of contents on the left (a menu at the top on a phone). Reached from one place only, a
 * text link on Home (Chiya, Nov 3); behind `c.help`. Static content: it changes nothing.
 */
import { useMemo } from "react";
import { t } from "../../lib/i18n";
import { renderMarkdown } from "../../lib/markdown";

// Read at build time; an empty object until the document exists, so the build never breaks on it.
const DOCS = import.meta.glob("../../../../docs/guide/knobs.md", { query: "?raw", import: "default", eager: true }) as Record<string, string>;
export const HELP_SOURCE = Object.values(DOCS)[0] ?? "";

export function Help() {
	const doc = useMemo(() => renderMarkdown(HELP_SOURCE), []);
	const toc = (
		<ol className="pf-doc__toc-list">
			{doc.headings.map((h) => (
				<li key={h.id} className={h.level === 3 ? "is-sub" : undefined}>
					<a href={`#${h.id}`} onClick={(e) => jump(e, h.id)}>
						{h.text}
					</a>
				</li>
			))}
		</ol>
	);
	return (
		<div className="pf-scroll">
			<div className="pf-inner pf-doc">
				{doc.headings.length ? (
					<>
						<nav className="pf-doc__toc" aria-label={t("help.contents")}>
							<b>{t("help.contents")}</b>
							{toc}
						</nav>
						<details className="pf-doc__menu">
							<summary>{t("help.contents")}</summary>
							<nav aria-label={t("help.contents")}>{toc}</nav>
						</details>
					</>
				) : null}
				<article className="pf-doc__body">{HELP_SOURCE ? doc.body : <p>{t("help.missing")}</p>}</article>
			</div>
		</div>
	);
}

/** In-page jumps without touching the hash route (#/help stays the page's address). */
function jump(e: React.MouseEvent, id: string): void {
	e.preventDefault();
	const el = document.getElementById(id);
	if (!el) return;
	// Close the phone's menu first: closing it mid-scroll moved the page under the jump (a11y pass, Nov 3).
	(e.currentTarget.closest("details") as HTMLDetailsElement | null)?.removeAttribute("open");
	const still = window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;
	el.scrollIntoView({ behavior: still ? "auto" : "smooth", block: "start" });
}

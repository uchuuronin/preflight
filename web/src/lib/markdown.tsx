/**
 * A small Markdown reader for the Help page (docs/guide/knobs.md): headings, paragraphs, lists, fenced
 * code, block quotes, pipe tables, and inline code, bold, italics and links. It builds React elements,
 * never HTML strings, so nothing in the document can run or inject markup; a link opens only when it is
 * http(s), an in-page #anchor, or a site hash route (#/...).
 */
import type { ReactNode } from "react";

export interface Heading {
	level: number;
	text: string;
	id: string;
}

/** A heading's anchor: lower case, words joined by dashes, unique within the document. */
export function slug(text: string, seen: Map<string, number>): string {
	const base =
		text
			.toLowerCase()
			.replace(/`/g, "")
			.replace(/[^a-z0-9]+/g, "-")
			.replace(/^-|-$/g, "") || "section";
	const n = seen.get(base) ?? 0;
	seen.set(base, n + 1);
	return n ? `${base}-${n + 1}` : base;
}

function safeHref(href: string): string | null {
	if (/^https?:\/\//i.test(href)) return href;
	if (/^#[\w/-]*$/.test(href)) return href;
	return null;
}

/** Inline: `code`, **bold**, *italic* or _italic_, [text](href). */
export function inline(text: string, key = "i"): ReactNode[] {
	const out: ReactNode[] = [];
	const re = /`([^`]+)`|\*\*([^*]+)\*\*|\*([^*\s][^*]*)\*|(?<![\w])_([^_\s][^_]*)_(?![\w])|\[([^\]]+)\]\(([^)\s]+)\)/g;
	let last = 0;
	let m: RegExpExecArray | null;
	let n = 0;
	while ((m = re.exec(text))) {
		if (m.index > last) out.push(text.slice(last, m.index));
		const k = `${key}.${n++}`;
		if (m[1] !== undefined) out.push(<code key={k}>{m[1]}</code>);
		else if (m[2] !== undefined) out.push(<strong key={k}>{inline(m[2], k)}</strong>);
		else if (m[3] !== undefined) out.push(<em key={k}>{inline(m[3], k)}</em>);
		else if (m[4] !== undefined) out.push(<em key={k}>{inline(m[4], k)}</em>);
		else {
			const href = safeHref(m[6]!);
			out.push(
				href ? (
					<a key={k} href={href} {...(href.startsWith("http") ? { target: "_blank", rel: "noopener noreferrer" } : {})}>
						{inline(m[5]!, k)}
					</a>
				) : (
					<span key={k}>{inline(m[5]!, k)}</span>
				),
			);
		}
		last = m.index + m[0].length;
	}
	if (last < text.length) out.push(text.slice(last));
	return out;
}

const cells = (row: string): string[] =>
	row
		.trim()
		.replace(/^\||\|$/g, "")
		.split(/(?<!\\)\|/)
		.map((c) => c.trim().replace(/\\\|/g, "|"));

/** The document as elements, and its headings (level 2 and 3) for a table of contents. */
export function renderMarkdown(src: string): { body: ReactNode[]; headings: Heading[] } {
	const lines = src.replace(/\r\n?/g, "\n").split("\n");
	const body: ReactNode[] = [];
	const headings: Heading[] = [];
	const seen = new Map<string, number>();
	let i = 0;
	let k = 0;
	const isBlockStart = (l: string) => /^(#{1,6}\s|```|>\s?|\s*([-*+]|\d+[.)])\s|\|)/.test(l);
	while (i < lines.length) {
		const line = lines[i]!;
		if (!line.trim()) {
			i++;
			continue;
		}
		const h = /^(#{1,6})\s+(.*?)\s*#*\s*$/.exec(line);
		if (h) {
			const level = h[1]!.length;
			const text = h[2]!;
			const id = slug(text, seen);
			const plain = text.replace(/`/g, "");
			if (level === 2 || level === 3) headings.push({ level, text: plain, id });
			const Tag = `h${Math.min(6, level + 0)}` as "h1";
			body.push(
				<Tag key={k++} id={id}>
					{inline(text, `h${k}`)}
				</Tag>,
			);
			i++;
			continue;
		}
		if (line.startsWith("```")) {
			const code: string[] = [];
			i++;
			while (i < lines.length && !lines[i]!.startsWith("```")) code.push(lines[i++]!);
			i++;
			body.push(
				<pre key={k++}>
					<code>{code.join("\n")}</code>
				</pre>,
			);
			continue;
		}
		if (/^>\s?/.test(line)) {
			const quote: string[] = [];
			while (i < lines.length && /^>\s?/.test(lines[i]!)) quote.push(lines[i++]!.replace(/^>\s?/, ""));
			body.push(<blockquote key={k++}>{renderMarkdown(quote.join("\n")).body}</blockquote>);
			continue;
		}
		if (line.trim().startsWith("|") && i + 1 < lines.length && /^\s*\|?\s*:?-{3,}/.test(lines[i + 1]!)) {
			const head = cells(line);
			i += 2;
			const rows: string[][] = [];
			while (i < lines.length && lines[i]!.trim().startsWith("|")) rows.push(cells(lines[i++]!));
			const tk = k++;
			body.push(
				<div key={tk} className="pf-doc__table">
					<table>
						<thead>
							<tr>
								{head.map((c, j) => (
									<th key={j}>{inline(c, `t${tk}.h${j}`)}</th>
								))}
							</tr>
						</thead>
						<tbody>
							{rows.map((r, ri) => (
								<tr key={ri}>
									{head.map((_, j) => (
										<td key={j}>{inline(r[j] ?? "", `t${tk}.${ri}.${j}`)}</td>
									))}
								</tr>
							))}
						</tbody>
					</table>
				</div>,
			);
			continue;
		}
		const li = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(line);
		if (li) {
			const ordered = /\d/.test(li[1]!);
			const items: string[] = [];
			while (i < lines.length) {
				const m = /^\s*([-*+]|\d+[.)])\s+(.*)$/.exec(lines[i]!);
				if (m) {
					items.push(m[2]!);
					i++;
				} else if (lines[i]!.trim() && /^\s{2,}/.test(lines[i]!) && items.length) {
					items[items.length - 1] += ` ${lines[i++]!.trim()}`;
				} else break;
			}
			const lk = k++;
			const List = ordered ? "ol" : "ul";
			body.push(
				<List key={lk}>
					{items.map((t, j) => (
						<li key={j}>{inline(t, `l${lk}.${j}`)}</li>
					))}
				</List>,
			);
			continue;
		}
		const para: string[] = [line.trim()];
		i++;
		while (i < lines.length && lines[i]!.trim() && !isBlockStart(lines[i]!)) para.push(lines[i++]!.trim());
		const pk = k++;
		body.push(<p key={pk}>{inline(para.join(" "), `p${pk}`)}</p>);
	}
	return { body, headings };
}

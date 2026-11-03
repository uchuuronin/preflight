// @vitest-environment jsdom
import { render } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { renderMarkdown } from "./markdown";

const DOC = `# Knobs

Intro with \`batch_size\` and **bold** and a [link](https://example.com) and [bad](javascript:alert(1)).

## Batch size

- one
- two \`lr\`

### Under it

| Knob | Meaning |
| --- | --- |
| \`lr\` | the learning rate |

\`\`\`
preflight run train.py
\`\`\`

## Batch size
<script>alert(1)</script>
`;

describe("the Help page's Markdown", () => {
	it("reads headings into a contents list with unique anchors", () => {
		const { headings } = renderMarkdown(DOC);
		expect(headings.map((h) => [h.level, h.id])).toEqual([
			[2, "batch-size"],
			[3, "under-it"],
			[2, "batch-size-2"],
		]);
	});
	it("renders code, tables and lists, and no markup or unsafe link from the text", () => {
		const { container } = render(<div>{renderMarkdown(DOC).body}</div>);
		expect(container.querySelectorAll("code").length).toBeGreaterThanOrEqual(4);
		expect(container.querySelector("table td code")?.textContent).toBe("lr");
		expect(container.querySelectorAll("li")).toHaveLength(2);
		expect(container.querySelector("script")).toBeNull();
		expect(container.textContent).toContain("<script>alert(1)</script>");
		const hrefs = [...container.querySelectorAll("a")].map((a) => a.getAttribute("href"));
		expect(hrefs).toEqual(["https://example.com"]);
	});
});

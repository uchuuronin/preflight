import { act, cleanup, fireEvent, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const on = new Set<string>();
vi.mock("../lib/flags", () => ({ flag: (n: string) => on.has(n) }));
vi.mock("./route", () => ({ go: vi.fn() }));

import { CheckinHelp } from "./CheckinHelp";
import { rememberRequest } from "./checkinTrouble";

const install = { step: "install" as const, missing: [{ module: "torchvision", pkg: "torchvision" }] };

describe("check-in help on Boarding", () => {
	afterEach(() => {
		cleanup();
		on.clear();
		vi.unstubAllGlobals();
	});
	it("says the step and what's missing, and how to fix it by hand while the runner can't take extra packages", () => {
		const { container } = render(<CheckinHelp trouble={install} jobId="j1" />);
		expect(container.textContent).toContain("Stopped at an import");
		expect(container.textContent).toContain("Your script imports torchvision, which isn't installed here.");
		expect(container.textContent).toContain("Add torchvision to a requirements.txt next to your script");
		expect(container.querySelector("input")).toBeNull();
	});
	it("doesn't send a student to PyPI for what may be a file of their own", () => {
		const { container } = render(<CheckinHelp trouble={{ step: "install", missing: [{ module: "utils", pkg: null }] }} jobId="j0" />);
		expect(container.textContent).toContain("If utils is a file of yours, upload it with your script.");
		expect(container.textContent).not.toContain("Add utils to a requirements.txt");
	});
	it("offers the extra-packages box, filled in, with the flag on and the check-in remembered", () => {
		on.add("c.checkin_help");
		rememberRequest("j2", { source: { kind: "repo", uri: "https://github.com/a/b" } } as never);
		const { container, getByRole } = render(<CheckinHelp trouble={install} jobId="j2" />);
		expect((container.querySelector("input") as HTMLInputElement).value).toBe("torchvision");
		expect(getByRole("button", { name: "Check in again with these packages" })).toBeTruthy();
	});
	it("offers starting from one file for a repo too big to read", () => {
		on.add("c.checkin_help");
		rememberRequest("j3", { source: { kind: "repo", uri: "https://github.com/a/b", entrypoint: "lm_eval/__main__.py" } } as never);
		const { container, getByRole } = render(<CheckinHelp trouble={{ step: "read", limit: "repo" }} jobId="j3" />);
		expect(container.textContent).toContain("more than Preflight reads whole: 2,000 files and 25 MB of text");
		expect((container.querySelector("input") as HTMLInputElement).value).toBe("lm_eval/__main__.py");
		expect((getByRole("button", { name: "Check in from this file" }) as HTMLButtonElement).disabled).toBe(false);
	});

	const resend = async (flags: string[]) => {
		for (const f of flags) on.add(f);
		rememberRequest("j4", { source: { kind: "upload", files: { "train.py": "import einops" }, entrypoint: "train.py" } } as never);
		const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "j5" }), { status: 202 }));
		vi.stubGlobal("fetch", fetch);
		const { getByRole } = render(<CheckinHelp trouble={{ step: "install", missing: [{ module: "einops", pkg: "einops" }] }} jobId="j4" />);
		await act(async () => {
			fireEvent.click(getByRole("button", { name: "Check in again with these packages" }));
		});
		return JSON.parse(fetch.mock.calls[0][1].body as string);
	};
	it("names the job it follows with A's free resend on, so an import stop's resend can be free", async () => {
		const body = await resend(["c.checkin_help", "runner.free_resend"]);
		expect(body.resend_of).toBe("j4");
		expect(body.extra_packages).toEqual(["einops"]);
	});
	it("sends no resend_of with the free resend off: the API's schema would refuse the field", async () => {
		const body = await resend(["c.checkin_help"]);
		expect(body).not.toHaveProperty("resend_of");
		expect(body.extra_packages).toEqual(["einops"]);
	});
});

describe("a check-in that measured nothing (PM, Nov 3)", () => {
	afterEach(() => {
		cleanup();
		on.clear();
		vi.unstubAllGlobals();
	});
	it("says so plainly, with why and a way to try again that resends the same check-in", async () => {
		const body = { source: { kind: "repo", uri: "https://github.com/karpathy/nanoGPT" } } as never;
		rememberRequest("j9", body);
		on.add("runner.free_resend");
		on.add("c.checkin_help");
		const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "j10" }), { status: 202 }));
		vi.stubGlobal("fetch", fetch);
		const { container, getByRole } = render(<CheckinHelp trouble={{ step: "measure", code: "gpu.unavailable" }} jobId="j9" />);
		expect(container.textContent).toContain("Nothing was measured");
		expect(container.textContent).toContain("No GPU was free in time");
		await act(async () => {
			fireEvent.click(getByRole("button", { name: "Try again" }));
		});
		expect(JSON.parse(fetch.mock.calls[0][1].body as string)).toEqual({ source: { kind: "repo", uri: "https://github.com/karpathy/nanoGPT" }, resend_of: "j9" });
	});
	it("names the repo's own module, not requirements.txt, and sends no identical resend that would stop the same way", () => {
		rememberRequest("jx", { source: { kind: "repo", uri: "https://github.com/karpathy/nanoGPT" } } as never);
		on.add("c.checkin_help");
		const { container, queryByRole } = render(<CheckinHelp trouble={{ step: "measure", code: "missing_module.local" }} jobId="jx" />);
		expect(queryByRole("button", { name: "Try again" })).toBeNull();
		expect(container.textContent).toContain("whole repository");
		expect(container.textContent).toContain("Check in again from New flight.");
	});
});

import { cleanup, render } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

const on = new Set<string>();
vi.mock("../lib/flags", () => ({ flag: (n: string) => on.has(n) }));

import { CheckinHelp } from "./CheckinHelp";
import { rememberRequest } from "./checkinHelp";

const install = { step: "install" as const, missing: [{ module: "torchvision", pkg: "torchvision" }] };

describe("check-in help on Boarding", () => {
	afterEach(() => {
		cleanup();
		on.clear();
	});
	it("says the step and what's missing, and how to fix it by hand while the runner can't take extra packages", () => {
		const { container } = render(<CheckinHelp trouble={install} jobId="j1" />);
		expect(container.textContent).toContain("Stopped while installing");
		expect(container.textContent).toContain("Your script imports torchvision, which isn't installed here.");
		expect(container.textContent).toContain("Add torchvision to a requirements.txt next to your script");
		expect(container.querySelector("input")).toBeNull();
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
		expect(container.textContent).toContain("bigger than Preflight reads: at most 2,000 files and 25 MB of text");
		expect((container.querySelector("input") as HTMLInputElement).value).toBe("lm_eval/__main__.py");
		expect((getByRole("button", { name: "Check in from this file" }) as HTMLButtonElement).disabled).toBe(false);
	});
});

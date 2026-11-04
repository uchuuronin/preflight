import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { Home } from "./Home";

afterEach(cleanup);

describe("Home", () => {
	it("says it was built during the hackathon and links the repo (PM, Nov 3: the rules)", () => {
		render(<Home />);
		expect(screen.getByText(/Built during the Meta Global AI Developer Hackathon/)).toBeTruthy();
		expect(screen.getByRole("link", { name: "GitHub" }).getAttribute("href")).toBe("https://github.com/uchuuronin/preflight");
	});
	it("gives each how-it-works card an h2 under the page's h1", () => {
		render(<Home />);
		expect(screen.getAllByRole("heading", { level: 2 }).length).toBeGreaterThanOrEqual(4);
		expect(screen.queryAllByRole("heading", { level: 3 })).toHaveLength(0);
	});
});

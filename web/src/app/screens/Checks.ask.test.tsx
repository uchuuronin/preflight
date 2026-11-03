// @vitest-environment jsdom
import { cleanup, render, screen, within } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import ask from "../../coverage/ask_model.json";
import { Checks } from "./Checks";

afterEach(cleanup);

describe("the checks page's same-code comparison (B, Nov 2)", () => {
	it("shows each row's counts for Preflight and Muse Spark alone, as B scored them", () => {
		render(<Checks />);
		const table = screen.getByRole("table", { name: /Same code, asked to Muse Spark without probes/ });
		const rows = within(table).getAllByRole("row").slice(1);
		expect(rows.length).toBe(ask.rows.length);
		expect(within(rows[0]!).getByText("Peak memory within 10%")).toBeTruthy();
		expect(rows[0]!.textContent).toContain("2 of 4");
		expect(rows[0]!.textContent).toContain("0 of 4");
	});
});

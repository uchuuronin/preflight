// @vitest-environment jsdom
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { afterEach, describe, expect, it, vi } from "vitest";

const flagsOn = vi.hoisted(() => new Set<string>());
vi.mock("../../lib/flags", () => ({ flag: (name: string) => name === "runner.queue" || flagsOn.has(name) }));

import { type IntakePanelProps, registerAppPanel, unregister } from "../../registry";
import en from "../../i18n/en.json";
import { DATA_USE_OK_ID } from "../DataUse";
import { NewFlight, repoUri } from "./NewFlight";

const SOURCE = { kind: "notebook", files: { "train.ipynb": '{"cells":[]}' } };

function Panel({ setSource }: IntakePanelProps) {
	return (
		<button type="button" onClick={() => setSource?.(SOURCE)}>
			hand over
		</button>
	);
}

afterEach(() => {
	cleanup();
	flagsOn.clear();
	unregister("notebook.intake");
	vi.unstubAllGlobals();
	try {
		localStorage.clear();
	} catch {
		// no storage in this environment
	}
});

describe("the check-in desk and intake panels", () => {
	it("checks in the source an intake panel hands over, with the goal and GPU", async () => {
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: Panel });
		const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "01JOB" }), { status: 202 }));
		vi.stubGlobal("fetch", fetch);
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		fireEvent.click(screen.getByRole("checkbox"));
		const checkIn = screen.getByRole("button", { name: "Check in" }) as HTMLButtonElement;
		expect(checkIn.disabled).toBe(true); // nothing handed over yet
		fireEvent.click(screen.getByRole("button", { name: "hand over" }));
		expect(checkIn.disabled).toBe(false);
		await act(async () => {
			fireEvent.click(checkIn);
		});
		const body = JSON.parse(fetch.mock.calls[0][1].body as string);
		expect(body.source).toEqual(SOURCE);
		expect(body.objective.kind).toBe("min_cost");
		expect(body.gpu).toBe("L4"); // "L4 or A10" by default: the runner gives way to an A10 when no L4 is free
		expect(body).not.toHaveProperty("extra_packages");
	});

	it("starts on an L4 (or an A10), marked default, so a visitor who doesn't know isn't sent to an A100", () => {
		render(<NewFlight onSample={() => {}} />);
		const l4 = screen.getByRole("button", { name: new RegExp(`^${en["newflight.gpu.default"]}`) });
		expect(l4.textContent).toBe(`${en["newflight.gpu.default"]} ${en["newflight.default"]}`);
		expect(l4.getAttribute("aria-pressed")).toBe("true");
		expect(screen.getByRole("button", { name: "A100-SXM-80GB" }).getAttribute("aria-pressed")).toBe("false");
		// The starting picks say they're defaults, not choices already made; nothing else is marked.
		const cheap = screen.getByRole("button", { name: new RegExp(`^${en["goal.min_cost"]}`) });
		expect(cheap.textContent).toBe(`${en["goal.min_cost"]} ${en["newflight.default"]}`);
		expect(cheap.getAttribute("aria-pressed")).toBe("true");
		expect(screen.getByRole("button", { name: en["goal.min_time"] }).textContent).not.toContain(en["newflight.default"]);
		expect(screen.getByRole("button", { name: en["newflight.gpu.unsure"] }).getAttribute("aria-pressed")).toBe("false");
	});

	it("drops a tab's source when the visitor leaves it, since the panel forgets it too", () => {
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: Panel });
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		fireEvent.click(screen.getByRole("checkbox"));
		fireEvent.click(screen.getByRole("button", { name: "hand over" }));
		const checkIn = () => screen.getByRole("button", { name: "Check in" }) as HTMLButtonElement;
		expect(checkIn().disabled).toBe(false);
		fireEvent.click(screen.getAllByRole("tab")[0]);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		expect(checkIn().disabled).toBe(true);
	});

	it("gives a panel the same setSource on every render, so calling it from an effect can't loop", () => {
		const seen = new Set<unknown>();
		function EffectPanel({ setSource }: IntakePanelProps) {
			seen.add(setSource);
			useEffect(() => {
				setSource?.(SOURCE);
			}, [setSource]);
			return <span>effect panel</span>;
		}
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: EffectPanel });
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		expect(screen.getByText("effect panel")).toBeTruthy();
		expect(seen.size).toBe(1);
	});

	it("refuses a handed-over source over the size limit, before anything is sent", () => {
		const big = { kind: "notebook", files: { "train.ipynb": "x".repeat(5_000_001) } };
		function BigPanel({ setSource }: IntakePanelProps) {
			return (
				<button type="button" onClick={() => setSource?.(big)}>
					hand over big
				</button>
			);
		}
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: BigPanel });
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		fireEvent.click(screen.getByRole("checkbox"));
		fireEvent.click(screen.getByRole("button", { name: "hand over big" }));
		expect((screen.getByRole("button", { name: "Check in" }) as HTMLButtonElement).disabled).toBe(true);
		expect(screen.getByRole("alert").textContent).toMatch(/more than 5 MB/);
	});

	it("sends notebooks to that tab instead of the Upload tab", () => {
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: Panel });
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getByRole("tab", { name: "Upload" }));
		const input = document.querySelector("input[type=file]") as HTMLInputElement;
		expect(input.accept).not.toContain(".ipynb");
		expect(input.accept).toContain(".py");
	});

	it("says why Check in is grey when the link isn't a GitHub repo, and marks the field", () => {
		render(<NewFlight onSample={() => {}} />);
		const field = screen.getByLabelText("GitHub repository") as HTMLInputElement;
		expect(screen.queryByText(en["newflight.repo.invalid"])).toBeNull(); // nothing typed: no complaint
		for (const bad of ["not a link", "https://gitlab.com/a/b", "github.com/owner"]) {
			fireEvent.change(field, { target: { value: bad } });
			expect(screen.queryByText(en["newflight.repo.invalid"])).toBeNull(); // still typing
			fireEvent.blur(field);
			const msg = screen.getByText(en["newflight.repo.invalid"]);
			expect(msg.id).not.toBe("");
			expect(field.getAttribute("aria-describedby")?.split(" ")).toContain(msg.id);
			expect(field.getAttribute("aria-invalid")).toBe("true");
			expect(msg.closest("[role=status]")).not.toBeNull(); // spoken when it appears
		}
		fireEvent.change(field, { target: { value: "github.com/karpathy/nanoGPT" } });
		fireEvent.blur(field);
		expect(screen.queryByText(en["newflight.repo.invalid"])).toBeNull();
		expect(field.getAttribute("aria-invalid")).toBeNull();
	});

	it("takes the visitor to the Got it box, then back to Check in once it's ticked", async () => {
		render(<NewFlight onSample={() => {}} />);
		Element.prototype.scrollIntoView = vi.fn();
		fireEvent.change(screen.getByLabelText("GitHub repository"), { target: { value: "github.com/karpathy/nanoGPT" } });
		const box = document.getElementById(DATA_USE_OK_ID) as HTMLInputElement;
		fireEvent.click(screen.getByRole("button", { name: en["datause.needed"] }));
		expect(document.activeElement).toBe(box);
		fireEvent.click(box);
		await act(async () => {
			await new Promise((r) => requestAnimationFrame(() => r(null)));
		});
		expect(document.activeElement).toBe(screen.getByRole("button", { name: "Check in" }));
	});

	it("states the measured check-in time, with no dollar promise, once the notice is ticked", () => {
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getByRole("checkbox"));
		const hint = screen.getByText(en["newflight.cost_hint"]);
		// The production check-ins' range (4 min 59 s and 10 min 21 s), not the first smoke run's time.
		expect(hint.textContent).toMatch(/about 5 to 11 minutes/);
		expect(hint.textContent).not.toMatch(/\$|US\$|£/);
	});
});

describe("check in by command, and extra packages", () => {
	const send = async (setup: () => void) => {
		const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "01JOB" }), { status: 202 }));
		vi.stubGlobal("fetch", fetch);
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getByRole("checkbox", { name: en["datause.accept"] }));
		setup();
		const go = screen.getByRole("button", { name: "Check in" }) as HTMLButtonElement;
		if (go.disabled) return null;
		await act(async () => {
			fireEvent.click(go);
		});
		return JSON.parse(fetch.mock.calls[0][1].body as string);
	};

	it("offers Run a command only with runner.command_checkin on", () => {
		render(<NewFlight onSample={() => {}} />);
		expect(screen.queryByRole("tab", { name: en["newflight.tab.command"] })).toBeNull();
	});

	it("sends the command as its words, quotes kept together", async () => {
		flagsOn.add("runner.command_checkin");
		const body = await send(() => {
			fireEvent.click(screen.getByRole("tab", { name: en["newflight.tab.command"] }));
			fireEvent.change(screen.getByLabelText(en["newflight.command.label"]), {
				target: { value: `lm_eval --model hf --model_args "pretrained=EleutherAI/pythia-160m,dtype=float16" --tasks arc_easy` },
			});
		});
		expect(body.source).toEqual({
			kind: "command",
			args: ["lm_eval", "--model", "hf", "--model_args", "pretrained=EleutherAI/pythia-160m,dtype=float16", "--tasks", "arc_easy"],
		});
	});

	it("won't send a command Preflight doesn't run, and says which it does", async () => {
		flagsOn.add("runner.command_checkin");
		const body = await send(() => {
			fireEvent.click(screen.getByRole("tab", { name: en["newflight.tab.command"] }));
			const field = screen.getByLabelText(en["newflight.command.label"]);
			fireEvent.change(field, { target: { value: "vllm bench throughput" } });
			fireEvent.blur(field);
		});
		expect(body).toBeNull();
		expect(screen.getByText(/doesn't run “vllm” by name. It runs: lm_eval/)).toBeTruthy();
	});

	it("sends extra packages with any check-in, and holds Check in on a bad name", async () => {
		const body = await send(() => {
			fireEvent.change(screen.getByLabelText("GitHub repository"), { target: { value: "github.com/karpathy/nanoGPT" } });
			fireEvent.change(screen.getByLabelText(/Extra packages/), { target: { value: "einops, timm" } });
		});
		expect(body.extra_packages).toEqual(["einops", "timm"]);
		cleanup();
		localStorage.clear();
		const none = await send(() => {
			fireEvent.change(screen.getByLabelText("GitHub repository"), { target: { value: "github.com/karpathy/nanoGPT" } });
			fireEvent.change(screen.getByLabelText(/Extra packages/), { target: { value: "einops; rm -rf" } });
		});
		expect(none).toBeNull();
	});
});

describe("private triage on the check-in desk", () => {
	const checkIn = async (tick: boolean) => {
		registerAppPanel({ id: "notebook.intake", slot: "intake", titleKey: "newflight.desk", component: Panel });
		const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "01JOB" }), { status: 202 }));
		vi.stubGlobal("fetch", fetch);
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getAllByRole("tab")[2]);
		fireEvent.click(screen.getByRole("checkbox", { name: en["datause.accept"] }));
		if (tick) fireEvent.click(screen.getByRole("checkbox", { name: en["newflight.private.label"] }));
		fireEvent.click(screen.getByRole("button", { name: "hand over" }));
		await act(async () => {
			fireEvent.click(screen.getByRole("button", { name: "Check in" }));
		});
		return JSON.parse(fetch.mock.calls[0][1].body as string);
	};

	it("isn't offered while harness.glimmer_private is off, and nothing private is sent", async () => {
		flagsOn.add("harness.spark_open_triage");
		const body = await checkIn(false);
		expect(screen.queryByRole("checkbox", { name: en["newflight.private.label"] })).toBeNull();
		expect(screen.queryByText(en["newflight.private.wait"])).toBeNull();
		expect(body.private).toBeUndefined();
	});

	it("isn't offered when probe logs aren't triaged at all", () => {
		flagsOn.add("harness.glimmer_private");
		render(<NewFlight onSample={() => {}} />);
		expect(screen.queryByRole("checkbox", { name: en["newflight.private.label"] })).toBeNull();
	});

	it("says the first triage can take several minutes, and sends private only when ticked", async () => {
		flagsOn.add("harness.spark_open_triage");
		flagsOn.add("harness.glimmer_private");
		expect(await checkIn(false)).not.toHaveProperty("private");
		cleanup();
		unregister("notebook.intake");
		localStorage.clear(); // the first check-in remembered "Got it"
		const body = await checkIn(true);
		expect(body.private).toBe(true);
		const box = screen.getByRole("checkbox", { name: en["newflight.private.label"] });
		const hint = document.getElementById(box.getAttribute("aria-describedby") ?? "");
		expect(hint?.textContent).toContain(en["newflight.private.wait"]);
		expect(hint?.textContent).toMatch(/5 to 8 minutes/);
		// The map and diagnosis lines show only with their own flags on.
		expect(hint?.textContent).not.toContain(en["newflight.private.map"]);
		expect(hint?.textContent).not.toContain(en["newflight.private.diagnosis"]);
	});

	it("says Muse Spark still builds the map, and the diagnosis is skipped, when those are on", () => {
		for (const f of ["harness.spark_open_triage", "harness.glimmer_private", "harness.spark_script_map", "harness.spark_diagnosis"]) flagsOn.add(f);
		render(<NewFlight onSample={() => {}} />);
		const box = screen.getByRole("checkbox", { name: en["newflight.private.label"] });
		// They sit in the private box's fold ("What else changes"), not in the checkbox's description: a
		// closed <details> isn't read as part of one (review, Nov 2).
		const fold = box.closest(".pf-private")?.querySelector("details");
		expect(fold?.textContent).toContain(en["newflight.private.map"]);
		expect(fold?.textContent).toContain(en["newflight.private.diagnosis"]);
		const hint = document.getElementById(box.getAttribute("aria-describedby") ?? "");
		expect(hint?.contains(fold ?? null)).toBe(false);
	});
});

describe("the repo link's entry point and arguments (PM, Nov 3)", () => {
	const send = async (entry: string, args: string) => {
		const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ job_id: "01JOB" }), { status: 202 }));
		vi.stubGlobal("fetch", fetch);
		render(<NewFlight onSample={() => {}} />);
		fireEvent.click(screen.getByRole("checkbox", { name: en["datause.accept"] }));
		fireEvent.change(screen.getByLabelText(en["newflight.repo.label"]), { target: { value: "github.com/pytorch/examples" } });
		const e = screen.getByLabelText(new RegExp(`^${en["newflight.entry.label"]}`));
		const a = screen.getByLabelText(new RegExp(`^${en["newflight.args.label"]}`));
		fireEvent.change(e, { target: { value: entry } });
		fireEvent.change(a, { target: { value: args } });
		fireEvent.blur(a);
		const go = screen.getByRole("button", { name: "Check in" }) as HTMLButtonElement;
		if (go.disabled) return null;
		await act(async () => {
			fireEvent.click(go);
		});
		return JSON.parse(fetch.mock.calls[0][1].body as string);
	};

	it("sends the script and its flags as words", async () => {
		const body = await send("word_language_model/main.py", `--accel --epochs 6 --save "model a.pt"`);
		expect(body.source).toEqual({
			kind: "repo",
			uri: "https://github.com/pytorch/examples",
			entrypoint: "word_language_model/main.py",
			args: ["--accel", "--epochs", "6", "--save", "model a.pt"],
		});
	});

	it("leaves both out when empty, so Preflight picks the script", async () => {
		const body = await send("", "  ");
		expect(body.source).toEqual({ kind: "repo", uri: "https://github.com/pytorch/examples" });
	});

	it("takes a path with a space, as the runner does", async () => {
		const body = await send("Chapter 3/train.py", "");
		expect(body.source.entrypoint).toBe("Chapter 3/train.py");
	});

	it("won't send a hidden or dotted path the repo reader never keeps", async () => {
		expect(await send("src/.cfg/x.py", "")).toBeNull();
		cleanup();
		expect(await send("a/./b.py", "")).toBeNull();
	});

	it("takes ./ off the front of the path", async () => {
		const body = await send("./mnist/main.py", "");
		expect(body.source.entrypoint).toBe("mnist/main.py");
	});

	it("won't send a path outside the repo or a non-Python file, and says why", async () => {
		expect(await send("../secrets.py", "")).toBeNull();
		expect(screen.getByText(en["newflight.entry.path"])).toBeTruthy();
		cleanup();
		expect(await send("README.md", "")).toBeNull();
		expect(screen.getByText(en["newflight.entry.not_py"])).toBeTruthy();
	});

	it("won't send arguments that need a shell", async () => {
		expect(await send("main.py", "--epochs 6 | tee log")).toBeNull();
		expect(screen.getByText(/“\|” needs a shell/)).toBeTruthy();
	});
});

describe("a repo link with a commit pin (D's walkthrough, Nov 3)", () => {
	it("turns owner/repo@<sha> into the /tree/<sha> link the API reads", () => {
		const sha = "a".repeat(40);
		expect(repoUri(`github.com/pytorch/examples@${sha}`)).toBe(`https://github.com/pytorch/examples/tree/${sha}`);
		expect(repoUri(`https://github.com/karpathy/nanoGPT.git@v1`)).toBe("https://github.com/karpathy/nanoGPT/tree/v1");
		expect(repoUri("github.com/karpathy/nanoGPT")).toBe("https://github.com/karpathy/nanoGPT");
		expect(repoUri(`https://github.com/karpathy/nanoGPT/commit/${sha}`)).toBe(`https://github.com/karpathy/nanoGPT/commit/${sha}`);
	});
});

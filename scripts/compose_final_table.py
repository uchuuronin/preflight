"""Compose the final validation table (Chiya, Nov 3) from the frozen scorer's output on docs/validation/final/.

	uv run python scripts/compose_final_table.py <main.md> --first docs/validation/final/unseen_first.json \
		--after docs/validation/final/unseen_after.json [--fresh docs/validation/final/fresh.json] \
		--out docs/validation/table-final-nov3.md [--readme README.md docs/readme-final.md]

<main.md> is `scripts/score_validation.py docs/validation/final/main.json --out <md>` run with 56c2141's
`python/preflight/corpus/validate.py` and `scripts/score_validation.py` over today's engine. Every row but the unseen one
is that table's; the unseen row becomes three lines, counted from the three manifests over all nine scripts; the
cross-GPU row's AMD text and the memory and step-time notes are written here. So the scorer's own `--readme`
(`into_readme`) must never be run over the README block again: it would put back the single-manifest table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

START, END = "<!-- validation:start -->", "<!-- validation:end -->"
AMD = "22% within 20% (NVIDIA torch 2.8.0 to MI300X torch 2.5.1+rocm6.2: card and software differ)"


def wilson(k: int, n: int, z: float = 1.959963984540054) -> str:
	p = k / n
	d = 1 + z * z / n
	c = (p + z * z / (2 * n)) / d
	h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
	return f"{round(100 * (c - h))}% to {round(100 * (c + h))}%"


def rows(md: str) -> list[str]:
	return [line for line in md.splitlines() if line.startswith("| ")]


def stamp(md: str) -> str:
	return next((line for line in md.splitlines() if line.startswith('A "within" target')), "")


def unseen_line(label: str, k: int, n: int, status: str, note: str) -> str:
	hit = "hit" if n and k / n >= 8 / 9 else "miss"
	return f"| Unseen scripts harnessed: {label} | 8 of 9 without hand edits | One per job type | {k} of {n} | {wilson(k, n)} | {hit} | {status} | {note} |"


def compose(main_md: str, unseen: list[tuple[str, int, int, str, str]]) -> str:
	out = []
	for r in rows(main_md):
		if r.startswith("| Unseen scripts harnessed |"):
			out += [unseen_line(*u) for u in unseen]
			continue
		if r.startswith("| Step time, unmeasured GPU |"):
			r = r.replace("; no AMD measurements |", "; NVIDIA GPUs |")
			r = (
				r.rstrip(" |")
				+ f"; NVIDIA to MI300X, scored apart on the MI300X corpus slice (295 configs, 1,147 pairs): {AMD}; its sigma-0.22 band covers 36%, so a move across vendors abstains |"
			)
		if r.startswith("| Knob response |"):
			r = r.replace(
				"; stranger test: base https://preflight-cqn.pages.dev,",
				"; C's live-site run (remote run 41655597938, 3 samples x 20 moves at 1366x768 and 390x844; C's own p95, 67.5 ms, takes the floor(0.95 n) sample): base https://preflight-cqn.pages.dev,",
			)
		if r.startswith("| Peak memory error |"):
			r = r.rstrip(" |") + (
				"; causes: gnn_pyg (39% under; @L4 36% under): the allocator's reserved pool grows for three epochs past "
				"any probe (2.9 GB after the first step, 8.54 GB by about step 325, allocated flat at 2.62 GB); "
				"retrieval_st (57% under; @L4 37% under): the peak is in the script's own eval after training (encoding "
				"100,000 answers), which no probe reaches; vision_timm@L4 (27% over, the safe side): not traced. "
				"vision_timm's hit is at 9.96%, inside the 10% by 0.04 points |"
			)
		if r.startswith("| Step time, same GPU |"):
			r = r.rstrip(" |") + (
				"; the same GPU class: the records say A10, and vision_timm's and pretrain_nanogpt's full runs got an A10G (`bare.device`, about 11% slower on the LoRA script); causes: vision_timm (17% under, which the slower A10G pushes the same way): its 2 loader workers stall every other step once their prefetched "
				"batches run out, after the probe's 5 timed steps; retrieval_st (22% over): the recorder's hooks (1.106 "
				"measured) plus a five-step sample; pretrain_nanogpt (36% over): the probe's later untimed steps "
				"match the full run, so only its timed window ran slow; unexplained (A to log the GPU clock per step) |"
			)
		out.append(r)
	return "\n".join(out)


FOOTER = (
	"Composed by `scripts/compose_final_table.py`: every row but the unseen one is the frozen scorer's (56c2141's "
	"`validate.py` and `score_validation.py`, today's engine and production flags) on `docs/validation/final/main.json`; "
	"the three unseen lines are counted from `unseen_first.json`, `unseen_after.json`{fresh} (each script's "
	"`harnessed_without_edits`, over every script on its list, where the scorer counts only scripts that produced a record: its row "
	'read 0 of 6 for the first attempts and "not measured" after the freeze, whose ops runs made no Preflight record); '
	"the cross-GPU row's AMD text and the memory and step-time notes are written by the script. **Don't re-run "
	"`score_validation.py --readme` (`into_readme`) over this block**: it would put back a single-manifest table. "
	"**Relabelled copies:** the frozen scorer reads records up to contract 0.6.2, so a later record is scored as a "
	"copy with `schema_version` 0.6.2 (contracts 0.6.3 to 0.6.5 are additive and the frozen schema ignores what it "
	"doesn't know); the copies are committed in `docs/validation/final/files/` and listed in "
	"`docs/validation/final/RELABELLED.md`. **Memory compares unlike quantities:** the engine's point includes the "
	"CUDA context, the full run's reserved peak doesn't, so each memory error reads about 0.3 GB high, and "
	"vision_timm's hit sits at 9.96%. The scorer's own note: {stamp}"
)


def count(manifest: Path) -> tuple[int, int]:
	runs = [r for r in json.loads(manifest.read_text())["runs"] if r.get("set") == "unseen"]
	return sum(bool(r.get("harnessed_without_edits")) for r in runs), len(runs)


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("main_md", type=Path, help="the frozen scorer's table on main.json")
	ap.add_argument("--first", type=Path, required=True, help="unseen_first.json")
	ap.add_argument("--after", type=Path, required=True, help="unseen_after.json")
	ap.add_argument("--fresh", type=Path, help="fresh.json, when A's fresh runs have landed")
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--readme", type=Path, nargs="*", default=[])
	a = ap.parse_args(argv)
	main_md = a.main_md.read_text()
	k1, n1 = count(a.first)
	k2, n2 = count(a.after)
	unseen = [
		(
			"first attempts (final)",
			k1,
			n1,
			"measured",
			"the nine as sent on the frozen commit; none reached its first step",
		),
		(
			"after the freeze",
			k2,
			n2,
			"measured",
			"fixes designed on these nine (A's item-8 ops rerun on the A10 with every bring-your-own-script fix on, the "
			"list's own arguments, no hand edits; Nov 3): 4 (smolagents, its first model call), 7 (BEIR, ran out of memory), "
			"8 (OGB arxiv, all 500 epochs; its probe measures) and 9 (airbench, its probe measures) reach their first step; "
			"through production 7 (BEIR) gets as far, out of memory on the A10 (`a-fresh-live-beir-checkin` 41675905714), and "
			"8 (OGB) stopped at its dataset download after A's 10:52 AM deploy (its host is plain HTTP, which A reads as the "
			"cause, the sandbox opening only HTTPS [A's reading, not checked]; `a-fresh-live-ogb-2-checkin` 41668696736); "
			"OGB's archive is served from Preflight's dataset volume since A's 1:05 PM deploy, not yet re-checked; not a validation, "
			"since the fixes were made for them",
		),
	]
	if a.fresh:
		k3, n3 = count(a.fresh)
		unseen.append(
			(
				"fresh scripts, run once",
				k3,
				n3,
				"measured",
				"a second held-out list (the PM's, hash in `docs/validation/fresh-hash.txt`), each pinned before its probe "
				"and run once through production (A's runs, Nov 3): none reaches its first step. F1 (pytorch/examples "
				"word_language_model): the list's command passes `--cuda`, which the script at that commit renamed "
				"`--accel`; F2 (Hugging Face's text-classification notebook): its interactive `notebook_login()` cell "
				"stops it, labelled a missing package, wrongly; F3 (pytorch-cifar): its progress bar asks the terminal "
				"its size, and a probe has no terminal. The full runs as listed stop at the same lines, so there is "
				"no memory, time or out-of-memory to score",
			)
		)
	body = compose(main_md, [u for u in unseen if u[2]])
	if not a.fresh:
		body = body.replace(
			"| Knob response |",
			"| Unseen scripts harnessed: fresh scripts, run once | 8 of 9 without hand edits | One per job type | not measured |  |  | not yet run | A's runs land this afternoon |\n| Knob response |",
			1,
		)
	head = "| Metric | Target | Measured on | Result | 95% interval | Hit | Status | Notes |\n|---|---|---|---|---|---|---|---|"
	body = "\n".join(
		line
		for line in body.splitlines()
		if not line.startswith("| Metric |") and not line.startswith("|---")
	)
	foot = FOOTER.format(fresh=" and `fresh.json`" if a.fresh else "", stamp=stamp(main_md))
	table = f"{head}\n{body}\n\n{foot}\n"
	a.out.write_text(table)
	for p in a.readme:
		s = p.read_text()
		i, j = s.index(START), s.index(END)
		p.write_text(s[: i + len(START)] + "\n" + table + s[j:])
	print(table)
	return 0


if __name__ == "__main__":
	sys.exit(main())

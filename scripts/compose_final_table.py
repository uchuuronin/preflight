"""Compose the final validation table (Chiya, Nov 3) from the frozen scorer's output on docs/validation/final/.

	uv run python scripts/compose_final_table.py <main.md> <unseen_first.md> <unseen_after.md> [<fresh.md>] \
		--out docs/validation/table-final-nov3.md [--readme README.md docs/readme-final.md]

Each .md is `scripts/score_validation.py <manifest> --out <md>` with the scorer of the frozen commit 56c2141
(`python/preflight/corpus/validate.py` and `scripts/score_validation.py` at 56c2141, today's engine). The block is
hand-composed from those runs: the unseen row becomes three lines, one per manifest; the cross-GPU row's AMD text and
the memory and step-time notes are written here. So the scorer's own `--readme` (`into_readme`) must never be run
over the README block again: it would put back the single-manifest table.
"""

from __future__ import annotations

import argparse
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
		if r.startswith("| Peak memory error |"):
			r = r.rstrip(" |") + (
				"; causes: gnn_pyg (39% under; @L4 37% under): the allocator's reserved pool grows for three epochs past "
				"any probe (2.9 GB after the first step, 8.54 GB by about step 325, allocated flat at 2.62 GB); "
				"retrieval_st (57% under; @L4 37% under): the peak is in the script's own eval after training (encoding "
				"100,000 answers), which no probe reaches; vision_timm@L4 (27% over, the safe side): not traced. "
				"vision_timm's hit is at 10.0% |"
			)
		if r.startswith("| Step time, same GPU |"):
			r = r.rstrip(" |") + (
				"; causes: vision_timm (17% under): its 2 loader workers stall every other step once their prefetched "
				"batches run out, after the probe's 5 timed steps; retrieval_st (22% over): the recorder's hooks (1.106 "
				"measured) plus a five-step sample; pretrain_nanogpt (35% over): unexplained (the timed window ran slow; "
				"A to log the GPU clock per step) |"
			)
		out.append(r)
	return "\n".join(out)


FOOTER = (
	"Composed by hand (`scripts/compose_final_table.py`) from three or four runs of the frozen scorer (56c2141's "
	"`validate.py` and `score_validation.py`, today's engine and production flags) on `docs/validation/final/`: "
	"`main.json` (every row but the unseen one), `unseen_first.json`, `unseen_after.json`{fresh}. **Don't re-run "
	"`score_validation.py --readme` (`into_readme`) over this block**: it would put back a single-manifest table. "
	"**Relabelled copies:** the frozen scorer reads records up to contract 0.6.2, so a later record is scored as a "
	"copy with `schema_version` 0.6.2 (contracts 0.6.3 to 0.6.5 are additive and the frozen schema ignores what it "
	"doesn't know); the copies are committed in `docs/validation/final/files/` and listed in "
	"`docs/validation/final/RELABELLED.md`. **Memory compares unlike quantities:** the engine's point includes the "
	"CUDA context, the full run's reserved peak doesn't, so each memory error reads about 0.3 GB high, and "
	"vision_timm's hit sits at exactly 10.0%. The unseen lines count all nine scripts, where the scorer counts only "
	"those that produced a record (its row read 0 of 6 for the first attempts). The scorer's own note: {stamp}"
)


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("main_md", type=Path)
	ap.add_argument("first_md", type=Path)
	ap.add_argument("after_md", type=Path)
	ap.add_argument("fresh_md", type=Path, nargs="?")
	ap.add_argument("--fresh-count", default=None, help="k/n for the fresh scripts, e.g. 1/3")
	ap.add_argument("--out", type=Path, required=True)
	ap.add_argument("--readme", type=Path, nargs="*", default=[])
	a = ap.parse_args(argv)
	main_md = a.main_md.read_text()
	unseen = [
		(
			"first attempts (final)",
			0,
			9,
			"measured",
			"the nine as sent on the frozen commit; none reached its first step",
		),
		(
			"after the freeze",
			3,
			9,
			"measured",
			"fixes designed on these nine (A's ops runs with the install flags, no hand edits): 4 (smolagents, its first "
			"model call), 7 (BEIR, ran out of memory) and 8 (OGB arxiv, all 500 epochs) reach their first step; not a "
			"validation, since the fixes were made for them",
		),
	]
	if a.fresh_count:
		k, n = (int(x) for x in a.fresh_count.split("/"))
		unseen.append(("fresh scripts, run once", k, n, "measured",
			"a second held-out list (PM, hash `docs/validation/fresh-hash.txt`), each run once through production"))  # fmt: skip
	else:
		unseen.append(("fresh scripts, run once", 0, 0, "not yet run", "A's runs land this afternoon"))
	body = compose(main_md, [u for u in unseen if u[2]])
	if not a.fresh_count:
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
	foot = FOOTER.format(fresh=", `fresh.json`" if a.fresh_md else "", stamp=stamp(main_md))
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

"""Score the model's answers (corpus/ask_model.py) with the validation scorer's own rows, beside Preflight's.

	uv run --extra probe python -m preflight.corpus.ask_model_score <manifest.json> <ask_model.json> [--out table.md]

The manifest is the validation manifest (corpus/validate.py), with each run's `ask_model_id` naming its case in the
model's answers. Both sides go through `validate.build` on the same manifest, so they score the same runs with the
same rules and the same calibration exclusions; only where the numbers come from differs. For the model's side each
run's Scored is filled from its answer instead of `api.predict`. Preflight's side should be run from the frozen
checkout (the unseen rule); this module only reads the table it is given.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from unittest import mock

from preflight.corpus import validate as V

GIB = 2**30
ROWS = ("Peak memory error", "OOM prediction", "Step time, same GPU", "Interval coverage")


class _Est:
	"""What validate._inside and the row code read off an estimate."""

	def __init__(self, point, lo, hi):
		self.point, self.lo, self.hi, self.status = point, lo, hi, "predicted"


def well_formed(ans: dict) -> str | None:
	"""Why an answer can't be scored, or None (decided with the prompt: ask_model.py)."""
	for key in ("peak_memory_gib", "step_time_s"):
		iv = ans.get(key) or {}
		if not all(
			isinstance(iv.get(k), int | float) and math.isfinite(iv[k]) for k in ("point", "lo", "hi")
		):
			return f"{key} is not three finite numbers"
		if not iv["lo"] <= iv["point"] <= iv["hi"] or iv["lo"] < 0:
			return f"{key} is not lo <= point <= hi with lo >= 0"
	p = ans.get("p_oom")
	if not isinstance(p, int | float) or not 0 <= p <= 1:
		return "p_oom is not a probability"
	return None


def scored_from_answer(run: dict, base: Path, answer: dict | None) -> V.Scored:
	"""The model's Scored for one manifest run, by the same rules as validate.score_run."""
	s = V.Scored(run.get("script", "?"), run.get("set", "zoo"))
	gt_path = base / run["ground_truth"] if run.get("ground_truth") else None
	if gt_path is None or not gt_path.exists():
		s.notes.append(f"{s.script}: not measured, no ground truth yet")
		return s
	gt = json.loads(gt_path.read_text())
	# review, Nov 2: no answer is held like Preflight's abstain: a run that ran out still counts, as a missed OOM
	s.oom_actual = V.ran_out(gt)
	if answer is None:
		s.notes.append(f"{s.script}: the model gave no answer")
		return s
	why = well_formed(answer)
	if why:
		s.notes.append(f"{s.script}: malformed answer ({why})")
		return s
	bare = gt.get("bare") or {}
	m = answer["peak_memory_gib"]
	mem = _Est(m["point"] * GIB, m["lo"] * GIB, m["hi"] * GIB)
	peak = bare.get("peak_reserved_bytes")
	if bare.get("ended") in V.FINISHED and peak:
		s.memory = V._within(mem.point, peak, 0.10)
		s.memory_inside = V._inside(mem, peak)
	s.oom_p = float(answer["p_oom"])
	actual_step = V.steady_step_s(gt)
	if answer["step_applies"] and actual_step:
		st = answer["step_time_s"]
		e = _Est(st["point"], st["lo"], st["hi"])
		# the model is asked about the GPU the full run used, so every step comparison is same-GPU
		s.step_same = V._within(e.point, actual_step, 0.15)
		s.step_inside = V._inside(e, actual_step)
	return s


def build(manifest: dict, base: Path, answers: dict[str, dict]) -> list[V.Row]:
	# per run, not per script (review, Nov 2: two runs of one script are two answers); validate.build passes the
	# manifest's own run dicts, so each run object is its own key
	by_run = {id(r): answers.get(r.get("ask_model_id") or r.get("script")) for r in manifest.get("runs", [])}

	def fake(run, base_):
		return scored_from_answer(run, base_, by_run.get(id(run)))

	with mock.patch.object(V, "score_run", fake):
		rows, _ = V.build(manifest, base, None, None)
	return [r for r in rows if r.metric in ROWS]


def side_by_side(pf: list[V.Row], model: list[V.Row]) -> str:
	head = "| Metric | Target | Preflight | 95% interval | Hit | Asking Muse Spark | 95% interval | Hit |"
	lines = [head, "|---|---|---|---|---|---|---|---|"]
	mrow = {r.metric: r for r in model}
	for r in pf:
		if r.metric not in ROWS:
			continue
		a, b = (
			r.cells(),
			mrow[r.metric].cells() if r.metric in mrow else ["", "", "", "not asked", "", "", "", ""],
		)
		lines.append(f"| {a[0]} | {a[1]} | {a[3]} | {a[4]} | {a[5]} | {b[3]} | {b[4]} | {b[5]} |")
	return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
	ap = argparse.ArgumentParser()
	ap.add_argument("manifest", type=Path)
	ap.add_argument("answers", type=Path)
	ap.add_argument("--out", type=Path)
	args = ap.parse_args(argv)
	manifest = json.loads(args.manifest.read_text())
	got = json.loads(args.answers.read_text())
	answers = {r["id"]: r.get("answer") for r in got["rows"]}
	rows = build(manifest, args.manifest.parent, answers)
	text = V.markdown(
		rows, f"Muse Spark ({got.get('model')}, prompt {got.get('prompt')} {got.get('prompt_hash')})."
	)
	if args.out:
		args.out.write_text(text)
	print(text)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())

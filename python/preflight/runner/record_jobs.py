"""Jobs on an existing record: the guide's run_probe, run_verify and sample_more (C's request).

A job whose source is `{kind: "record", record_ref}` reuses that record's script,
env and data instead of a new upload. It never changes the record it starts
from: a shared link stays what its owner saw, and two jobs on one record can't
overwrite each other. It writes a new record whose `parent_id` is the old one,
carrying the parent's map, knobs and probes plus the new measurement, and its
share token arrives on the job's events as usual.

  record_ref = share token   the stored record (records volume); its script map
                             and knobs carry over, so no Muse Spark call is made
  record_ref = sample:<id>   one of C's bundled samples: its zoo script, mapped
                             fresh by the static reader; parent_id = the ref
"""

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError

from preflight.probes.local import InvalidSubmission
from preflight.schema import Evidence, Finding, FlightRecord, Knob, Probe, Source
from preflight.schema.ids import new_share_token, new_ulid

# cited: C's bundled samples (api/preflight_api/stores.py SAMPLES, Oct 27) and the zoo script each
# record's entrypoint names (zoo/<dir>/train.py); the agent sample has no script to measure
SAMPLE_REFS = {
	"lora-7b": "finetune_lora_hf",
	"lora-qwen": "finetune_lora_hf",
	"pretrain": "pretrain_nanogpt",
	# C, Nov 3: the Fix button on these samples' findings (eval batch, bf16). Their data is on Hugging Face or made in
	# the script (gnn_pyg's FakeDataset), which the probe reaches. GRPO and the agent sample stay out: a GRPO fix race
	# verifies at two seeds on an A10 (dollars, not cents), and an agent run isn't fixed by a GPU probe
	"retrieval": "retrieval_st",
	"gnn": "gnn_pyg",
	"vision": "vision_timm",
}
VERIFY_WARMUP = 2  # simulated: probe-plan D1 (same warmup as every probe)
VERIFY_STEPS = 20  # guess: long enough to see the loss move past warmup; set from the zoo verify runs
VERIFY_SEED = 20261104  # derived: any fixed value; what matters is the plan and the base share it
VERIFY_SEED_2 = 20261105  # derived: the base again at another seed, to measure seed-to-seed spread
# The pass line is measured per job, never a fixed percentage (Chiya, Oct 27): the current setting
# runs at two seeds, and the plan passes only if its loss stays as close to the current setting at
# the same seed as the two seeds stay to each other. Seeds are forced for these runs (a script's
# own set_seed is overridden), so the spread is the seed's, not zero.
UPLOAD_KINDS = ("upload", "cli", "inflight", "notebook")


@dataclass
class Origin:
	fetch_source: dict  # what sources.fetch() reads
	parent: FlightRecord | None  # the stored record, when the ref is a share token
	parent_id: str | None


def resolve(source: dict, *, reader, blobs) -> Origin:
	ref = source.get("record_ref")
	if not isinstance(ref, str) or not ref:
		raise InvalidSubmission("a record source needs record_ref", code="submission.invalid")
	if ref.startswith("sample:"):
		sid = ref.removeprefix("sample:")
		zoo = SAMPLE_REFS.get(sid)
		if zoo is None:
			raise InvalidSubmission(
				f"the sample {sid!r} has no script Preflight can measure", code="submission.sample_unknown"
			)
		return Origin({"kind": "sample", "uri": f"sample://zoo/{zoo}"}, None, ref)
	parent = reader.get(ref) if reader is not None else None
	if parent is None:
		raise InvalidSubmission("no record has that link", code="submission.record_unknown")
	return Origin(refetch(parent.job.source, blobs), parent, parent.record_id)


def refetch(src: Source, blobs) -> dict:
	"""The fetch() source that reads a record's code again, exactly as it was read the first time."""
	base: dict = {"args": list(src.args)}
	if src.entrypoint and not (src.kind == "sample" and "/" in src.entrypoint):
		base["entrypoint"] = src.entrypoint  # a sample's tree is read from its own directory
	if src.scope == "entrypoint" and "entrypoint" in base:  # 0.6.3: read the same files the first time read
		base["scope"] = "entrypoint"
	if src.extra_packages:  # and install the same extra packages
		base["extra_packages"] = list(src.extra_packages)
	if src.kind in ("sample", "template"):
		return {"kind": src.kind, "uri": src.uri, **base}
	if src.kind == "repo":
		return {"kind": "repo", "uri": src.uri, "commit": src.commit, **base}
	if src.kind == "command":  # the command again (its args are the whole command line)
		return {
			"kind": "command",
			"args": list(src.args),
			**{k: v for k, v in base.items() if k == "extra_packages"},
		}
	if src.kind in UPLOAD_KINDS:
		ref = (src.uri or "").removeprefix("upload://")
		if not ref.startswith("sha256:") or blobs is None:
			raise InvalidSubmission(
				"this record's files weren't kept, so it can't be measured again; check the script in again",
				code="submission.files_gone",
			)
		return {"kind": src.kind, "files_ref": ref, **base}
	raise InvalidSubmission(f"a {src.kind} record can't be measured again", code="submission.invalid")


def derive(parent: FlightRecord, ids: tuple[str, str] | None = None) -> FlightRecord:
	"""A new record that starts as the parent: same map, knobs, probes and costs so far."""
	record_id, share_token = ids if ids and all(ids) else (new_ulid(), new_share_token())
	return parent.model_copy(
		update={
			"record_id": record_id,
			"share_token": share_token,
			"parent_id": parent.record_id,
			"created_at": datetime.now(UTC),
		}
	)


def setting(rec: FlightRecord, knobs: dict | None) -> tuple[dict, list[str]]:
	"""The knobs the record declares, each checked against its domain; and the ids it doesn't know."""
	known = {k.id: k for k in rec.knobs}
	out, unknown = {}, []
	for kid, value in (knobs or {}).items():
		k = known.get(kid)
		if k is None:
			unknown.append(kid)
			continue
		try:
			Knob.model_validate({**k.model_dump(), "current": value})
		except ValidationError as exc:
			raise InvalidSubmission(
				f"{kid}={value!r} is outside what this knob takes", code="submission.knob_value"
			) from exc
		out[kid] = value
	return out, unknown


def full_setting(rec: FlightRecord, knobs: dict) -> dict:
	return {**{k.id: k.current for k in rec.knobs}, **knobs}


def _series(p: Probe) -> list[float | None]:
	"""The loss per recorded step, None where it wasn't finite (the recorder stores NaN as None)."""
	raw = p.stability.series.get("loss", []) if p.stability else []
	return [v if v is not None and math.isfinite(v) else None for v in raw]


def _losses(p: Probe) -> list[float]:
	return [v for v in _series(p) if v is not None]


def _eval_oom(p: Probe) -> bool:
	"""Ran out of memory in the eval pass (A's error code), after training steps."""
	return p.status == "oom" and (p.error_code or "").startswith("oom.eval")


def _diverged(p: Probe) -> bool:
	"""A NaN or inf loss anywhere, or a step with no finite loss: the run didn't train cleanly."""
	st = p.stability
	return bool(st and (st.nan_count or st.inf_count)) or any(v is None for v in _series(p))


def _effective_batch(knobs: dict) -> float | None:
	bs = knobs.get("batch_size")
	if not isinstance(bs, int | float) or isinstance(bs, bool):
		return None
	out = float(bs)
	for k in ("grad_accum", "gpus"):
		v = knobs.get(k)
		if isinstance(v, int | float) and not isinstance(v, bool):
			out *= v
	return out


def loss_delta(plan: Probe, base: Probe) -> float | None:
	"""Mean relative difference of the two runs' losses, step by step, where both have one."""
	pairs = [
		(x, y) for x, y in zip(_series(plan), _series(base), strict=False) if x is not None and y is not None
	]
	if not pairs:
		return None
	scale = sum(abs(y) for _, y in pairs) / len(pairs)
	if scale == 0:
		return 0.0 if all(x == y for x, y in pairs) else math.inf
	return sum(abs(x - y) for x, y in pairs) / len(pairs) / scale


def verify_finding(
	plan: Probe,
	base: Probe | None,
	plan_knobs: dict,
	base_knobs: dict,
	node_ids: set[str],
	other_seed: Probe | None = None,
) -> Finding:
	"""Did the planned setting run, and does it learn what the current one does?

	plan and base share a seed; other_seed is the base at a second seed. The plan's loss "holds"
	when its mean step-by-step distance from the base is within the base's own seed-to-seed spread
	(the mean distance between base and other_seed), both measured in this job."""
	node = "loss" if "loss" in node_ids else None
	fid = f"verify.{plan.id}"
	steps = len(_losses(plan))

	def finding(severity, code, title, detail=None, evidence=()):
		return Finding(id=fid, severity=severity, code=code, title=title, detail=detail, node_id=node,
			evidence=list(evidence), source="rule")  # fmt: skip

	if _eval_oom(plan):
		# A's verify design (Nov 1): the window ends with one eval on every side, and running out
		# there is the side's result, never re-run with the eval off
		both = base is not None and _eval_oom(base)
		return finding("error", "verify.failed",
			"The eval ran out of memory with and without the change" if both else "The eval ran out of memory with the change",
			detail=plan.error or "oom.eval",
			evidence=[Evidence(metric="probe.eval.peak_allocated_bytes",
				value=plan.eval.peak_allocated_bytes if plan.eval else None, probe_id=plan.id)])  # fmt: skip
	if plan.status not in ("ok", "cached") or steps == 0:
		why = plan.error or plan.status
		return finding("error", "verify.failed", "The verify run didn't finish", detail=why)
	if _diverged(plan):
		st = plan.stability
		return finding("error", "verify.diverged", "The loss went NaN or infinite in the verify run",
			detail=f"{st.nan_count if st else 0} NaN and {st.inf_count if st else 0} inf values; "
			f"{steps} of {len(_series(plan))} steps had a finite loss",
			evidence=[Evidence(metric="probe.stability.nan_count", value=st.nan_count if st else None, probe_id=plan.id)])  # fmt: skip
	if base is None:
		return finding("info", "verify.passed", f"The plan ran {steps} steps at a fixed seed with a finite loss",
			evidence=[Evidence(metric="probe.stability.loss", value=_losses(plan)[-1], probe_id=plan.id)])  # fmt: skip

	def usable(p):
		# the current settings running out at the window's eval still trained up to it: their loss
		# steps before the eval are compared (A's verify design, Nov 1)
		ran = p is not None and (p.status in ("ok", "cached") or _eval_oom(p))
		return ran and _losses(p) and not _diverged(p)

	if not usable(base) or not usable(other_seed):
		bad = base if not usable(base) else other_seed
		why = (bad.error or bad.status) if bad is not None else "not run"
		return finding("warn", "verify.no_baseline", "The plan ran, but the current settings didn't run "
			"cleanly at both seeds, so there's nothing to compare its loss with", detail=why)  # fmt: skip
	delta = loss_delta(plan, base)
	noise = loss_delta(other_seed, base)

	def num(x):
		return None if x is None or math.isinf(x) else round(x, 6)

	ev = [
		Evidence(metric="verify.loss_delta", value=num(delta), threshold=num(noise), probe_id=plan.id),
		Evidence(metric="verify.seed_spread", value=num(noise), probe_id=other_seed.id),
	]
	pct = "an unbounded amount" if delta is None or math.isinf(delta) else f"{delta:.2%}"
	spread = "none" if noise is None else f"{noise:.2%}"
	if delta is not None and noise is not None and delta <= noise:
		return finding("info", "verify.passed", "The loss holds: it stays within the current setting's "
			"own seed-to-seed spread", detail=f"mean step-by-step difference {pct} over {steps} steps; two seeds "
			f"of the current setting differ by {spread} (measured in this job)", evidence=ev)  # fmt: skip
	eb_plan, eb_base = _effective_batch(plan_knobs), _effective_batch(base_knobs)
	if eb_plan is not None and eb_base is not None and eb_plan != eb_base:
		why = (f"The effective batch changed from {eb_base:g} to {eb_plan:g}, so some change is expected; "
			"keep it the same (more accumulation for a smaller batch) to learn the same thing.")  # fmt: skip
	else:
		why = "With the same data, seed and effective batch, the change alters what the model learns."
	return finding("warn", "verify.loss_changed", f"Loss moved {pct} from the current settings at the same "
		f"seed, more than the {spread} two seeds differ by", detail=why, evidence=ev)  # fmt: skip

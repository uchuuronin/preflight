"""Spark Open triage of a probe's log (plan item 29): one call per probe, every call in provenance.

	finding, call = triage_probe(ModelClient.baseten_spark_open(), probe, log_tail)

Spark Open (open weights, on Baseten) reads the tail of the probe's own output and its status, and
says what happened in one line: the category comes from a fixed list and the line it rests on is
quoted from the log, so a reader can check it. Code decides what that becomes: for a probe that
didn't finish, an info finding beside the probe's own error; for one that did, the call alone, in
provenance. Effort is low: triage is a short read, not a diagnosis. The log may hold what a
visitor's script printed, so the runner calls this only with harness.spark_open_triage on, which
the data-use notice must disclose (as harness.spark_script_map is).
"""

from typing import Literal, get_args

from pydantic import BaseModel, Field

from preflight.schema import Evidence, Finding

TAIL_CHARS = 3000  # the end of the log: where a traceback or the last progress line is
ERROR_CHARS = 1500  # the recorded error: the classifier's line and the exception's own text
Category = Literal["ok", "oom", "missing_package", "data_error", "nan_or_divergence", "timeout", "script_error",
	"infrastructure", "unclear"]  # fmt: skip


class Triage(BaseModel):
	category: Category
	summary: str = Field(description="one sentence, under 160 characters, plain words")
	evidence_line: str = Field(description="the log line the summary rests on, copied exactly, or empty")


SYSTEM = ("You triage the output of a short probe run of a machine-learning script. Say what happened, "
	"from the log only: a category from the list, one plain sentence, and the one log line it rests on, "
	"copied exactly. Never guess beyond the log; use 'unclear' when it doesn't say.")  # fmt: skip
# the answer's shape in words too: a model served without guided decoding (Glimmer on vLLM, Nov 1)
# doesn't see the response_format schema, and answered with the probe's error code as the category
ANSWER = ('Answer with JSON only: {"category": one of ' + ", ".join(f'"{c}"' for c in get_args(Category))
	+ ', "summary": "<one sentence>", "evidence_line": "<the log line, copied exactly, or empty>"}')  # fmt: skip


def _who(client) -> str:
	"""The model the finding names: Glimmer for a private job (0.6.2), else Spark Open."""
	return "Muse Glimmer" if getattr(client, "provider", "") == "modal-glimmer" else "Spark Open"


def triage_probe(client, probe, log_tail: str, node_id: str | None = None) -> tuple[Finding | None, list]:
	"""(an info finding for a probe that didn't finish, or None; the model calls made)."""
	# the recorder catches what ended the probe (an OOM in eval, say), so its traceback is in the
	# probe's recorded error, not the stdout tail (Nov 1 production run): both go in, both can be quoted
	tail = (log_tail or "")[-TAIL_CHARS:]
	if probe.error:
		tail += "\n[the probe's recorded error]\n" + probe.error[:ERROR_CHARS]
	status = f"status {probe.status}" + (f", error code {probe.error_code}" if probe.error_code else "")
	msgs = [{"role": "system", "content": SYSTEM},
		{"role": "user", "content": f"Probe {probe.id}: {status}, {probe.steps} optimizer steps.\nLog tail:\n{tail}\n\n{ANSWER}"}]  # fmt: skip
	before = len(client.calls)
	try:
		t, _ = client.structured(msgs, Triage, purpose="triage", reasoning_effort="low")
	except Exception:
		return None, client.calls[before:]  # failed attempts stay in provenance; the probe stands as it is
	calls = client.calls[before:]
	if probe.status in ("ok", "cached"):
		return None, calls
	line = t.evidence_line.strip()
	quoted = len(line) >= 8 and line in tail  # code checks the quote is really in the log, and is a line
	detail = t.summary.strip()[:200] + (f' (log: "{line[:160]}")' if quoted else " (no log line quoted)")
	return Finding(id=f"triage.{probe.id}", severity="info", code=f"triage.{t.category}",
		title=f"{_who(client)} on probe {probe.id}: {t.category.replace('_', ' ')}", detail=detail,
		node_id=node_id, source="spark_open_triage",
		# tied to its probe, so a follow-up that drops the probe (another GPU) drops this too
		evidence=[Evidence(metric="triage", value=1.0, probe_id=probe.id)]), calls  # fmt: skip

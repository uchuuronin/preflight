"""Every failure names its step and a way past it by hand (plan item 43)."""

import re
from pathlib import Path

from preflight.runner.failures import FALLBACK, describe, with_step

SRC = Path(__file__).resolve().parents[1] / "python" / "preflight"
# where a failure's code is set: the recorder's classifier, the job's errors, a probe's error_code
PATTERNS = (r'Classified\(\s*"([a-z_.]+)"', r'(?:InvalidSubmission|PermanentError)\([^()]*?code="([a-z_.]+)"',
	r'"error_code":\s*"([a-z_.]+)"', r'error_code="([a-z_.]+)"',
	r'status, code(?:, error)? = \(?\s*"[a-z]+",\s*"([a-z_.]+)"')  # fmt: skip


def produced_codes() -> set[str]:
	codes = set()
	for f in [*SRC.glob("recorder/*.py"), *SRC.glob("probes/*.py"), *SRC.glob("runner/*.py")]:
		text = f.read_text()
		for pat in PATTERNS:
			codes |= set(re.findall(pat, text, re.S))
	return codes


def test_every_error_code_the_code_produces_has_a_step_and_a_manual_entry_point():
	codes = produced_codes()
	assert {
		"oom",
		"recorder",
		"submission.invalid",
		"missing_package.install",
	} <= codes  # the scan finds them
	assert sorted(c for c in codes if describe(c) == FALLBACK) == []


def test_codes_match_by_their_longest_known_prefix():
	assert describe("oom.eval")[0] == "the eval pass"
	assert describe("oom.host")[0].startswith("host memory")
	assert describe("oom.something_new") == describe("oom")
	assert describe("submission.brand_new") == describe("submission")
	assert describe(None) == FALLBACK and describe("never_seen") == FALLBACK


def test_the_message_names_the_step_once_with_the_scripts_own_file():
	m = with_step("Ran out of GPU memory at step 3.", "oom", entrypoint="finetune.py")
	assert m.startswith("Ran out of GPU memory at step 3. Step: the training steps. By hand: ")
	assert "preflight run finetune.py" in m and "train.py" not in m
	assert with_step(m, "oom") == m  # idempotent


def test_every_code_a_visitor_can_meet_has_its_own_step_not_only_its_familys():
	# the overnight pass (Nov 2): the submission, patch and demo-limit codes had only their family's
	# line ("checking the submission"), which says nothing a visitor can act on
	from preflight.runner.failures import STEPS

	text = "".join(f.read_text() for f in SRC.glob("runner/*.py"))
	codes = set(re.findall(r'code="((?:submission|patch|limit)\.[a-z_]+)"', text))
	codes |= {"queue_full"}
	assert {
		"limit.visitor_daily",
		"limit.daily_spend",
		"submission.sample_full",
		"patch.does_not_apply",
	} <= codes
	missing = sorted(c for c in codes if c not in STEPS)
	assert missing == [], missing
	step, by_hand = describe("limit.visitor_daily")
	assert "3 probe jobs" in step and "midnight ET" in by_hand

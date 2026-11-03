"""probes.torch_new (B's review of A's fixes, Nov 3; the PM's fix): when the torch 2.7.1 image overrides any of
the code's own pins, the record says each one ("measured on transformers 4.52.3; your code pins 4.46.3") and every
estimate carries it in its reason."""

from pathlib import Path

from preflight.engine import api
from preflight.recorder import requirements as R
from preflight.runner.images import PROBE_NEW_PACKAGES
from preflight.schema import FlightRecord

ZOO = Path(__file__).resolve().parent / "fixtures" / "validation" / "zoo_gt"
PINNED = "torch==2.6.0\ntransformers==4.46.3\ntrl==0.14.0\naccelerate>=0.30\n"
TRAIN = {"train.py": "import torch\nfor step in range(10):\n    pass\n"}


def test_each_overridden_pin_is_named_with_both_versions():
	over = R.image_overrides({"requirements.txt": PINNED}, PROBE_NEW_PACKAGES)
	assert [(n, v, s) for n, v, s in over] == [
		("torch", "2.7.1", "==2.6.0"),
		("transformers", "4.52.3", "==4.46.3"),
		("trl", "0.18.0", "==0.14.0"),
	]  # accelerate>=0.30 admits the image's 1.7.0: not an override
	note = R.overrides_note(over)
	assert "measured on transformers 4.52.3; your code pins 4.46.3" in note
	assert "measured on trl 0.18.0; your code pins 0.14.0" in note
	assert R.overrides_note([("transformers", "4.52.3", "<4.50")]) == (
		"measured on transformers 4.52.3; your code asks for transformers<4.50"
	)


def test_no_override_and_a_marker_or_unknown_package_say_nothing():
	files = {
		"requirements.txt": "torch>=2.7\ntransformers>=4.40\nfoo==1.0\ntrl==0.14.0; sys_platform == 'win32'\n"
	}
	assert R.image_overrides(files, PROBE_NEW_PACKAGES) == []


def test_the_pyproject_s_pins_count_too():
	files = {
		"pyproject.toml": '[project]\nname = "x"\nversion = "1"\ndependencies = ["torch>=2.6", "peft==0.13.2"]\n'
	}
	assert R.image_overrides(files, PROBE_NEW_PACKAGES) == [("peft", "0.15.2", "==0.13.2")]


def _fetched(files):
	from preflight.runner.sources import Fetched

	return Fetched(files=files, entrypoint="train.py")


def test_the_record_gets_a_visible_finding_only_on_the_newer_image(monkeypatch):
	import preflight.runner.pipeline as P
	from preflight.harness.static import map_static

	files = {**TRAIN, "requirements.txt": PINNED}
	sm = map_static(files, entrypoint="train.py")
	monkeypatch.setattr(P, "_flag", lambda name: False)
	assert P._image_findings(sm, _fetched(files)) == []  # the default image: nothing new said
	monkeypatch.setattr(P, "_flag", lambda name: name == "probes.torch_new")
	(f,) = P._image_findings(sm, _fetched(files))
	assert f.id == P.IMAGE_OVERRIDES == api.IMAGE_OVERRIDES and f.severity == "warn"
	assert f.title.startswith("Measured on torch 2.7.1; your code pins 2.6.0")
	assert "measured on transformers 4.52.3; your code pins 4.46.3" in f.title
	assert "transformers ==4.46.3 (measured on 4.52.3)" in f.detail
	clean = {**TRAIN, "requirements.txt": "torch>=2.7\ntransformers>=4.40\n"}
	assert P._image_findings(sm, _fetched(clean)) == []  # the newer image, no pin overridden


def test_every_estimate_carries_it_in_its_reason(monkeypatch):
	import preflight.runner.pipeline as P
	from preflight.harness.static import map_static

	rec = FlightRecord.model_validate_json((ZOO / "vision_timm_a10.record.json").read_text())
	before = api.predict(rec).root
	files = {**TRAIN, "requirements.txt": PINNED}
	monkeypatch.setattr(P, "_flag", lambda name: name == "probes.torch_new")
	(f,) = P._image_findings(map_static(files, entrypoint="train.py"), _fetched(files))
	after = api.predict(rec.model_copy(update={"findings": [*rec.findings, f]})).root
	assert set(after) == set(before)
	for k, e in after.items():
		if e.status == "abstain":
			assert e.reason == before[k].reason
		else:
			assert "your code pins 4.46.3" in e.reason and (e.point, e.lo, e.hi) == (
				before[k].point,
				before[k].lo,
				before[k].hi,
			)
	assert any(e.status != "abstain" for e in after.values())


def test_the_stored_predictions_carry_it_too(monkeypatch):
	import preflight.runner.pipeline as P
	from preflight.harness.static import map_static

	rec = FlightRecord.model_validate_json((ZOO / "vision_timm_a10.record.json").read_text())
	files = {**TRAIN, "requirements.txt": PINNED}
	monkeypatch.setattr(P, "_flag", lambda name: name == "probes.torch_new")
	(f,) = P._image_findings(map_static(files, entrypoint="train.py"), _fetched(files))
	filled = api.fill_predictions(rec.model_copy(update={"findings": [*rec.findings, f]}))
	est = filled.predictions.estimates["memory.peak"]
	assert est.status != "abstain" and "your code pins 4.46.3" in est.reason


def test_a_base_file_s_pins_and_a_triple_equals_pin_are_read_review():
	files = {
		"requirements.txt": "-r requirements/base.txt\n",
		"requirements/base.txt": "transformers===4.46.3\n",
	}
	over = R.image_overrides(files, PROBE_NEW_PACKAGES)
	assert over == [("transformers", "4.52.3", "===4.46.3")]
	assert R.overrides_note(over) == "measured on transformers 4.52.3; your code pins 4.46.3"

"""After the freeze (Nov 2): a GPU move within the probes' vendor and software stack keeps their measured context
(engine.context_same_stack), instead of the 0.81 GB guess."""

from contextlib import contextmanager
from pathlib import Path

from preflight.engine import api, memory
from preflight.schema import load_record

ROOT = Path(__file__).resolve().parents[1]
RETRIEVAL_L4 = ROOT / "zoo/retrieval_st/record_nov1_l4.json"
AMD = ROOT / "tests/fixtures/validation/amd/pretrain_nanogpt_mi300x.record.json"


@contextmanager
def _on(on: bool):
	token = api._FEATURES_NOW.set({**api.FEATURES, "context_same_stack": on})
	try:
		yield
	finally:
		api._FEATURES_NOW.reset(token)


def test_a_move_within_the_stack_keeps_the_measured_context_only_with_the_flag():
	rec = load_record(RETRIEVAL_L4.read_text())
	measured = api._context_bytes(rec)
	assert 0 < measured < memory.DEFAULT_CONTEXT_BYTES  # the probes logged whole-device use
	with _on(False):
		assert api._context_bytes(rec, "A10") == memory.DEFAULT_CONTEXT_BYTES
		off = api.predict(rec, {"gpu_type": "A10"}).root["memory.peak"].point
	with _on(True):
		scale = api._context_scale(rec, "A10")  # 72 SMs against the L4's 58
		assert 1.0 < scale < 1.3 and abs(api._context_bytes(rec, "A10") - measured * scale) < 1
		on = api.predict(rec, {"gpu_type": "A10"}).root["memory.peak"].point
		here = api.predict(rec).root["memory.peak"].point
		assert api._context_scale(rec, "T4") == 1.0  # never scaled down on a smaller card
	assert abs(off - on - (memory.DEFAULT_CONTEXT_BYTES - measured * scale)) < 0.05e9
	assert abs(on - here - measured * (scale - 1)) < 0.05e9  # the move adds only the bigger card's share


def test_another_vendor_or_stack_keeps_the_guess():
	rec = load_record(RETRIEVAL_L4.read_text())
	with _on(True):
		assert not api._same_stack(rec, "MI300X")
		assert api._context_bytes(rec, "MI300X") == memory.DEFAULT_CONTEXT_BYTES
		rocm = rec.model_copy(update={"env": rec.env.model_copy(update={"backend": "rocm"})})
		assert not api._same_stack(rocm, "A10")  # an NVIDIA card under a ROCm stack isn't the probes' stack
	amd = load_record(AMD.read_text())
	assert api._same_stack(amd, "MI210")  # another AMD card under the same ROCm stack
	assert not api._same_stack(amd, "A10")

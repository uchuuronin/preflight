"""Each training framework's own loop, probed on CPU: the recorder counts its optimizer steps,
splits the phases and reads the run's planned length. Skipped where the framework isn't installed."""

from pathlib import Path

import pytest

pytest.importorskip("torch")
from preflight.probes.local import ProbeSpec, run_probe

FIX = Path(__file__).resolve().parent / "fixtures" / "adapters"


@pytest.mark.parametrize(
	("script", "needs"),
	[("hf_trainer.py", "transformers"), ("trl_sft.py", "trl"), ("lightning_loop.py", "lightning")],
)
def test_a_framework_loop_is_probed(script, needs):
	pytest.importorskip(needs)
	if needs == "trl":
		pytest.importorskip("datasets")
	r = run_probe(ProbeSpec(probe_id="p1", files={script: (FIX / script).read_text()}, entrypoint=script, warmup=2,
		timed=5, timeout_s=300))  # fmt: skip
	p = r.probe
	assert p.status == "ok", p.error
	assert (
		p.steps >= 7
		and p.step_time.by_phase.get("forward", 0) > 0
		and p.step_time.by_phase.get("backward", 0) > 0
	)
	assert p.extras.get("planned_steps") == 40

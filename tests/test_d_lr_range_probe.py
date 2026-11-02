"""The learning-rate range test through the probe's subprocess, on CPU (D, Nov 2)."""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

from preflight.probes.local import ProbeSpec, run_probe
from preflight.probes.lr_range import curve, lr_at

FIX = Path(__file__).resolve().parent / "fixtures"
pytestmark = pytest.mark.torch
RAMP = {"start": 1e-6, "end": 10.0, "steps": 60}  # shorter than the real 100, for CI

LOOP = """
import torch
torch.manual_seed(0)
w = torch.randn(16, 1)
model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
opt = torch.optim.Adam([
	{"params": model[0].parameters(), "lr": 1e-3},
	{"params": model[2].parameters(), "lr": 5e-4},
])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5)  # the script's own schedule, overridden
LOW = float("inf")
for step in range(2000):
	x = torch.randn(32, 16)
	loss = torch.nn.functional.mse_loss(model(x), x @ w)
	LOW = min(LOW, loss.item()) if torch.isfinite(loss) else LOW
	if STOP_ON_BLOWUP and step >= 10 and (not torch.isfinite(loss) or loss.item() > 2 * LOW):
		raise ValueError("the loss blew up")
	opt.zero_grad()
	loss.backward()
	opt.step()
	sched.step()
"""


def probe(script: str, **kw):
	files = {"train.py": script}
	return run_probe(ProbeSpec(probe_id="p1-lr", files=files, entrypoint="train.py", warmup=RAMP["steps"],
		timed=0, eval_batches=0, lr_range=RAMP, timeout_s=180, **kw))  # fmt: skip


@pytest.fixture(scope="module")
def ramp():
	return probe(LOOP.replace("STOP_ON_BLOWUP", "False"))


def test_the_rate_follows_the_ramp_whatever_the_script_sets(ramp):
	p = ramp.probe
	assert p.status == "ok", p.error
	pts = curve(p.extras)
	assert len(pts) >= 10 and "lr_range.not_set" not in p.extras
	for i, (lr, _) in enumerate(pts):
		assert lr == pytest.approx(lr_at(i, RAMP["steps"], RAMP["start"], RAMP["end"]), rel=1e-6)
	# the optimizer really held the ramp's rate at every step, not the script's 0.5 schedule
	assert p.extras["lr_range.checked"] == 1.0


def test_it_reads_a_suggestion_and_keeps_no_measurement_a_fit_would_read(ramp):
	p = ramp.probe
	assert p.extras["lr_range"] == 1.0
	assert "lr_range.suggested_lr" in p.extras or p.extras.get("lr_range.flat") == 1.0
	if "lr_range.suggested_lr" in p.extras:
		assert RAMP["start"] < p.extras["lr_range.suggested_lr"] < RAMP["end"]
	assert p.memory is None and p.step_time is None and p.stability is None and p.step_log == []


def test_a_script_that_stops_itself_on_the_blow_up_still_leaves_its_curve():
	r = probe(LOOP.replace("STOP_ON_BLOWUP", "True"))
	# its own raise ends it (at 2x its lowest loss, before the recorder's 4x-the-average rule): an
	# error, and the curve is kept
	assert r.probe.status == "error" and "blew up" in (r.probe.error or "")
	assert r.probe.extras["lr_range"] == 1.0 and r.probe.extras["lr_range.steps"] >= 10


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers not installed")
def test_the_hf_trainers_scheduler_is_overridden_too():
	r = probe((FIX / "tiny_hf.py").read_text())
	assert r.probe.status == "ok", r.probe.error
	ex = r.probe.extras
	assert len(curve(ex)) >= 10 and "lr_range.not_set" not in ex and ex["lr_range.checked"] == 1.0


# ---------------------------------------------------------------- A's review (Nov 2)

BASE = """
import torch
torch.manual_seed(0)
w = torch.randn(16, 1)
model = torch.nn.Sequential(torch.nn.Linear(16, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
OPT
for step in range(LOOPS):
	x = torch.randn(32, 16)
	loss = torch.nn.functional.mse_loss(model(x), x @ w)
	BODY
"""
PLAIN = "opt.zero_grad()\n\tloss.backward()\n\topt.step()"


def script(opt: str, body: str = PLAIN, loops: int = 2000) -> str:
	return BASE.replace("OPT", opt).replace("BODY", body).replace("LOOPS", str(loops))


def test_an_optimizer_that_ignores_the_rate_isnt_read_as_checked():
	# Adafactor's relative step in miniature: a fixed-size step whatever the group's rate says
	opt = """
class Fixed(torch.optim.Optimizer):
	def __init__(self, params):
		super().__init__(params, {"lr": 1e-3})
	@torch.no_grad()
	def step(self, closure=None):
		for g in self.param_groups:
			for p in g["params"]:
				if p.grad is not None:
					p.sub_(1e-3 * p.grad.sign())
opt = Fixed(model.parameters())
"""
	ex = probe(script(opt)).probe.extras
	assert ex["lr_range.not_set"] == 1.0 and "lr_range.checked" not in ex
	assert "lr_range.suggested_lr" not in ex and not curve(ex)


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers not installed")
def test_adafactor_with_its_relative_step_isnt_read_as_checked():
	opt = """
from transformers.optimization import Adafactor
opt = Adafactor(model.parameters(), lr=None, relative_step=True, scale_parameter=True, warmup_init=False)
"""
	ex = probe(script(opt)).probe.extras
	assert "lr_range.checked" not in ex and "lr_range.suggested_lr" not in ex
	assert ex["lr_range.not_set"] == 1.0  # its updates don't grow with the ramp


@pytest.mark.parametrize("kind", ["tensor", "fused"])
def test_a_tensor_rate_follows_the_ramp(kind):
	extra = ", fused=True" if kind == "fused" else ""
	r = probe(script(f"opt = torch.optim.Adam(model.parameters(), lr=torch.tensor(1e-3){extra})")).probe
	assert r.status == "ok", r.error
	assert r.extras["lr_range.checked"] == 1.0 and curve(r.extras)


def test_a_second_optimizer_on_other_weights_is_flagged():
	opt = """
opt = torch.optim.Adam(model[0].parameters(), lr=1e-3)
opt2 = torch.optim.SGD(model[2].parameters(), lr=1e-2)
"""
	body = "opt.zero_grad()\n\topt2.zero_grad()\n\tloss.backward()\n\topt.step()\n\topt2.step()"
	ex = probe(script(opt, body)).probe.extras
	assert ex["lr_range.other_optimizers"] == 1.0 and "lr_range.suggested_lr" not in ex


@pytest.mark.parametrize("start", [40, 0])
def test_a_scaler_that_skips_every_step_stops_the_probe_not_the_timeout(start):
	import time

	# fp16 overflow as it happens: a finite loss, infinite gradients, from step `start` on (0: from
	# the first step, before any step is recorded)
	opt = f"""opt = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.amp.GradScaler("cpu")
STEP = [0]
for p in model.parameters():
	p.register_hook(lambda g: g * float("inf") if STEP[0] >= {start} else g)
"""
	body = """STEP[0] = step
	opt.zero_grad()
	scaler.scale(loss).backward()
	scaler.step(opt)
	scaler.update()"""
	t = time.monotonic()
	r = probe(script(opt, body, loops=10**9)).probe
	assert time.monotonic() - t < 120, "ran to the timeout"
	assert r.extras["lr_range.stalled"] == 1.0
	if start:  # 40 steps span enough rates to check the ramp, so its divergence stands
		assert r.status == "ok", r.error
		assert r.extras["lr_range.checked"] == 1.0
		last = lr_at(start - 1, RAMP["steps"], RAMP["start"], RAMP["end"])
		assert r.extras["lr_range.diverged_lr"] == pytest.approx(last)


def test_a_rebuilt_optimizer_isnt_read_as_a_divergence():
	body = """if step == 30:
		opt = torch.optim.Adam(model.parameters(), lr=1e-3)  # the script starts a new optimizer
	opt.zero_grad()
	loss.backward()
	opt.step()"""
	r = probe(script("opt = torch.optim.Adam(model.parameters(), lr=1e-3)", body)).probe
	assert r.status == "ok", r.error
	ex = r.extras
	assert ex["lr_range.not_set"] == 1.0 and "lr_range.stalled" not in ex and "lr_range.diverged_lr" not in ex


def test_weights_that_start_at_zero_dont_hide_a_ramp_that_took_effect():
	# LoRA's B and zero-init output projections: their update-to-weight ratio goes flat
	opt = """
torch.nn.init.zeros_(model[2].weight)
lora_b = torch.nn.Parameter(torch.zeros(64, 64))
opt = torch.optim.AdamW([*model.parameters(), lora_b], lr=1e-3)
"""
	body = "opt.zero_grad()\n\t(loss + (lora_b.sum() * 1e-3) ** 2).backward()\n\topt.step()"
	ex = probe(script(opt, body)).probe.extras
	assert ex["lr_range.checked"] == 1.0 and curve(ex)


def test_group_ratios_survive_a_warm_up_from_zero():
	opt = """
opt = torch.optim.Adam([
	{"params": model[0].parameters(), "lr": 1e-3},
	{"params": model[2].parameters(), "lr": 2.5e-4},
])
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, s / 10))  # a warm-up from 0
"""
	body = """opt.zero_grad()
	loss.backward()
	opt.step()
	print("RATIO", opt.param_groups[1]["lr"] / opt.param_groups[0]["lr"], flush=True)
	sched.step()"""
	r = probe(script(opt, body))
	assert r.probe.status == "ok", r.probe.error
	ratios = [float(line.split()[1]) for line in r.log_tail.splitlines() if line.startswith("RATIO ")]
	assert len(ratios) >= 5 and all(x == pytest.approx(0.25) for x in ratios)


def test_a_scaler_with_long_accumulation_isnt_read_as_a_stall():
	# fp16 on a T4 with HF-style accumulation: 64 backwards a step, nothing overflowing (the review's case)
	opt = 'opt = torch.optim.Adam(model.parameters(), lr=1e-3)\nscaler = torch.amp.GradScaler("cpu")'
	body = """for _ in range(64):
		scaler.scale(torch.nn.functional.mse_loss(model(x), x @ w) / 64).backward()
	scaler.step(opt)
	scaler.update()
	opt.zero_grad()"""
	r = probe(script(opt, body)).probe
	assert r.status == "ok", r.error
	assert "lr_range.stalled" not in r.extras and r.extras["lr_range.steps"] >= 10


def test_a_bigger_rebuilt_optimizer_stops_the_probe_and_keeps_its_curve():
	# an unfreeze stage: the new optimizer holds a weight the first didn't (the review's case)
	opt = "opt = torch.optim.Adam(model[0].parameters(), lr=1e-3)"
	body = """if step == 30:
		opt = torch.optim.Adam(model.parameters(), lr=1e-3)
	opt.zero_grad()
	loss.backward()
	opt.step()"""
	r = probe(script(opt, body)).probe
	assert r.status == "ok", r.error
	assert r.extras["lr_range.other_optimizers"] == 1.0 and r.extras["lr_range.steps"] >= 10

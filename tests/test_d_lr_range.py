"""The learning-rate range test's reading of its curve (D, Nov 2)."""

import math

import pytest

from preflight.probes.lr_range import LR_END, LR_START, STEPS, analyse, curve, lr_at


def synthetic(best_lr=1e-3, diverge_at=0.1, noise=0.0):
	"""A loss that falls fastest a decade before best_lr, bottoms there and blows up past diverge_at."""
	pts = []
	for i in range(STEPS):
		lr = lr_at(i)
		x = math.log10(lr)
		loss = 2.0 - 1.5 / (1 + math.exp(-2.5 * (x - math.log10(best_lr) + 1)))
		if lr > diverge_at:
			loss *= 1 + 20 * (x - math.log10(diverge_at))
		loss += noise * math.sin(i * 1.7)
		pts.append((lr, loss))
	return pts


def test_the_schedule_rises_exponentially_from_start_to_end():
	assert lr_at(0) == LR_START and math.isclose(lr_at(STEPS - 1), LR_END)
	ratios = {round(lr_at(i + 1) / lr_at(i), 9) for i in range(STEPS - 1)}
	assert len(ratios) == 1


def test_it_suggests_the_steepest_fall_and_finds_the_divergence():
	ex = analyse(synthetic())
	assert 1e-5 < ex["lr_range.suggested_lr"] < 1e-3  # below the minimum, on the slope
	assert ex["lr_range.suggested_lr"] < ex["lr_range.min_loss_lr"] <= 0.2
	assert 0.1 < ex["lr_range.diverged_lr"] <= 1.0
	assert "lr_range.flat" not in ex and 50 < ex["lr_range.steps"] < STEPS  # read up to the divergence
	pts = curve(ex)
	assert len(pts) == ex["lr_range.steps"] and pts[0][0] == LR_START
	assert all(lr < ex["lr_range.diverged_lr"] for lr, _ in pts)


def test_a_huge_blow_up_doesnt_move_the_reading():
	# past the divergence, its size doesn't matter: the curve is read only up to it
	mild = analyse([(lr, loss if lr <= 0.1 else loss * 1e3) for lr, loss in synthetic()])
	huge = analyse([(lr, loss if lr <= 0.1 else loss * 1e9) for lr, loss in synthetic()])
	assert huge == mild and "lr_range.suggested_lr" in huge
	assert huge["lr_range.suggested_lr"] == mild["lr_range.suggested_lr"]
	assert huge["lr_range.min_loss_lr"] == mild["lr_range.min_loss_lr"]


def test_a_loss_not_read_is_skipped_but_a_nan_flag_is_divergence():
	pts = synthetic()
	skipped = [(lr, None if i == 30 else loss) for i, (lr, loss) in enumerate(pts)]
	ex = analyse(skipped)
	assert ex["lr_range.diverged_lr"] > 0.1 and "lr_range.suggested_lr" in ex
	flagged = [(lr, None, True) if i == 30 else (lr, loss, False) for i, (lr, loss) in enumerate(pts)]
	assert analyse(flagged)["lr_range.diverged_lr"] == pts[30][0]


def test_a_non_finite_loss_ends_the_curve_as_divergence():
	pts = synthetic()
	pts[80] = (pts[80][0], float("nan"))
	ex = analyse(pts)
	assert ex["lr_range.diverged_lr"] == pts[80][0] and ex["lr_range.steps"] == 80


def test_a_flat_curve_gives_no_suggestion():
	ex = analyse([(lr_at(i), 1.0 + 0.001 * math.sin(i)) for i in range(STEPS)])
	assert ex["lr_range.flat"] == 1.0 and "lr_range.suggested_lr" not in ex


def test_too_few_points_gives_no_suggestion():
	ex = analyse([(lr_at(i), 2.0 - i * 0.1) for i in range(4)])
	assert ex["lr_range.flat"] == 1.0 and "lr_range.suggested_lr" not in ex


def test_noise_doesnt_move_the_suggestion_far():
	clean = analyse(synthetic())["lr_range.suggested_lr"]
	noisy = analyse(synthetic(noise=0.02))["lr_range.suggested_lr"]
	assert abs(math.log10(noisy) - math.log10(clean)) < 1.0


def test_extras_are_floats_only():
	ex = analyse(synthetic())
	assert all(isinstance(v, float) for v in ex.values())


# ---------------------------------------------------------------- did the rate take effect (A's review, Nov 2)


def _probe():
	from preflight.schema import Probe

	return Probe(id="p1-lr", knobs={}, status="ok", steps=STEPS, cost={"gpu_seconds": 1.0, "usd": 0.0})


RAMP = {"start": LR_START, "end": LR_END, "steps": STEPS}


def _summary(updates, **kw):
	pts = synthetic()
	return {
		"series": {"loss": [v for _, v in pts], "lr": [x for x, _ in pts]},
		"lr_range_update": updates,
		**kw,
	}


def test_updates_that_grow_with_the_rate_follow_it_and_flat_ones_dont():
	from preflight.probes.lr_range import follows_rate

	lrs = [lr_at(i) for i in range(STEPS)]
	assert follows_rate(lrs, [0.3 * x for x in lrs]) is True  # SGD, Adam: the update scales with the rate
	assert follows_rate(lrs, [1e-3] * STEPS) is False  # Adafactor's relative step ignores it
	assert follows_rate(lrs, [1e-2 / (i + 1) ** 0.5 for i in range(STEPS)]) is False  # its own decay
	assert follows_rate(lrs[:5], [x for x in lrs[:5]]) is None  # too few to tell
	assert follows_rate([1e-3] * 50, [1e-4] * 50) is None  # one rate: no span to read
	assert follows_rate(lrs, [None, True, float("nan"), 0.0] * 25) is None  # nothing measured


def test_a_rate_stands_only_when_the_updates_followed_the_ramp():
	from preflight.probes.local import with_lr_range

	lrs = [lr_at(i) for i in range(STEPS)]
	ok = with_lr_range(_probe(), _summary([0.5 * x for x in lrs]), RAMP).extras
	assert ok["lr_range.checked"] == 1.0 and "lr_range.suggested_lr" in ok and curve(ok)
	# the held rate always matches the ramp (the recorder wrote it): it isn't evidence any more
	ignored = with_lr_range(_probe(), _summary([1e-3] * STEPS), RAMP).extras
	assert ignored["lr_range.not_set"] == 1.0 and "lr_range.checked" not in ignored
	assert "lr_range.suggested_lr" not in ignored and not curve(ignored) and "lr_range.loss.000" in ignored
	assert ignored["lr_range.steps"] > 0  # the losses stay, without their rates
	none = with_lr_range(_probe(), _summary([None] * STEPS), RAMP).extras
	assert none["lr_range.unverified"] == 1.0 and "lr_range.suggested_lr" not in none


def test_a_second_optimizer_on_other_weights_drops_the_rates():
	from preflight.probes.local import with_lr_range

	lrs = [lr_at(i) for i in range(STEPS)]
	ex = with_lr_range(_probe(), _summary([0.5 * x for x in lrs], lr_range_other_optimizers=1), RAMP).extras
	assert ex["lr_range.other_optimizers"] == 1.0 and "lr_range.suggested_lr" not in ex
	assert "lr_range.checked" not in ex


def test_a_stalled_scaler_marks_where_it_diverged():
	from preflight.probes.local import with_lr_range

	n = 40
	lrs = [lr_at(i) for i in range(n)]
	s = _summary([0.5 * x for x in lrs], lr_range_stalled=n)
	s["series"] = {k: v[:n] for k, v in s["series"].items()}
	ex = with_lr_range(_probe(), s, RAMP).extras
	# the step that overflowed ran at the last rate applied, the one before the stall
	assert ex["lr_range.stalled"] == 1.0 and ex["lr_range.diverged_lr"] == lr_at(n - 1)
	assert ex["lr_range.checked"] == 1.0
	# stalled before any step: no rate was applied, so none is the divergence
	s0 = _summary([], lr_range_stalled=0)
	s0["series"] = {"loss": [], "lr": []}
	ex0 = with_lr_range(_probe(), s0, RAMP).extras
	assert ex0["lr_range.stalled"] == 1.0 and "lr_range.diverged_lr" not in ex0


def test_a_replaced_optimizer_and_every_reason_that_applies_are_kept():
	from preflight.probes.local import with_lr_range

	lrs = [lr_at(i) for i in range(STEPS)]
	ex = with_lr_range(_probe(), _summary([0.5 * x for x in lrs], lr_range_replaced=1), RAMP).extras
	assert ex["lr_range.not_set"] == 1.0 and "lr_range.checked" not in ex
	both = with_lr_range(_probe(), _summary([1e-3] * STEPS, lr_range_other_optimizers=2), RAMP).extras
	assert both["lr_range.not_set"] == 1.0 and both["lr_range.other_optimizers"] == 2.0


def test_the_losses_are_read_back_without_their_rates():
	from preflight.probes.local import with_lr_range
	from preflight.probes.lr_range import losses

	held = {"start": 1e-3, "end": 1e-3, "steps": 50}  # the GPU check's confirmation runs: one rate
	s = {"series": {"loss": [2.0 - i / 100 for i in range(50)]}, "lr_range_update": [1e-4] * 50}
	ex = with_lr_range(_probe(), s, held).extras
	assert ex["lr_range.unverified"] == 1.0 and not curve(ex)
	assert losses(ex) == pytest.approx([2.0 - i / 100 for i in range(50)])


def test_a_hostile_summary_reads_as_unverified_not_a_crash():
	from preflight.probes.local import with_lr_range

	for bad in ({}, {"series": "x"}, {"lr_range_update": "x", "series": {"loss": [True, "a"]}},
		{"lr_range_stalled": True, "lr_range_other_optimizers": "2"},
		{"lr_range_other_optimizers": float("inf"), "lr_range_stalled": float("nan")},
		{"lr_range_other_optimizers": 10**400, "lr_range_stalled": 10**400}):  # fmt: skip
		ex = with_lr_range(_probe(), bad, RAMP).extras
		assert ex["lr_range"] == 1.0 and "lr_range.checked" not in ex
		assert all(type(v) is float for v in ex.values())

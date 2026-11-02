"""The learning-rate range test's reading of its curve (D, Nov 2)."""

import math

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

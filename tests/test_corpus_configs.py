def test_the_realistic_divergence_sweep_stays_at_usual_rates_with_the_usual_triggers():
	"""Chiya's review (Nov 2): every Nov 1 divergence came at lr 0.07 or more; this sweep keeps lr at 3e-2 or under and
	turns on no clipping, no warm-up and fp16 without a scaler."""
	from preflight.corpus.configs import configs_for, divergence_configs

	cfgs = configs_for("divergence_real", 200, seed=0)
	assert len({c.key for c in cfgs}) == 200 and all(c.kind == "divergence_real" for c in cfgs)
	ps = [c.params for c in cfgs]
	assert all(
		1e-4 <= p["lr"] <= 3e-2 and p["warmup"] == 0 and p["grad_clip"] is None and p["scaler"] is False
		for p in ps
	)
	assert 0.35 < sum(p["precision"] == "fp16" for p in ps) / len(ps) < 0.65
	assert not {c.key for c in cfgs} & {c.key for c in divergence_configs(200, seed=0)}

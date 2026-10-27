"""Monte Carlo for B's engine settings, fixed seed: 20k trials for memory, 4k for divergence and cross-GPU, 3k budget sweeps.

S1: memory estimator on A's three probes (current, half batch, half seq).
S2: the 90% band rule (S2a), the fits / tight / fails cut-offs (S2b), the band with one or two probes (S2c),
    and how the band grows below the smallest probe (S2d).
S3: the rule that labels a short run diverged.
S4: carrying step time from the probed GPU to another GPU; S4b: step time below the smallest probe;
    S4c: K1's band when the per-kernel drift is larger than assumed.
S5: the corpus sweep's cost overhead factor and per-config time estimate.
S6: B's agent-total code (engine/agent_cost.py) re-checked on A's D3 generator; S6b: 30 tasks plus a
    20-task top-up when the cost range straddles the budget.

Assumptions follow A's docs/decisions/probe-plan.md where they overlap (allocator
overhead 2-12% plus 2 MiB rounding, eager attention's bs*s^2 term, queries within
2x of the largest probed bs*s), widened where B's models add their own error
(analytical coefficients off by 30%, misread attention kernels).

	uv run python scripts/mc_engine_settings.py
"""

import numpy as np

rng = np.random.default_rng(20261027)
N = 20_000
GB = 1e9


def pct(x) -> str:
	return f"{100 * np.mean(x):5.1f}%"


# ------------------------------------------------------------------ S1 and S2 memory
def memory_world():
	"""One simulated job per trial: true memory law, analytical guess, three probes, one query."""
	w = {}
	d = rng.choice([768, 1536, 2048, 4096], N)
	L = rng.choice([12, 24, 28, 32], N)
	V = rng.choice([32_000, 50_257, 151_936], N)
	H = d // 128
	w["static"] = rng.uniform(0.5, 16, N) * GB
	w["a"] = d * L * rng.uniform(12, 40, N)  # bytes per token, linear activation terms
	w["l"] = V * rng.uniform(6, 12, N)  # bytes per token, logits incl. fp32 upcast copies
	eager = rng.random(N) < 0.3
	w["g"] = np.where(eager, H * L * 2 * rng.uniform(1, 3, N), 0.0)  # bytes per bs*s^2
	w["frag"] = rng.uniform(0.02, 0.12, N)
	w["ctx"] = rng.uniform(0.5, 1.0, N) * GB
	# analytical guess, as the engine would make it from the script map and knobs
	w["static_hat"] = w["static"] * rng.lognormal(0, 0.05, N)
	w["a_hat"] = w["a"] * rng.lognormal(0, 0.30, N)
	w["l_hat"] = w["l"] * rng.lognormal(0, 0.15, N)
	misread = rng.random(N) < 0.10  # attention kernel guessed wrong
	g_nom = H * L * 2 * 2.0
	w["g_hat"] = np.where(misread, np.where(eager, 0.0, g_nom), np.where(eager, g_nom, 0.0))
	# current knobs placed so the current run lands at 30-110% of the GPU
	w["bs0"] = rng.choice([1, 2, 4, 8, 16], N).astype(float)
	w["s0"] = rng.choice([256, 512, 1024, 2048, 4096], N).astype(float)
	cur_true = alloc(w, w["bs0"], w["s0"]) * (1 + w["frag"]) + w["ctx"]
	w["cap"] = cur_true / rng.uniform(0.3, 1.1, N)
	# queries within 2x of the current bs*s, as A's plan assumes
	fb = rng.choice([0.25, 0.5, 1, 2], N)
	fs = rng.choice([0.25, 0.5, 1, 2], N)
	ok = fb * fs <= 2
	fb, fs = np.where(ok, fb, 1.0), np.where(ok, fs, 2.0)
	w["qb"], w["qs"] = w["bs0"] * fb, w["s0"] * fs
	return w


def _v(w, key, like):
	v = w[key]
	return v[:, None] if np.ndim(like) == 2 else v


def alloc(w, bs, s):
	return _v(w, "static", bs) + bs * s * (_v(w, "a", bs) + _v(w, "l", bs)) + bs * s * s * _v(w, "g", bs)


def reserved(w, bs, s, jitter=True):
	fr = np.clip(_v(w, "frag", bs) + (rng.normal(0, 0.02, np.shape(bs)) if jitter else 0), 0, None)
	r = alloc(w, bs, s) * (1 + fr)
	return np.ceil(r / 2**21) * 2**21


def analytical(w, bs, s):
	return (
		_v(w, "static_hat", bs)
		+ bs * s * (_v(w, "a_hat", bs) + _v(w, "l_hat", bs))
		+ bs * s * s * _v(w, "g_hat", bs)
	)


def probes(w):
	"""Three probes; batch-1 jobs vary sequence twice, as probes/plan.py does. OOM probes are NaN."""
	b, s = w["bs0"], w["s0"]
	one = b < 2
	pts = [(b, s), (np.where(one, b, b / 2), np.where(one, s / 2, s)), (b, np.where(one, s / 4, s / 2))]
	bs = np.stack([p[0] for p in pts], 1)
	ss = np.stack([p[1] for p in pts], 1)
	m = reserved(w, bs, ss)
	m = np.where(m + w["ctx"][:, None] > w["cap"][:, None], np.nan, m)
	return bs, ss, m


def fit(cols, y):
	"""Per-trial least squares over the probes that didn't OOM; drops trailing columns when short of points."""
	X = np.stack(cols, 2)  # (N, P, K)
	n, _, k = X.shape
	good = ~np.isnan(y)
	scale = np.nanmax(np.abs(np.where(good[:, :, None], X, np.nan)), axis=1)
	scale = np.where((scale > 0) & np.isfinite(scale), scale, 1.0)
	Xn = np.where(good[:, :, None], X / scale[:, None, :], 0.0)
	yn = np.where(good, y, 0.0)
	coef = np.zeros((n, k))
	cnt = good.sum(1)
	for m in range(1, X.shape[1] + 1):
		sel = cnt == m
		if not sel.any():
			continue
		kk = min(k, m)
		sol = np.linalg.pinv(Xn[sel][:, :, :kk]) @ yn[sel][:, :, None]
		coef[sel, :kk] = sol[:, :, 0] / scale[sel][:, :kk]
	return coef


def s1_estimators(w, pr):
	b, s, m = pr
	A = analytical(w, b, s)
	A_dyn = A - w["static_hat"][:, None]
	qa = analytical(w, w["qb"], w["qs"])
	qa_dyn = qa - w["static_hat"]
	x = b * s
	qx = w["qb"] * w["qs"]
	ones = np.ones_like(x)
	est = {}
	c = fit([ones, x, x * s], m)
	est["E1 empirical S + b*bs*s + g*bs*s^2 (A's D2)"] = c[:, 0] + c[:, 1] * qx + c[:, 2] * qx * w["qs"]
	est["E2 analytical + constant offset"] = qa + np.nanmean(m - A, 1)
	c = fit([ones, x], m - A)
	est["E3 analytical + residual a + b*bs*s"] = qa + c[:, 0] + c[:, 1] * qx
	c = fit([ones, A_dyn], m)
	est["E4 analytical scaled: S + alpha*A_dyn"] = c[:, 0] + c[:, 1] * qa_dyn
	est["E5 mean of E1 and E4"] = 0.5 * (
		est["E1 empirical S + b*bs*s + g*bs*s^2 (A's D2)"] + est["E4 analytical scaled: S + alpha*A_dyn"]
	)
	n_ok = (~np.isnan(m)).sum(1)
	return est, n_ok


def s1():
	print("S1  memory estimator   within 10%   within 10% near cap   missed OOM   false alarm")
	w = memory_world()
	pr = probes(w)
	truth = reserved(w, w["qb"], w["qs"]) + w["ctx"]
	est, n_ok = s1_estimators(w, pr)
	usable = n_ok >= 1
	near = usable & (np.abs(truth / w["cap"] - 1) < 0.2)
	out = {}
	for name, alloc_pred in est.items():
		p = alloc_pred + 0.75 * GB
		err = np.abs(p / truth - 1)
		oom = truth > w["cap"]
		missed = np.mean((oom & (p <= w["cap"]))[usable])
		false = np.mean((~oom & (p > w["cap"]))[usable])
		print(
			f"  {name:<46} {pct(err[usable] <= 0.10)}       {pct(err[near] <= 0.10)}           {missed * 100:5.2f}%      {false * 100:5.2f}%"
		)
		out[name] = (p, err)
	print(f"  (trials with every probe OOM, so the engine abstains: {100 * np.mean(~usable):.1f}%)")
	return w, truth, out, usable, pr


def loo_spread(w, pr):
	"""Leave-one-probe-out |log error| of E3 at the held-out probe, max over probes (0 with < 3 probes)."""
	b, s, m = pr
	A = analytical(w, b, s)
	x = b * s
	out = np.zeros(N)
	for j in range(3):
		keep = [i for i in range(3) if i != j]
		c = fit([np.ones_like(x[:, keep]), x[:, keep]], (m - A)[:, keep])
		pred = A[:, j] + c[:, 0] + c[:, 1] * x[:, j]
		e = np.abs(np.log(m[:, j] / pred))
		full = (~np.isnan(m)).all(1)
		out = np.where(full & np.isfinite(e), np.maximum(out, e), out)
	return out


def s2(w, truth, est, usable, pr):
	print("\nS2a 90% band rule (estimator E3)      coverage   coverage near cap   median width (hi/lo)")
	p = est["E3 analytical + residual a + b*bs*s"][0]
	p4 = est["E4 analytical scaled: S + alpha*A_dyn"][0]
	u = usable & np.isfinite(p) & (p > 0) & np.isfinite(p4) & (p4 > 0)
	logerr = np.log(truth / np.where(u, p, 1.0))
	near = u & (np.abs(truth / w["cap"] - 1) < 0.2)
	z = 1.6448536269514722
	q90 = float(np.quantile(np.abs(logerr[u]), 0.9))
	disagree = np.abs(np.log(np.where(u, p4 / p, 1.0)))
	loo = loo_spread(w, pr)
	rules = {
		"R1 fixed sigma 0.03": np.full(N, 0.03 * z),
		"R2 fixed sigma 0.05": np.full(N, 0.05 * z),
		f"R3 simulation-calibrated constant ({q90:.3f})": np.full(N, q90),
		"R4 constant + E3-vs-E4 disagreement": q90 + disagree,
		"R5 constant floor, widened by leave-one-out spread": np.maximum(q90, 1.5 * loo),
	}
	for name, h in rules.items():
		cov = np.abs(logerr) <= h
		print(
			f"  {name:<52} {pct(cov[u])}      {pct(cov[near])}            {np.median(np.exp(2 * h[u])):.3f}"
		)
	return p, u, q90, disagree, loo


def s2b(w, truth, p, u, h):
	print(
		"\nS2b status cut-offs (fits < a, fails >= b)   P(fits | OOM)   P(fails | fits on GPU)   share tight   share wrong-side"
	)
	from scipy.stats import norm

	sigma = h / 1.6448536269514722
	p_oom = norm.sf((np.log(w["cap"]) - np.log(np.where(u, p, 1.0))) / sigma)
	oom = truth > w["cap"]
	for a, b in ((0.01, 0.5), (0.02, 0.3), (0.05, 0.5), (0.1, 0.5), (0.05, 0.8), (0.01, 0.8)):
		fits, fails = p_oom < a, p_oom >= b
		tight = ~fits & ~fails
		wrong = (oom & fits) | (~oom & fails)
		print(
			f"  a={a:<5} b={b:<5}                               {100 * np.mean(fits[u & oom]):5.2f}%          "
			f"{100 * np.mean(fails[u & ~oom]):5.2f}%                  {pct(tight[u])}      {pct(wrong[u])}"
		)


def s2c(w, pr):
	"""Band when only one or two probes are usable (the current probe OOMed, or a probe failed)."""
	print("\nS2c band with 1-2 probes (E3)     subset            coverage   near cap   median width")
	b, s, m = pr
	truth = reserved(w, w["qb"], w["qs"]) + w["ctx"]
	subsets = {
		"current only": [0],
		"current + half batch": [0, 1],
		"current + half seq": [0, 2],
		"half batch + half seq": [1, 2],
	}
	cands = [
		("0.107 (R3 constant)", 0.107),
		("0.134 (1.25x)", 0.134),
		("0.160 (1.5x)", 0.160),
		("0.214 (2x)", 0.214),
		("0.241 (prior)", 0.241),
	]
	results = {}
	for sname, keep in subsets.items():
		bk, sk, mk = b[:, keep], s[:, keep], m[:, keep]
		A = analytical(w, bk, sk)
		x = bk * sk
		c = fit([np.ones_like(x), x], mk - A)
		qx = w["qb"] * w["qs"]
		p = analytical(w, w["qb"], w["qs"]) + c[:, 0] + c[:, 1] * qx + 0.75 * GB
		ok = ~np.isnan(mk).any(1)
		xmin, xmax = (
			np.nanmin(np.where(np.isnan(mk), np.nan, x), 1),
			np.nanmax(np.where(np.isnan(mk), np.nan, x), 1),
		)
		inr = ok & (qx >= xmin / 2) & (qx <= 2 * xmax) & (p > 0)
		e = np.abs(np.log(truth / np.where(inr, p, 1.0)))
		near = inr & (np.abs(truth / w["cap"] - 1) < 0.2)
		results[sname] = (e, inr, near)
	for cname, h in cands:
		row = []
		for sname, (e, inr, near) in results.items():
			row.append(f"{sname}: {pct((e <= h)[inr])} / {pct((e <= h)[near])}")
		print(f"  {cname:<20} " + "   ".join(row))


def s2d(w, pr):
	"""Below the smallest probe: an extrapolated estimate whose band grows per halving (Chiya, Oct 27).

	Queries sit 2^-k below the smallest probed work units, batch lowered first (to 1), then sequence.
	The band starts from S2a's R5 (max(0.107, 1.5 x leave-one-out)) and the estimate abstains once
	it passes +-50%, the engine's existing rule, so "far outside" falls out of the growth rule.
	"""
	b, s, m = pr
	x = b * s
	ok = ~np.isnan(m).any(1)
	A = analytical(w, b, s)
	c = fit([np.ones_like(x), x], m - A)
	# leave-one-probe-out error, as Calibration.fit computes it
	loo = np.zeros(N)
	for i in range(3):
		keep = [j for j in range(3) if j != i]
		ci = fit([np.ones_like(x[:, keep]), x[:, keep]], (m - A)[:, keep])
		pred = A[:, i] + ci[:, 0] + ci[:, 1] * x[:, i]
		loo = np.maximum(
			loo, np.abs(np.log(np.where(pred > 0, m[:, i] / np.where(pred > 0, pred, 1), np.inf)))
		)
	h0 = np.maximum(0.107, 1.5 * loo)
	xmin = np.nanmin(np.where(np.isnan(m), np.nan, x), 1)
	bmin = b[np.arange(N), np.nanargmin(np.where(np.isnan(m), np.inf, x), 1)]
	cands = {
		"G1 constant x1.5": lambda h, k: h * 1.5,
		"G2 constant x2": lambda h, k: h * 2.0,
		"G3 x(1 + 0.25 per halving)": lambda h, k: h * (1 + 0.25 * k),
		"G4 x(1 + 0.5 per halving)": lambda h, k: h * (1 + 0.5 * k),
		"G5 + 0.05 per halving": lambda h, k: h + 0.05 * k,
	}
	ks = [0.5, 1, 2, 4, 6, 8, 10]
	print(
		"\nS2d band below the smallest probe   coverage of estimates given / share abstained, by halvings below"
	)
	print("  " + " " * 30 + "".join(f"{k:>5g} halvings   " for k in ks))
	for name, grow in cands.items():
		row = []
		for k in ks:
			qx = xmin * 2.0**-k
			qb = np.maximum(1.0, np.floor(bmin * 2.0**-k))
			qs = np.maximum(1.0, qx / qb)
			truth = reserved(w, qb, qs) + w["ctx"]
			p = analytical(w, qb, qs) + c[:, 0] + c[:, 1] * qb * qs + 0.75 * GB
			h = grow(h0, k)
			given = ok & (p > 0) & (h <= np.log(1.5))
			e = np.abs(np.log(truth / np.where(p > 0, p, 1.0)))
			cov = (e <= h)[given].mean() if given.any() else np.nan
			row.append(f"{100 * cov:5.1f}% / {100 * (1 - given[ok].mean()):4.1f}%")
		print(f"  {name:<30}" + "  ".join(row))


# ------------------------------------------------------------------ S3 divergence labels
def curves(n_steps=600):
	"""Simulated loss curves with known outcomes."""
	t = np.arange(1, n_steps + 1)
	kind = rng.choice(
		["stable", "spike_recover", "nan", "blowup", "drift_up", "plateau", "collapse"],
		N,
		p=[0.32, 0.14, 0.11, 0.16, 0.09, 0.09, 0.09],
	)
	L0 = rng.uniform(4, 11, N)
	floor = rng.uniform(1.0, 3.0, N)
	rate = rng.uniform(0.2, 0.6, N)
	base = floor[:, None] + (L0 - floor)[:, None] * t[None, :] ** (-rate[:, None])
	noise = rng.normal(0, 1, (N, n_steps)) * rng.uniform(0.01, 0.06, N)[:, None] * base
	loss = base + noise
	t0 = rng.integers(60, n_steps - 60, N)
	diverged = np.isin(kind, ["nan", "blowup", "drift_up", "collapse"])
	for i in np.where(kind == "spike_recover")[0]:
		w = rng.integers(5, 40)
		loss[i, t0[i] : t0[i] + w] *= rng.uniform(1.6, 4.0)
	for i in np.where(kind == "nan")[0]:
		loss[i, t0[i] :] = np.nan
	for i in np.where(kind == "blowup")[0]:
		k = np.arange(n_steps - t0[i])
		loss[i, t0[i] :] = loss[i, t0[i]] * np.exp(rng.uniform(0.02, 0.15) * k)
	for i in np.where(kind == "drift_up")[0]:
		k = np.arange(n_steps - t0[i])
		loss[i, t0[i] :] += rng.uniform(0.004, 0.02) * k
	for i in np.where(kind == "plateau")[0]:
		loss[i] = L0[i] * (1 + rng.normal(0, 0.02, n_steps))
	for i in np.where(kind == "collapse")[0]:
		# a spike, then stuck at the trivial baseline (uniform prediction ~ the initial loss);
		# half of them blow up in the first steps, before the run has learned anything
		if rng.random() < 0.5:
			t0[i] = rng.integers(1, 30)
		w = rng.integers(5, 40)
		loss[i, t0[i] : t0[i] + w] = L0[i] * rng.uniform(3, 40)
		loss[i, t0[i] + w :] = (
			L0[i] * (1 + rng.normal(0, 0.005, n_steps - t0[i] - w)) * rng.uniform(0.95, 1.0)
		)
	return loss, diverged, t0, kind


def rule_mine(x, persist=1, factor=0.5, floor=0.2, alpha=0.1):
	if not np.all(np.isfinite(x)):
		return True, int(np.argmax(~np.isfinite(x)))
	ema, best, run = x[0], x[0], 0
	for i, v in enumerate(x):
		ema = (1 - alpha) * ema + alpha * v
		best = min(best, ema)
		if i >= 10 and ema > best + max(factor * best, floor):
			run += 1
			if run >= persist:
				return True, i
		else:
			run = 0
	if len(x) >= 20 and np.mean(x[-10:]) > x[0]:
		return True, len(x) - 10
	return False, None


def s3():
	print(
		"\nS3  divergence label rule      accuracy   alarms on recovered spikes   alarms on plateaus   missed drift   missed collapse   median lag"
	)
	loss, div, t0, kind = curves()
	rules = {
		"D1 non-finite only": lambda x: (
			not np.all(np.isfinite(x)),
			int(np.argmax(~np.isfinite(x))) if not np.all(np.isfinite(x)) else None,
		),
		"D2 raw loss > 1.5 x running min": lambda x: _raw(x),
		"D3 EMA above best + max(0.5 best, 0.2) (B's first)": lambda x: rule_mine(x),
		"D4 D3 held for 20 steps": lambda x: rule_mine(x, persist=20),
		"D5 D3 held for 50 steps": lambda x: rule_mine(x, persist=50),
		"D6 rolling median(25) > 1.25 x its running min + 0.1": lambda x: _rolling(x, 25, 1),
		"D7 D6 held for 30 steps": lambda x: _rolling(x, 25, 30),
		"D8 hindsight: end median vs best, onset from D6": lambda x: _hindsight(x),
		"D9 D8, or a spike past 2x the start then no learning below 90% of it": lambda x: _collapse(x),
	}
	idx = rng.choice(N, 4000, replace=False)
	for name, f in rules.items():
		res = [f(loss[i]) for i in idx]
		pred = np.array([r[0] for r in res])
		acc = np.mean(pred == div[idx])
		spike = kind[idx] == "spike_recover"
		drift = kind[idx] == "drift_up"
		plat = kind[idx] == "plateau"
		coll = kind[idx] == "collapse"
		lags = [
			r[1] - t0[i]
			for r, i, d in zip(res, idx, div[idx], strict=True)
			if r[0] and d and r[1] is not None
		]
		print(
			f"  {name:<52} {pct(acc)}      {pct(pred[spike])}                   {pct(pred[plat])}            {pct(~pred[drift])}        {pct(~pred[coll])}         {np.median(lags) if lags else float('nan'):.0f}"
		)


def _rmed(x, win):
	pad = np.concatenate([np.full(win - 1, x[0]), x])
	idx = np.arange(win)[None, :] + np.arange(len(x))[:, None]
	return np.median(pad[idx], 1)


def _rolling(x, win, persist):
	if not np.all(np.isfinite(x)):
		return True, int(np.argmax(~np.isfinite(x)))
	med = _rmed(x, win)
	best = np.minimum.accumulate(med)
	above = (np.arange(len(x)) >= win) & (med > 1.25 * best + 0.1)
	run = 0
	for i, a in enumerate(above):
		run = run + 1 if a else 0
		if run >= persist:
			return True, i - persist + 1
	return False, None


def _hindsight(x):
	"""Offline label: non-finite, or the run ends well above its best smoothed loss."""
	if not np.all(np.isfinite(x)):
		return True, int(np.argmax(~np.isfinite(x)))
	med = _rmed(x, 25)
	best = np.minimum.accumulate(med)
	thr = 1.25 * best + 0.1
	end = np.median(x[-50:])
	if end <= 1.25 * best[-1] + 0.1:
		return False, None
	below = np.where(med <= thr)[0]
	onset = int(below[-1] + 1) if below.size else 0
	return True, onset


def _collapse(x):
	d, step = _hindsight(x)
	if d:
		return d, step
	tail = np.median(x[-50:])
	if np.max(x) > 2 * x[0] and tail >= 0.9 * x[0]:
		return True, int(np.argmax(x))
	return False, None


def _raw(x):
	if not np.all(np.isfinite(x)):
		return True, int(np.argmax(~np.isfinite(x)))
	best = np.minimum.accumulate(x)
	hit = np.where((np.arange(len(x)) >= 10) & (x > 1.5 * best))[0]
	return (True, int(hit[0])) if hit.size else (False, None)


# ------------------------------------------------------------------ S4 cross-GPU step time
def s4():
	print("\nS4  cross-GPU step time       within 20%   within 35%   median |err|")
	# source A100-80GB, targets drawn from the table; per-kernel truth with GPU-specific efficiency
	src = (312e12, 2039e9)
	targets = [(989.4e12, 3350e9), (121e12, 300e9), (362e12, 864e9), (1307.4e12, 5300e9), (125e12, 600e9)]
	k = 40
	errs = {
		n: []
		for n in (
			"K1 per-kernel roofline replay (B)",
			"K2 whole step x peak-FLOPs ratio",
			"K3 whole step x bandwidth ratio",
			"K4 per-phase by arithmetic intensity",
			"K5 same MFU",
		)
	}
	k1_log = []
	for _ in range(4000):
		P_t, B_t = targets[rng.integers(len(targets))]
		cross_vendor = P_t == 1307.4e12
		f = rng.lognormal(np.log(1e10), 2.0, k)
		ai = rng.lognormal(np.log(60), 1.5, k)
		by = f / ai
		eff_s = rng.uniform(0.3, 0.85, k)
		drift = rng.normal(0, 0.25 if cross_vendor else 0.12, k)
		eff_t = np.clip(eff_s * np.exp(drift), 0.05, 0.95)
		roof = lambda P, B, f=f, by=by: np.maximum(f / P, by / B)  # noqa: E731
		lat = 5e-6
		t_s = roof(*src) / eff_s + lat
		t_t = roof(P_t, B_t) / eff_t + lat
		host = rng.uniform(0, 0.15) * t_s.sum()
		true_t = t_t.sum() + host
		meas_s = t_s.sum() + host
		k1 = (roof(P_t, B_t) / np.minimum(1, roof(*src) / (t_s - lat)) + lat).sum() + host
		k2 = meas_s * src[0] / P_t
		k3 = meas_s * src[1] / B_t
		comp = ai > src[0] / src[1]
		k4 = t_s[comp].sum() * src[0] / P_t + t_s[~comp].sum() * src[1] / B_t + host
		mfu = f.sum() / meas_s / src[0]
		k5 = f.sum() / (mfu * P_t)
		for n, v in zip(errs, (k1, k2, k3, k4, k5), strict=True):
			errs[n].append(v / true_t - 1)
		k1_log.append((cross_vendor, abs(np.log(k1 / true_t))))
	for n, e in errs.items():
		e = np.abs(np.array(e))
		print(f"  {n:<40} {pct(e <= 0.2)}      {pct(e <= 0.35)}      {np.median(e) * 100:4.1f}%")
	# the step-level band K1 needs: 90th percentile of |log error|, by vendor (sets SIGMA_SAME/CROSS_VENDOR)
	for name, cv in (("same vendor", False), ("cross vendor", True)):
		e = np.array([x for c, x in k1_log if c == cv])
		q90 = np.quantile(e, 0.9)
		print(
			f"  K1 {name:<13} q90 |log error| {q90:.3f} -> sigma {q90 / 1.6448536269514722:.3f} ({len(e)} trials)"
		)


def s4c(trials=4000):
	"""K1's step-level band, robust to the per-kernel drift being larger than S4 assumes.

	S4 draws per-kernel efficiency drift at 0.12 (same vendor) and 0.25 (cross vendor). Nothing
	measured backs those yet, so the true drift is scaled by 1x, 1.5x and 2x and each candidate band
	(S4's q90 times a margin) is scored for 90% coverage. Source A100-80GB, as in S4.
	"""
	g = np.random.default_rng(20261028)
	src = (312e12, 2039e9)
	targets = [
		(989.4e12, 3350e9, False),
		(121e12, 300e9, False),
		(362e12, 864e9, False),
		(1307.4e12, 5300e9, True),
	]
	base = {False: 0.101, True: 0.179}  # S4's q90 |log error| by vendor
	k = 40
	res = {}
	for scale in (1.0, 1.5, 2.0):
		errs = {False: [], True: []}
		for _ in range(trials):
			P_t, B_t, cv = targets[g.integers(len(targets))]
			f = g.lognormal(np.log(1e10), 2.0, k)
			by = f / g.lognormal(np.log(60), 1.5, k)
			eff_s = g.uniform(0.3, 0.85, k)
			eff_t = np.clip(eff_s * np.exp(g.normal(0, (0.25 if cv else 0.12) * scale, k)), 0.05, 0.95)
			lat = 5e-6
			roof_s, roof_t = np.maximum(f / src[0], by / src[1]), np.maximum(f / P_t, by / B_t)
			t_s = roof_s / eff_s + lat
			host = g.uniform(0, 0.15) * t_s.sum()
			true_t = (roof_t / eff_t + lat).sum() + host
			k1 = (t_s * roof_t / roof_s).sum() + host
			errs[cv].append(abs(np.log(k1 / true_t)))
		res[scale] = {cv: np.array(v) for cv, v in errs.items()}
	print("\nS4c K1 band vs drift larger than assumed     coverage at drift x1 / x1.5 / x2")
	for m in (1.0, 1.25, 1.5, 2.0):
		row = []
		for cv in (False, True):
			h = base[cv] * m
			cov = " / ".join(f"{100 * np.mean(res[sc][cv] <= h):5.1f}%" for sc in (1.0, 1.5, 2.0))
			row.append(f"{'cross' if cv else 'same'} h={h:.3f}: {cov}")
		print(f"  margin x{m:<5g}" + "    ".join(row))


def s4b(trials=20_000):
	"""Same-GPU step time below the smallest probe. Small batches leave the GPU under-used, so time
	falls slower than linearly: work(x) = c1 x / util(x), util = min(1, (x / x_sat)^alpha).
	Probes at x0 and x0/2 (A's half batch and half sequence share bs*s), 3% timing noise; the fit is
	SameGpuModel's (non-negative intercept and slope). The band starts from SIGMA_SAME_GPU (0.08)."""
	g = np.random.default_rng(20261027)
	x0 = 2.0 ** g.integers(9, 17, trials)
	over = g.uniform(0.002, 0.05, trials)  # fixed per-step host and launch time, s
	c1 = g.uniform(2e-7, 2e-5, trials)  # s per work unit at full utilisation
	x_sat = x0 * np.exp(g.uniform(np.log(0.05), np.log(4.0), trials))
	alpha = g.uniform(0.2, 0.8, trials)

	def t(x):
		col = (lambda v: v[:, None]) if np.ndim(x) == 2 else (lambda v: v)
		util = np.minimum(1.0, (x / col(x_sat)) ** col(alpha))
		return col(over) + col(c1) * x / util

	xs = np.stack([x0, x0 / 2, x0 / 2], 1)
	ys = t(xs) * g.lognormal(0, 0.03, xs.shape)
	mx, my = xs.mean(1), ys.mean(1)
	slope = ((xs - mx[:, None]) * (ys - my[:, None])).sum(1) / ((xs - mx[:, None]) ** 2).sum(1)
	icpt = my - slope * mx
	prop = (xs * ys).sum(1) / (xs * xs).sum(1)
	slope, icpt = (
		np.where(slope < 0, 0.0, np.where(icpt < 0, prop, slope)),
		np.where(slope < 0, my, np.where(icpt < 0, 0.0, icpt)),
	)
	h0 = 1.6448536269514722 * 0.08
	cands = {
		"T0 no widening": lambda h, k: h,
		"T1 x(1 + 0.25 per halving), memory's rule": lambda h, k: h * (1 + 0.25 * k),
		"T2 x(1 + 0.5 per halving)": lambda h, k: h * (1 + 0.5 * k),
		"T3 x(1 + 1.0 per halving)": lambda h, k: h * (1 + 1.0 * k),
		"T4 + 0.1 per halving": lambda h, k: h + 0.1 * k,
	}
	# one-sided: the fit runs low at small batches, so only the upper end grows
	upper = {
		"T5 upper end + 0.1 per halving": lambda h, k: h + 0.1 * k,
		"T6 upper end + 0.15 per halving": lambda h, k: h + 0.15 * k,
		"T7 upper end + 0.2 per halving": lambda h, k: h + 0.2 * k,
	}
	ks = [0.5, 1, 2, 2.5, 3, 4, 6]
	print(
		"\nS4b step time below the smallest probe   coverage / abstained (band past +-50%), by halvings below"
	)
	print("  " + " " * 44 + "".join(f"{k:>6g} halvings   " for k in ks))
	for name, grow in {**cands, **upper}.items():
		row = []
		for k in ks:
			q = (x0 / 2) * 2.0**-k
			pred = np.maximum(icpt + slope * q, 1e-12)
			err = np.log(t(q) / pred)  # > 0: the fit ran low
			hi = grow(h0, k)
			lo = h0 if name in upper else hi
			given = np.full(trials, hi <= np.log(1.5))
			cov = ((err <= hi) & (err >= -lo))[given].mean() if given.any() else np.nan
			row.append(f"{100 * cov:5.1f}% / {100 * (1 - given.mean()):5.1f}%")
		print(f"  {name:<44}" + "  ".join(row))


# ------------------------------------------------------------------ S5 corpus budget
def s5():
	print(
		"\nS5  corpus budget rule (budget $45, A100 $2.50/h)            P(overspend)   mean overspend   budget unused   configs"
	)
	trials = 3000
	budget, rate = 45.0, 2.50
	cands = {
		"B1 overhead 1.00, fixed 45 s": (1.0, None, False),
		"B2 overhead 1.25, fixed 45 s (B's first)": (1.25, None, False),
		"B3 overhead 1.25, running p75 of real configs": (1.25, 0.75, False),
		"B4 overhead 1.35, running p90, hold one wave in reserve": (1.35, 0.90, True),
		"B5 overhead 1.25, running p90, hold one wave in reserve": (1.25, 0.90, True),
	}
	for name, (oh, q, reserve) in cands.items():
		over, amt, unused, ran = [], [], [], []
		for _ in range(trials):
			true_over = rng.uniform(1.05, 1.35)  # cold starts, image pulls, idle scale-down, retries
			mean_s = rng.uniform(30, 90)
			spent, est, seen, n = 0.0, 45.0, [], 0
			while True:
				per = est * oh / 3600 * rate
				left = budget - spent - (32 * per if reserve and n else 0)
				room = int(left / per) if per > 0 else 0
				if room <= 0:
					break
				wave = min(room, 32)
				secs = rng.lognormal(np.log(mean_s), 0.6, wave)
				spent += secs.sum() * true_over / 3600 * rate
				seen.extend(secs)
				n += wave
				if q is not None:
					est = float(np.quantile(seen, q))
			over.append(spent > budget)
			amt.append(max(0.0, spent - budget))
			unused.append(max(0.0, budget - spent) / budget)
			ran.append(n)
		print(
			f"  {name:<60} {pct(over)}         ${np.mean(amt):5.2f}          {np.mean(unused) * 100:5.1f}%        {np.mean(ran):5.0f}"
		)


# ------------------------------------------------------------------ S6 agent totals
def s6(trials=10_000, pop=500):
	"""A's D3 generator (seed and laws from scripts/mc_agent_sample.py), scored with B's estimator."""
	from preflight.engine.agent_cost import estimate_total, runaway_rate

	g = np.random.default_rng(20261027)
	sig = g.uniform(0.5, 1.2, trials)
	p_run = g.uniform(0.0, 0.05, trials)
	mult = g.uniform(10, 40, trials)
	pops = []
	for i in range(trials):
		c = g.lognormal(np.log(0.05), sig[i], pop)
		run = g.random(pop) < p_run[i]
		full = np.where(run, c * mult[i], c)
		pops.append((full, (full > 8 * np.median(full)).mean()))  # the share risk.runaway estimates
	print(
		"\nS6  agent total: 90% interval covers the run total   (95% CI)       median width/total   runaway rate covered"
	)
	for n, bound in [
		(30, 0.75),
		(30, 0.80),
		(30, 0.85),
		(30, 0.90),
		(50, 0.75),
		(50, 0.80),
		(50, 0.85),
		(50, 0.90),
	]:
		r = np.random.default_rng(7)
		cov, width, rate_cov = np.zeros(trials, bool), np.zeros(trials), np.zeros(trials, bool)
		for i, (popc, true_rate) in enumerate(pops):
			smp = r.choice(popc, n, replace=False)
			e = estimate_total(smp, pop, runaway_bound=bound)
			total = popc.sum()
			cov[i] = e.lo <= total <= e.hi
			width[i] = (e.hi - e.lo) / total
			rr = runaway_rate(smp)
			rate_cov[i] = rr.lo <= true_rate <= rr.hi
		p = cov.mean()
		se = np.sqrt(p * (1 - p) / trials)
		print(
			f"  {n} tasks, {int(bound * 100)}% runaway bound{'':<22}{p * 100:5.1f}% ({(p - 1.96 * se) * 100:4.1f}-{(p + 1.96 * se) * 100:4.1f})"
			f"        {np.median(width):5.2f}              {pct(rate_cov)}"
		)


def s6b(trials=10_000, pop=500):
	"""Chiya's rule: 30 tasks, then 20 more (distinct tasks) only when the range straddles the budget.

	Topping up only when the first range straddles the budget is a choice made after seeing the data,
	so the final interval's coverage is checked here rather than assumed from S6. Budgets are drawn at
	0.5x to 2x the true total, log-uniform, so straddling cases are common.
	"""
	from preflight.engine.agent_cost import budget_check, estimate_total

	g = np.random.default_rng(20261027)
	sig = g.uniform(0.5, 1.2, trials)
	p_run = g.uniform(0.0, 0.05, trials)
	mult = g.uniform(10, 40, trials)
	r = np.random.default_rng(11)
	cov = np.zeros(trials, bool)
	topped = np.zeros(trials, bool)
	still = np.zeros(trials, bool)
	call_right = []
	cov30 = np.zeros(trials, bool)
	for i in range(trials):
		c = g.lognormal(np.log(0.05), sig[i], pop)
		full = np.where(g.random(pop) < p_run[i], c * mult[i], c)
		total = full.sum()
		budget = total * np.exp(r.uniform(np.log(0.5), np.log(2.0)))
		order = r.permutation(pop)
		e = estimate_total(full[order[:30]], pop)
		cov30[i] = e.lo <= total <= e.hi
		chk = budget_check(e, budget)
		if chk.offer_top_up:
			topped[i] = True
			e = estimate_total(full[order[: 30 + chk.top_up_tasks]], pop)
			chk = budget_check(e, budget)
			still[i] = chk.verdict == "straddles"
		cov[i] = e.lo <= total <= e.hi
		if chk.verdict in ("under", "over"):
			call_right.append((chk.verdict == "under") == (total <= budget))
	ci = lambda p: 1.96 * np.sqrt(p * (1 - p) / trials)  # noqa: E731
	print("\nS6b 30 tasks + top-up of 20 when the range straddles the budget (budget 0.5-2x the run total)")
	print(f"  30 tasks alone: coverage {pct(cov30)}")
	print(
		f"  topped up: {pct(topped)} of runs; still straddling after the top-up: {pct(still[topped])} of those"
	)
	print(
		f"  final coverage, all runs: {pct(cov)} (+-{100 * ci(cov.mean()):.1f}); topped-up runs: {pct(cov[topped])}"
	)
	print(
		f"  under/over calls that were right: {pct(np.array(call_right))} of {len(call_right)} runs with a clear call"
	)


if __name__ == "__main__":
	import sys

	if sys.argv[1:] == ["s6b"]:
		s6b()
		raise SystemExit
	if sys.argv[1:] == ["s6"]:
		s6()
		raise SystemExit

	if sys.argv[1:] == ["s3"]:
		s3()
		raise SystemExit
	if sys.argv[1:] == ["s2c"]:
		w = memory_world()
		s2c(w, probes(w))
		raise SystemExit
	if sys.argv[1:] == ["s4c"]:
		s4c()
		raise SystemExit
	if sys.argv[1:] == ["s4b"]:
		s4b()
		raise SystemExit
	if sys.argv[1:] == ["s2d"]:
		w = memory_world()
		s2d(w, probes(w))
		raise SystemExit
	w, truth, est, usable, pr = s1()
	p, u, q90, disagree, loo = s2(w, truth, est, usable, pr)
	s2b(w, truth, p, u, np.maximum(q90, 1.5 * loo))
	s2c(w, pr)
	s2d(w, pr)
	s3()
	s4()
	s4b()
	s4c()
	s5()
	s6()
	s6b()

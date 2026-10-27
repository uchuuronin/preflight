"""Monte Carlo for B's engine settings, fixed seed: 20k trials for memory, 4k for divergence and cross-GPU, 3k budget sweeps.

S1: memory estimator on A's three probes (current, half batch, half seq).
S2: the 90% band rule (S2a), the fits / tight / fails cut-offs (S2b), the band with one or two probes (S2c).
S3: the rule that labels a short run diverged.
S4: carrying step time from the probed GPU to another GPU.
S5: the corpus sweep's cost overhead factor and per-config time estimate.

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
	for n, e in errs.items():
		e = np.abs(np.array(e))
		print(f"  {n:<40} {pct(e <= 0.2)}      {pct(e <= 0.35)}      {np.median(e) * 100:4.1f}%")


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


if __name__ == "__main__":
	import sys

	if sys.argv[1:] == ["s3"]:
		s3()
		raise SystemExit
	if sys.argv[1:] == ["s2c"]:
		w = memory_world()
		s2c(w, probes(w))
		raise SystemExit
	w, truth, est, usable, pr = s1()
	p, u, q90, disagree, loo = s2(w, truth, est, usable, pr)
	s2b(w, truth, p, u, np.maximum(q90, 1.5 * loo))
	s2c(w, pr)
	s3()
	s4()
	s5()

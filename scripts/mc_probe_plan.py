"""Monte Carlo for three probe-plan decisions. 20k trials each, fixed seed.

D1: warmup/timed steps and the step-time estimator (target: within 15% of the long-run mean).
D2: which probe points calibrate memory (target: peak within 10%, no missed OOMs).
D3: how many agent tasks to sample, and the interval rule (target: 90% interval covers the full-run total).
"""

import numpy as np

rng = np.random.default_rng(20261027)
N = 20_000


def pct(x):
	return f"{100 * np.mean(x):5.1f}%"


# ------------------------------------------------------------------ D1 step time
def d1():
	print("D1  step time: P(|err|<=5%)  P(|err|<=15%)  mean|err|  probe steps")
	sigma = rng.uniform(0.02, 0.10, N)  # steady jitter (compute-bound .. data-bound)
	p_out = rng.uniform(0.0, 0.08, N)  # per-step stall chance (GC, worker refill, logging)
	out_mult = (1.5, 3.0)
	true_mean = np.exp(sigma**2 / 2) * (1 + p_out * (np.mean(out_mult) - 1))

	def steps(n):
		base = rng.lognormal(0, sigma[:, None], (N, n))
		stall = rng.random((N, n)) < p_out[:, None]
		base = base * np.where(stall, rng.uniform(*out_mult, (N, n)), 1.0)
		warm = np.ones((N, n))
		warm[:, 0] *= rng.uniform(2, 6, N)  # compile, autotune, first allocations
		if n > 1:
			warm[:, 1] *= rng.uniform(1.0, 1.4, N)
		if n > 2:
			warm[:, 2] *= np.where(rng.random(N) < 0.5, rng.uniform(1.0, 1.1, N), 1.0)
		return base * warm

	def trimmed(x, frac):
		x = np.sort(x, axis=1)
		k = int(np.floor(frac * x.shape[1]))
		return x[:, k : x.shape[1] - k].mean(axis=1) if k else x.mean(axis=1)

	cands = [
		("A  warmup 2, timed 3, mean", 2, 3, lambda x: x.mean(1)),
		("B  warmup 2, timed 3, median", 2, 3, lambda x: np.median(x, 1)),
		("C  warmup 2, timed 5, median", 2, 5, lambda x: np.median(x, 1)),
		("D  warmup 3, timed 8, 20% trimmed mean", 3, 8, lambda x: trimmed(x, 0.2)),
		("E  warmup 3, timed 8, mean", 3, 8, lambda x: x.mean(1)),
	]
	for name, w, t, est in cands:
		x = steps(w + t)[:, w:]
		err = est(x) / true_mean - 1
		print(
			f"  {name:<40} {pct(abs(err) <= 0.05)}      {pct(abs(err) <= 0.15)}     {np.mean(abs(err)) * 100:4.1f}%    {w + t}"
		)


# ------------------------------------------------------------------ D2 memory calibration
def d2():
	print("\nD2  memory: P(err<=10%)  missed OOM  false alarm  probes   (queries within 2x of probed bs*seq)")
	GB = 1e9
	cap = 24 * GB
	V, L, H, d = 151_936, 28, 12, 1536
	static = rng.uniform(3.6, 4.4, N) * GB  # weights + LoRA state + context
	act = d * L * rng.uniform(12, 22, N)  # bytes per token, linear terms
	logit_c = V * 4 * rng.uniform(1.0, 3.0, N)  # fp32 logits + softmax grad copies
	eager = rng.random(N) < 0.5  # eager attention keeps bs*H*L*s^2 scores
	attn = np.where(eager, H * L * 2 * rng.uniform(1, 3, N), 0.0)
	frag0 = rng.uniform(0.02, 0.12, N)

	def mem(bs, s, frag):
		alloc = static + bs * s * (act + logit_c) + bs * s * s * attn
		res = alloc * (1 + frag)
		return np.ceil(res / 2**21) * 2**21  # 2 MiB rounding

	bs0, s0 = 4, 1024
	grid = [(b, s) for b in (1, 2, 4, 8, 16) for s in (256, 512, 1024, 2048, 4096) if b * s <= 2 * bs0 * s0]
	qi = rng.integers(0, len(grid), N)
	qb = np.array([grid[i][0] for i in qi], float)
	qs = np.array([grid[i][1] for i in qi], float)
	frag_q = np.clip(frag0 + rng.normal(0, 0.02, N), 0, None)
	truth = mem(qb, qs, frag_q)

	def measure(bs, s):
		fr = np.clip(frag0 + rng.normal(0, 0.02, N), 0, None)
		return mem(bs, s, fr), fr

	def fit_linear(points):
		# M = S + beta * bs*s, on reserved values; frag folded into the fit
		(b1, s1, m1), (b2, s2, m2) = points
		beta = (m1 - m2) / (b1 * s1 - b2 * s2)
		S = m1 - beta * b1 * s1
		return lambda b, s: S + beta * b * s

	m00, _ = measure(bs0, s0)
	m_half_b, _ = measure(bs0 / 2, s0)
	m_dbl_b, _ = measure(2 * bs0, s0)
	m_half_s, _ = measure(bs0, s0 / 2)

	preds = {}
	preds["A  current + half batch (linear in tokens)"] = (
		fit_linear([(bs0, s0, m00), (bs0 / 2, s0, m_half_b)])(qb, qs),
		2,
	)
	# B: double batch, but it OOMs when over the cap -> fall back to half batch (3 probes then)
	dbl_ok = m_dbl_b < cap
	pb = np.where(
		dbl_ok,
		fit_linear([(bs0, s0, m00), (2 * bs0, s0, m_dbl_b)])(qb, qs),
		fit_linear([(bs0, s0, m00), (bs0 / 2, s0, m_half_b)])(qb, qs),
	)
	preds["B  current + double batch (half on OOM)"] = (pb, 2 + np.mean(~dbl_ok))
	preds["C  current + half seq (linear in tokens)"] = (
		fit_linear([(bs0, s0, m00), (bs0, s0 / 2, m_half_s)])(qb, qs),
		2,
	)
	# D: three probes, fit S + beta*bs*s + gamma*bs*s^2 exactly
	A = np.array(
		[
			[1, bs0 * s0, bs0 * s0 * s0],
			[1, bs0 / 2 * s0, bs0 / 2 * s0 * s0],
			[1, bs0 * s0 / 2, bs0 * s0 / 2 * s0 / 2],
		]
	)
	Ainv = np.linalg.inv(A)
	coef = Ainv @ np.vstack([m00, m_half_b, m_half_s])
	preds["D  current + half batch + half seq (adds s^2)"] = (
		coef[0] + coef[1] * qb * qs + coef[2] * qb * qs * qs,
		3,
	)
	# E: analytical only, nominal constants (no calibration probe)
	nominal = 4.0 * GB + qb * qs * (d * L * 17 + V * 4 * 2.0) + 0
	preds["E  analytical terms only, one probe"] = (nominal * 1.05, 1)

	for name, (p, n_probes) in preds.items():
		err = np.abs(p / truth - 1)
		missed = np.mean((truth > cap) & (p <= cap))
		false = np.mean((truth <= cap) & (p > cap))
		print(
			f"  {name:<48} {pct(err <= 0.10)}     {missed * 100:5.2f}%     {false * 100:5.2f}%     {n_probes:.2f}"
		)


# ------------------------------------------------------------------ D3 agent sample
def d3():
	print("\nD3  agent cost: coverage of 90% interval   median width/total   sample share of run")
	pop = 500
	sig = rng.uniform(0.5, 1.2, 4000)
	p_run = rng.uniform(0.0, 0.05, 4000)
	run_mult = rng.uniform(10, 40, 4000)
	trials = 4000

	def population(i):
		c = rng.lognormal(np.log(0.05), sig[i], pop)
		runaway = rng.random(pop) < p_run[i]
		return np.where(runaway, c * run_mult[i], c)

	def interval(sample, rule3):
		n = len(sample)
		boots = rng.choice(sample, (1000, n)).mean(1) * pop
		lo, hi = np.quantile(boots, [0.05, 0.95])
		if rule3:
			# if no runaway-sized task was seen, add the rule-of-three tail: up to 3/n of tasks could be runaways
			med = np.median(sample)
			seen = np.any(sample > 8 * med)
			if not seen:
				hi += pop * (3 / n) * 20 * med
		return lo, hi

	cands = [
		("A  20 tasks, bootstrap", 20, False),
		("B  30 tasks, bootstrap", 30, False),
		("C  50 tasks, bootstrap", 50, False),
		("D  30 tasks, bootstrap + rule-of-3 tail", 30, True),
		("E  50 tasks, bootstrap + rule-of-3 tail", 50, True),
	]
	for name, n, rule3 in cands:
		cov, width = [], []
		for i in range(trials):
			popc = population(i)
			total = popc.sum()
			lo, hi = interval(rng.choice(popc, n, replace=False), rule3)
			cov.append(lo <= total <= hi)
			width.append((hi - lo) / total)
		print(
			f"  {name:<42} {pct(cov)}                    {np.median(width):5.2f}          {n / pop * 100:4.1f}%"
		)


d1()
d2()
d3()

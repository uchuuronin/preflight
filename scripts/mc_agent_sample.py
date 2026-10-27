"""D3 rerun: agent sample size and runaway-tail margin. 10k trials per candidate, same generator as D3."""

import numpy as np

rng = np.random.default_rng(20261027)
POP, TRIALS, BOOT = 500, 10_000, 600
sig = rng.uniform(0.5, 1.2, TRIALS)
p_run = rng.uniform(0.0, 0.05, TRIALS)
run_mult = rng.uniform(10, 40, TRIALS)
pops = []
for i in range(TRIALS):
	c = rng.lognormal(np.log(0.05), sig[i], POP)
	pops.append(np.where(rng.random(POP) < p_run[i], c * run_mult[i], c))


def run(n, k):
	cov, width = np.zeros(TRIALS, bool), np.zeros(TRIALS)
	r = np.random.default_rng(7)  # same samples for every margin at a given n
	for i, popc in enumerate(pops):
		total = popc.sum()
		s = r.choice(popc, n, replace=False)
		boots = r.choice(s, (BOOT, n)).mean(1) * POP
		lo, hi = np.quantile(boots, [0.05, 0.95])
		med = np.median(s)
		if k and not np.any(s > 8 * med):
			hi += POP * (k / n) * 20 * med
		cov[i] = lo <= total <= hi
		width[i] = (hi - lo) / total
	p = cov.mean()
	se = np.sqrt(p * (1 - p) / TRIALS)
	return p, se, np.median(width)


print(f"{'candidate':<34}{'coverage (95% CI)':<24}{'median width/total':>18}")
rows = [(30, 3)] + [(50, k) for k in (0, 3, 4, 5, 6)]
for n, k in rows:
	p, se, w = run(n, k)
	label = f"{n} tasks, " + (f"tail {k}/n" if k else "plain bootstrap")
	print(
		f"{label:<34}{p * 100:5.1f}% ({(p - 1.96 * se) * 100:4.1f}-{(p + 1.96 * se) * 100:4.1f})       {w:10.2f}"
	)


def misses(n, k):
	r = np.random.default_rng(7)
	out = {"low_seen": 0, "low_unseen": 0, "high_seen": 0, "high_unseen": 0}
	for popc in pops:
		total = popc.sum()
		s = r.choice(popc, n, replace=False)
		boots = r.choice(s, (BOOT, n)).mean(1) * POP
		lo, hi = np.quantile(boots, [0.05, 0.95])
		med = np.median(s)
		seen = np.any(s > 8 * med)
		if k and not seen:
			hi += POP * (k / n) * 20 * med
		if total > hi:
			out["high_seen" if seen else "high_unseen"] += 1
		elif total < lo:
			out["low_seen" if seen else "low_unseen"] += 1
	return {key: f"{100 * v / TRIALS:.1f}%" for key, v in out.items()}


print("\nmisses, 50 tasks tail 3/n:", misses(50, 3))
print("misses, 50 tasks tail 6/n:", misses(50, 6))


def cp_upper(x, n, conf):
	"""One-sided Clopper-Pearson upper bound on a binomial rate, by bisection on the binomial CDF."""
	from math import comb

	if x >= n:
		return 1.0
	lo, hi = x / n, 1.0
	for _ in range(60):
		mid = (lo + hi) / 2
		cdf = sum(comb(n, i) * mid**i * (1 - mid) ** (n - i) for i in range(x + 1))
		lo, hi = (mid, hi) if cdf > 1 - conf else (lo, mid)
	return hi


def run_cp(n, conf):
	r = np.random.default_rng(7)
	cov, width = np.zeros(TRIALS, bool), np.zeros(TRIALS)
	ups = {x: cp_upper(x, n, conf) for x in range(n + 1)}
	for i, popc in enumerate(pops):
		total = popc.sum()
		s = r.choice(popc, n, replace=False)
		boots = r.choice(s, (BOOT, n)).mean(1) * POP
		lo, hi = np.quantile(boots, [0.05, 0.95])
		med = np.median(s)
		run_mask = s > 8 * med
		x = int(run_mask.sum())
		run_cost = s[run_mask].mean() if x else 20 * med
		hi += POP * (ups[x] - x / n) * run_cost
		cov[i] = lo <= total <= hi
		width[i] = (hi - lo) / total
	p = cov.mean()
	se = np.sqrt(p * (1 - p) / TRIALS)
	return p, se, np.median(width)


print("\nrunaway-rate upper bound applied whether or not a runaway was seen:")
for n in (30, 50):
	for conf in (0.70, 0.75, 0.80, 0.85, 0.90, 0.95):
		p, se, w = run_cp(n, conf)
		print(
			f"  {n} tasks, {int(conf * 100)}% bound: coverage {p * 100:5.1f}% ({(p - 1.96 * se) * 100:4.1f}-{(p + 1.96 * se) * 100:4.1f}), width {w:.2f}"
		)

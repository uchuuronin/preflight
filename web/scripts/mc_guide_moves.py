"""Mini Monte Carlo: how risky may a setting the guide proposes be? (docs/decisions/guide-moves.md)

Each simulated job has a ladder of settings (batch 1 to 64); cheaper settings use more memory.
The engine predicts peak memory with log-normal error at B's calibrated widths and turns it
into P(OOM) with B's p_oom. The guide proposes the cheapest setting its rule allows. A crash
loses 5 to 30% of that run (up to the first eval) and retries one step safer, paying again on
each crash. Results are compared with an oracle that knows the true peak.

	uv run python web/scripts/mc_guide_moves.py
"""

from __future__ import annotations

import math
import statistics

import numpy as np

Z90 = 1.6448536269514722
# B's 90% half-widths in log space (engine/memory.py) and how often each applies
WIDTHS = [(0.107, 0.6), (0.160, 0.3), (0.241, 0.1)]
LADDER = [1, 2, 4, 8, 16, 32, 64]
CAP_GIB = 80.0
FITS_BELOW, FAILS_AT = 0.01, 0.80
PROBE_WORTH = 0.10  # a tight setting must beat the best fitting one by this share of the aim
PROBE_COST = 0.02  # a short probe, as a share of the run it checks


def p_oom(point: float, sigma: float, cap: float) -> float:
	z = (math.log(cap) - math.log(point)) / sigma
	return 0.5 * math.erfc(z / math.sqrt(2))


RULES = {
	"A fits only (<1%)": FITS_BELOW,
	"B <5%": 0.05,
	"C <10% (first build)": 0.10,
	"D not fails (<80%)": FAILS_AT,
	"E fits, measure tight first": None,
}


def simulate(seed: int, n: int = 20_000) -> dict[str, dict[str, list[float]]]:
	rng = np.random.default_rng(seed)
	out: dict[str, dict[str, list[float]]] = {k: {"oom": [], "regret": []} for k in RULES}
	halves = [w for w, _ in WIDTHS]
	weights = [f for _, f in WIDTHS]
	for _ in range(n):
		resident = rng.uniform(10, 55)
		act = rng.uniform(0.3, 6.0)  # GiB per sample
		sigma = rng.choice(halves, p=weights) / Z90
		b0 = rng.uniform(4, 24)
		cost = {b: 1.0 / (1 - math.exp(-b / b0)) for b in LADDER}  # falls with batch, then flattens
		true = {b: resident + act * b for b in LADDER}
		err = rng.normal(0, sigma)  # one shared model error per job, as a biased fit would have
		pred = {b: true[b] * math.exp(err + rng.normal(0, sigma * 0.3)) for b in LADDER}
		p = {b: p_oom(pred[b], sigma, CAP_GIB) for b in LADDER}
		fitting = [b for b in LADDER if true[b] <= CAP_GIB]
		if not fitting:
			continue
		oracle = min(cost[b] for b in fitting)

		def realise(
			pick: int, true: dict[int, float] = true, cost: dict[int, float] = cost
		) -> tuple[float, bool]:
			total, crashed, i = 0.0, False, LADDER.index(pick)
			while true[LADDER[i]] > CAP_GIB and i > 0:
				crashed = True
				total += rng.uniform(0.05, 0.30) * cost[LADDER[i]]
				i -= 1
			return total + cost[LADDER[i]], crashed

		for name, limit in RULES.items():
			extra = 0.0
			if limit is None:
				fits = [b for b in LADDER if p[b] < FITS_BELOW] or [min(LADDER, key=lambda b: p[b])]
				pick = min(fits, key=lambda b: cost[b])
				tight = [
					b
					for b in LADDER
					if FITS_BELOW <= p[b] < FAILS_AT and cost[b] <= cost[pick] * (1 - PROBE_WORTH)
				]
				if tight:
					cand = min(tight, key=lambda b: cost[b])
					extra = PROBE_COST * cost[cand]
					if true[cand] <= CAP_GIB:
						pick = cand
			else:
				ok = [b for b in LADDER if p[b] < limit] or [min(LADDER, key=lambda b: p[b])]
				pick = min(ok, key=lambda b: cost[b])
			total, crashed = realise(pick)
			out[name]["oom"].append(float(crashed))
			out[name]["regret"].append((total + extra) / oracle - 1)
	return out


def main() -> None:
	for seed in (7, 11, 23):
		print(f"seed {seed}")
		for name, r in simulate(seed).items():
			reg = sorted(r["regret"])
			print(
				f"  {name:30s} crash {statistics.mean(r['oom']):6.2%}  "
				f"mean extra {statistics.mean(reg):6.2%}  p95 extra {reg[int(0.95 * len(reg))]:6.2%}"
			)


if __name__ == "__main__":
	main()

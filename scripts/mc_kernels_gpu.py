"""Monte Carlo for two probe decisions (Oct 27). Fixed seed; reproduce with
uv run python scripts/mc_kernels_gpu.py

K: how many kernel rows the profiled probe keeps for B's cross-GPU model (K1:
each kernel keeps its measured efficiency). Device time outside the table
lands in B's "host" bucket, which does not scale to the target GPU.

G: which GPU to probe on when the visitor has not chosen one.

L: the top of that ladder. Ties on memory go to the cheaper GPU (H100 holds no
more than A100-80GB), so the question is what, if anything, sits above 80 GB.
"""

import numpy as np

rng = np.random.default_rng(20261027)


def pct(x):
	return f"{100 * np.mean(x):5.1f}%"


# ---------------------------------------------------------------- K: kernel table size

CLASSES = [
	"gemm",
	"attention",
	"norm",
	"softmax",
	"elementwise",
	"reduction",
	"embedding",
	"index",
	"loss",
	"optimizer",
	"copy",
	"memset",
	"comm",
	"other",
]
COMPUTE = {"gemm", "attention"}


def kernel_trial():
	n = int(rng.integers(80, 700))  # distinct kernels in one step
	s = rng.uniform(0.6, 1.5)  # Zipf exponent of time by rank
	times = 1.0 / np.arange(1, n + 1) ** s
	times *= rng.lognormal(0, 0.3, n)
	order = np.argsort(-times)
	times = times[order]
	# big kernels are mostly gemm/attention; the tail is mostly memory-bound
	p_compute = np.clip(0.85 - 0.8 * np.arange(n) / n, 0.05, 0.9)
	is_compute = rng.random(n) < p_compute
	cls = np.where(is_compute, rng.choice(["gemm", "attention"], n, p=[0.8, 0.2]), rng.choice(CLASSES[2:], n))
	device = times / times.sum() * rng.uniform(0.7, 1.0)  # device share of the step; the rest is host time
	host = 1.0 - device.sum()
	# target GPU: compute scales by the peak ratio, memory-bound by the bandwidth ratio, comm by link ratio
	rc, rm, rl = rng.uniform(0.08, 3.0), rng.uniform(0.08, 3.0), rng.uniform(0.3, 3.0)
	ratio = np.where(np.isin(cls, list(COMPUTE)), rc, np.where(cls == "comm", rl, rm))
	ratio *= rng.lognormal(0, 0.08, n)  # per-kernel efficiency drift between GPUs
	truth = host + (device * ratio).sum()
	return device, cls, ratio, host, truth, rc, rm, rl


def predict(device, cls, ratio, host, k, roll_tail):
	kept = device[:k]
	pred = host + (kept * ratio[:k]).sum()
	tail = device[k:]
	if roll_tail and tail.size:
		# tail folded into one row per class: it scales by its class's ratio (no per-kernel drift known)
		for c in np.unique(cls[k:]):
			m = cls[k:] == c
			class_ratio = (
				np.median(ratio[:k][cls[:k] == c]) if np.any(cls[:k] == c) else np.median(ratio[k:][m])
			)
			pred += tail[m].sum() * class_ratio
	else:
		pred += tail.sum()  # unscaled, as B's host bucket
	return pred


def run_k(trials=20_000):
	print("K  kernel rows: P(err<=5%)  P(err<=20%)  median err  rows (median)")
	cands = [
		("top 32", 32, False),
		("top 64 (B's ask)", 64, False),
		("top 128", 128, False),
		("top 64 + tail rolled up per class", 64, True),
		("every kernel", 10_000, False),
	]
	data = [kernel_trial() for _ in range(trials)]
	for name, k, roll in cands:
		errs, rows = [], []
		for device, cls, ratio, host, truth, *_ in data:
			p = predict(device, cls, ratio, host, k, roll)
			errs.append(abs(p / truth - 1))
			n_tail_classes = len(np.unique(cls[k:])) if roll and len(cls) > k else 0
			rows.append(min(k, len(device)) + n_tail_classes)
		errs = np.array(errs)
		print(
			f"  {name:<36}{pct(errs <= 0.05)}     {pct(errs <= 0.20)}     {np.median(errs) * 100:5.1f}%     {int(np.median(rows))}"
		)


# ---------------------------------------------------------------- G: default probe GPU

# usable memory in GB (1e9), as probes/gpu_choice.py computes it from B's usable_capacity_gib()
GPUS = {  # name: (usable GB, $/h Modal, relative step speed vs L4 for bf16 training)
	"L4": (23.03, 0.80, 1.0),
	"A10": (23.71, 1.10, 1.1),
	"L40S": (47.42, 1.95, 2.6),
	"A100-80GB": (84.99, 2.50, 3.0),
	"H100": (85.02, 3.95, 6.5),
	"RTX-PRO-6000": (94.83, 3.03, 3.4),
	"H200": (149.88, 4.54, 7.0),
	"B200": (191.34, 6.25, 11.0),
}
# share of GPU jobs whose size nothing in the code gives away (a model built in the script, no
# parameter count, no size in a model id): the policy estimates 0 GB and starts on the cheapest GPU
UNKNOWN_SIZE = 0.2
LADDER = ["L4", "L40S", "A100-80GB", "H100"]


def job():
	mode = rng.choice(["lora", "full_ft", "pretrain", "infer", "agent"], p=[0.38, 0.17, 0.15, 0.15, 0.15])
	params = np.exp(rng.normal(np.log(1.2e9), 1.3))  # median 1.2B, heavy tail to tens of B
	params = float(np.clip(params, 5e7, 7e10))
	bytes_per_param = {"lora": 2.3, "full_ft": 16.0, "pretrain": 16.0, "infer": 2.1, "agent": 0.0}[mode]
	act = (
		rng.lognormal(np.log(4.0), 0.8)
		if mode in ("lora", "full_ft", "pretrain")
		else rng.lognormal(np.log(1.5), 0.6)
	)
	need_gb = params * bytes_per_param / 1e9 + act + 1.0
	if mode == "agent":
		need_gb = 0.0  # API calls: no GPU at all
	return mode, params, need_gb


def probe_seconds(params, gpu, mode):
	if mode == "agent":
		return 0.0
	load = 40 + params * 2 / 1.0e9  # container start plus weights at ~1 GB/s
	step_l4 = params / 1.2e9 * 0.9  # seconds per step on an L4 at 1.2B params
	steps = 7 * 3 + 5  # three probes of warmup 2 + timed 5, plus profiled steps
	return load * 3 + steps * step_l4 / GPUS[gpu][2]


def run_g(trials=40_000):
	print(
		"\nG  default probe GPU: measured on 1st try   measured at all   mean GPU $/job   P($ > 0.60)   mean minutes"
	)
	jobs = [job() for _ in range(trials)]
	est_noise = rng.lognormal(0, 0.3, trials)  # the analytical memory estimate's error before any probe
	est_noise[rng.random(trials) < UNKNOWN_SIZE] = 0.0

	def attempt(gpu, mode, params, need):
		fits = need <= GPUS[gpu][0]
		secs = (
			probe_seconds(params, gpu, mode) if fits else 60 + params * 2 / 1e9
		)  # an OOM fails at load/first step
		return fits, secs, secs * GPUS[gpu][1] / 3600

	def fixed(gpu):
		def pol(i, mode, params, need):
			ok, s, c = attempt(gpu, mode, params, need)
			return ok, ok, s, c

		return pol

	def sized(margin, escalate):
		def pol(i, mode, params, need):
			est = need * est_noise[i]
			order = [g for g in LADDER if GPUS[g][0] >= est * margin] or [LADDER[-1]]
			first = order[0]
			ok, s, c = attempt(first, mode, params, need)
			if ok or not escalate:
				return ok, ok, s, c
			nxt = [g for g in LADDER if GPUS[g][0] >= GPUS[first][0] * 1.05]
			if not nxt:
				return False, False, s, c
			ok2, s2, c2 = attempt(nxt[0], mode, params, need)
			return False, ok2, s + s2, c + c2

		return pol

	cands = [
		("always L4", fixed("L4")),
		("always A100-80GB", fixed("A100-80GB")),
		("always H100", fixed("H100")),
		("size-aware, 1.3x margin, escalate once", sized(1.3, True)),
		("size-aware, 1.15x margin, escalate once", sized(1.15, True)),
	]
	for name, pol in cands:
		first, any_, cost, mins = [], [], [], []
		for i, (mode, params, need) in enumerate(jobs):
			if mode == "agent":
				continue
			ok1, ok, s, c = pol(i, mode, params, need)
			first.append(ok1)
			any_.append(ok)
			cost.append(c)
			mins.append(s / 60)
		cost = np.array(cost)
		print(
			f"  {name:<42}{pct(first)}               {pct(any_)}          ${cost.mean():.3f}        {pct(cost > 0.60)}       {np.mean(mins):5.1f}"
		)


def run_l(trials=40_000):
	print(
		"\nL  ladder top (size-aware 1.3x, one step up): measured   mean GPU $/job   P($ > 0.60)   mean minutes"
	)
	jobs = [job() for _ in range(trials)]
	est_noise = rng.lognormal(0, 0.3, trials)
	est_noise[rng.random(trials) < UNKNOWN_SIZE] = 0.0

	def attempt(gpu, mode, params, need):
		fits = need <= GPUS[gpu][0]
		secs = probe_seconds(params, gpu, mode) if fits else 60 + params * 2 / 1e9
		return fits, secs, secs * GPUS[gpu][1] / 3600

	def policy(ladder):
		def pol(i, mode, params, need):
			est = need * est_noise[i]
			fits = [g for g in ladder if GPUS[g][0] >= est * 1.3]
			first = (
				min(fits, key=lambda g: GPUS[g][1])
				if fits
				else max(ladder, key=lambda g: (GPUS[g][0], -GPUS[g][1]))
			)
			ok, s, c = attempt(first, mode, params, need)
			if ok:
				return ok, s, c
			bigger = [g for g in ladder if GPUS[g][0] >= GPUS[first][0] * 1.05]
			if not bigger:
				return False, s, c
			ok2, s2, c2 = attempt(min(bigger, key=lambda g: GPUS[g][1]), mode, params, need)
			return ok2, s + s2, c + c2

		return pol

	base = ["L4", "L40S", "A100-80GB"]
	for name, ladder in [
		("top A100-80GB", base),
		("top A100-80GB + H100 (tie)", [*base, "H100"]),
		("+ RTX-PRO-6000 96 GB", [*base, "RTX-PRO-6000"]),
		("+ H200 141 GB", [*base, "H200"]),
		("+ H200 + B200 180 GB", [*base, "H200", "B200"]),
	]:
		pol = policy(ladder)
		ok_, cost, mins = [], [], []
		for i, (mode, params, need) in enumerate(jobs):
			if mode == "agent":
				continue
			ok, sec, c = pol(i, mode, params, need)
			ok_.append(ok)
			cost.append(c)
			mins.append(sec / 60)
		cost = np.array(cost)
		print(
			f"  {name:<42}{pct(ok_)}     ${cost.mean():.3f}        {pct(cost > 0.60)}       {np.mean(mins):5.1f}"
		)


def run_u(trials=40_000):
	"""U: what to do after an OOM when nothing gave the model's size away (the estimate was 0)."""
	print(
		"\nU  unknown size, ladder L4/L40S/A100/H200: measured (all)  measured (unknown)  mean $/job  mean minutes"
	)
	jobs = [job() for _ in range(trials)]
	est_noise = rng.lognormal(0, 0.3, trials)
	unknown = rng.random(trials) < UNKNOWN_SIZE
	ladder = ["L4", "L40S", "A100-80GB", "H200"]

	def attempt(gpu, mode, params, need):
		fits = need <= GPUS[gpu][0]
		secs = probe_seconds(params, gpu, mode) if fits else 60 + params * 2 / 1e9
		return fits, secs, secs * GPUS[gpu][1] / 3600

	def policy(start_unknown, steps_unknown, steps_known=1):
		def pol(i, mode, params, need):
			if unknown[i]:
				first = start_unknown
			else:
				est = need * est_noise[i]
				fits = [g for g in ladder if GPUS[g][0] >= est * 1.3]
				first = fits[0] if fits else ladder[-1]
			plan = [first]
			if unknown[i] and steps_unknown == "largest":
				plan.append(ladder[-1])
			else:
				n = steps_unknown if unknown[i] else steps_known
				idx = ladder.index(first)
				plan += ladder[idx + 1 : idx + 1 + n]
			total_s = total_c = 0.0
			for g in dict.fromkeys(plan):
				ok, sec, c = attempt(g, mode, params, need)
				total_s, total_c = total_s + sec, total_c + c
				if ok:
					return True, total_s, total_c
			return False, total_s, total_c

		return pol

	for name, pol in [
		("L4, one step up", policy("L4", 1)),
		("L4, up to two steps", policy("L4", 2)),
		("L4, then straight to the largest", policy("L4", "largest")),
		("L40S, one step up", policy("L40S", 1)),
		("L4, walk the whole ladder (chosen)", policy("L4", 3)),
		("whole ladder, and two steps when sized", policy("L4", 3, 2)),
	]:
		ok_all, ok_unk, cost, mins = [], [], [], []
		for i, (mode, params, need) in enumerate(jobs):
			if mode == "agent":
				continue
			ok, sec, c = pol(i, mode, params, need)
			ok_all.append(ok)
			if unknown[i]:
				ok_unk.append(ok)
			cost.append(c)
			mins.append(sec / 60)
		print(
			f"  {name:<40}{pct(ok_all)}           {pct(ok_unk)}           ${np.mean(cost):.3f}     {np.mean(mins):5.1f}"
		)


run_k()
run_g()
run_l()
run_u()

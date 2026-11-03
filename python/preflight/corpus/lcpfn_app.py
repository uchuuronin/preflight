"""LC-PFN's public weights into a Modal volume (Chiya, Nov 1: yes to one CPU fetch run, about one cent).

LC-PFN (Adriaensen et al., NeurIPS 2023; github.com/automl/lcpfn, MIT) is a transformer trained on synthetic learning
curves that gives a posterior over a curve's future from its first steps. The `lcpfn` package pins torch 1.11 and
numpy below 2, and downloads its checkpoints from ml.informatik.uni-freiburg.de, which the agents' workspace can't
reach. This run fetches both checkpoints in a small image with the package's own pins, records their sha256, and
saves what a modern torch can read with `weights_only=True`: each model's state dict, its hyperparameters, and the
package's own quantile predictions on a fixed curve, so the port in the runner can be checked against them.

    modal run --env main python/preflight/corpus/lcpfn_app.py::fetch
    modal run --env main python/preflight/corpus/lcpfn_app.py::check   # the port (corpus/lcpfn.py) against them
    modal run --env main python/preflight/corpus/lcpfn_app.py::coverage --curves <curves.json.gz>  # held out

Writes out/lcpfn_fetch.json, out/lcpfn_check.json and out/lcpfn_coverage.json. The weights stay in the `preflight-lcpfn` volume, never in git.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import modal

APP_NAME = "preflight-lcpfn-fetch"
VOLUME = "preflight-lcpfn"
MOUNT = "/lcpfn"
URL = "https://ml.informatik.uni-freiburg.de/research-artifacts/lcpfn/{name}.gz"
NAMES = (
	"pfn_EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000.pt",
	"pfn_EPOCH1000_EMSIZE512_NLAYERS6_NBUCKETS1000.pt",
)
# cited: lcpfn 0.1.3's requires_dist (torch<=1.11.0, numpy<2,>=1.21.2, requests); Python 3.10 has torch 1.11 wheels
PINS = ("lcpfn==0.1.3", "torch==1.11.0", "numpy==1.24.4", "requests==2.32.3")

app = modal.App(APP_NAME)
vol = modal.Volume.from_name(VOLUME, create_if_missing=True)
image = modal.Image.debian_slim(python_version="3.10").pip_install(*PINS)

# a fixed curve for the reference predictions: a noisy power law over 100 steps, normalised to [0, 1] as LC-PFN expects
REFERENCE_STEPS = 100
REFERENCE_SEEN = 30
REFERENCE_QS = (0.05, 0.5, 0.95)


@app.function(image=image, volumes={MOUNT: vol}, timeout=15 * 60, cpu=2, memory=4096)
def fetch_weights() -> dict:
	import gzip
	import hashlib
	import shutil

	import lcpfn
	import requests
	import torch

	# str(): torch.__version__ is a TorchVersion, which can't be unpickled where torch isn't installed (Nov 1)
	out: dict = {"package": "lcpfn==0.1.3", "torch": str(torch.__version__), "models": {}}
	for name in NAMES:
		gz = Path(MOUNT) / f"{name}.gz"
		if not gz.exists():
			t0 = time.monotonic()
			r = requests.get(URL.format(name=name), timeout=300)
			r.raise_for_status()
			gz.write_bytes(r.content)
			took = round(time.monotonic() - t0, 1)
		else:
			took = None
		raw = Path("/tmp") / name
		with gzip.open(gz, "rb") as src, open(raw, "wb") as dst:
			shutil.copyfileobj(src, dst)
		model = torch.load(
			raw, map_location="cpu"
		)  # the published file is a whole pickled module (lcpfn.model)
		model.eval()
		state = {k: v.detach().clone() for k, v in model.state_dict().items()}
		layer0 = model.transformer_encoder.layers[0]
		meta = {
			"url": URL.format(name=name),
			"sha256_gz": hashlib.sha256(gz.read_bytes()).hexdigest(),
			"sha256_pt": hashlib.sha256(raw.read_bytes()).hexdigest(),
			"bytes_gz": gz.stat().st_size,
			"download_s": took,
			"class": f"{type(model).__module__}.{type(model).__name__}",
			"ninp": int(model.ninp),
			"nhead": int(layer0.self_attn.num_heads),
			"nhid": int(layer0.linear1.out_features),
			"nlayers": len(model.transformer_encoder.layers),
			"n_out": int(getattr(model, "n_out", 0) or 0) or None,
			"activation": getattr(getattr(layer0, "activation", None), "__name__", None),
			"pre_norm": bool(getattr(layer0, "pre_norm", False)),
			"modules": repr(model)[:4000],
			"state_keys": {k: list(v.shape) for k, v in state.items()},
		}
		stem = name.removesuffix(".pt")
		torch.save(state, Path(MOUNT) / f"{stem}.state.pt")
		(Path(MOUNT) / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1))
		# the package's own predictions on a fixed curve, for checking the port
		# review, Nov 1: LCPFN() reloads the pickle without map_location; reuse the model already loaded onto the CPU
		pfn = object.__new__(lcpfn.LCPFN)
		torch.nn.Module.__init__(pfn)
		pfn.model = model
		x = torch.arange(1, REFERENCE_STEPS + 1, dtype=torch.float32).unsqueeze(1)
		g = torch.Generator().manual_seed(0)
		y = (0.9 * x.squeeze(1) ** -0.35 + 0.02 * torch.randn(REFERENCE_STEPS, generator=g)).clamp(0, 1)
		q = pfn.predict_quantiles(
			x_train=x[:REFERENCE_SEEN],
			y_train=y[:REFERENCE_SEEN],
			x_test=x[REFERENCE_SEEN:],
			qs=list(REFERENCE_QS),
		)
		meta["reference"] = {
			"x_seen": REFERENCE_SEEN,
			"y_train": [round(float(v), 6) for v in y[:REFERENCE_SEEN]],
			"qs": list(REFERENCE_QS),
			"quantiles_at_100": [round(float(v), 6) for v in q[-1]],
		}
		(Path(MOUNT) / f"{stem}.meta.json").write_text(json.dumps(meta, indent=1))
		out["models"][stem] = {k: v for k, v in meta.items() if k not in ("modules", "state_keys")}
	vol.commit()
	return out


# the port's check (Nov 1): the runner's torch, CPU wheels; corpus/lcpfn.py loads the saved state dicts with
# weights_only=True and must give the package's own quantiles on the reference curve
CHECK_TOLERANCE = (
	1e-4  # guess: the meta keeps the curve to 6 decimals; the port matched the package's own code to 2.4e-6
)
port_image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install("torch==2.8.0", "numpy", index_url="https://download.pytorch.org/whl/cpu")
	.add_local_python_source("preflight")
)


@app.function(image=port_image, volumes={MOUNT: vol}, timeout=10 * 60, cpu=2, memory=4096)
def check_port() -> dict:
	import torch

	from preflight.corpus import lcpfn

	out: dict = {"torch": str(torch.__version__), "models": {}}
	for name in NAMES:
		stem = name.removesuffix(".pt")
		meta = json.loads((Path(MOUNT) / f"{stem}.meta.json").read_text())
		row: dict = {}
		try:
			model = lcpfn.load(Path(MOUNT) / f"{stem}.state.pt", meta)
		except (
			Exception
		) as exc:  # a key or shape the port doesn't have: report it, with the original's structure
			row["load_error"] = f"{type(exc).__name__}: {exc}"[:2000]
			row["modules"] = meta.get("modules", "")[:4000]
			out["models"][stem] = row
			continue
		try:
			ref = meta["reference"]
			x = torch.arange(1, REFERENCE_STEPS + 1, dtype=torch.float32)
			y = torch.tensor(ref["y_train"], dtype=torch.float32)
			q = model.predict_quantiles(x[: ref["x_seen"]], y, x[ref["x_seen"] :], list(ref["qs"]))[-1]
			got = [round(float(v), 6) for v in q]
			diff = max(abs(a - b) for a, b in zip(got, ref["quantiles_at_100"], strict=True))
			# a loss curve through the whole path: a falling power law seen for 200 of 2,000 steps
			steps = torch.arange(1, 201, dtype=torch.float32)
			loss = (2.0 + 4.0 * steps**-0.5).tolist()
			fc = lcpfn.forecast_loss(model, loss, last_step=2000)
			row.update(
				{
					"quantiles_at_100": got,
					"package_quantiles_at_100": ref["quantiles_at_100"],
					"max_abs_diff": diff,
					"matches": diff <= CHECK_TOLERANCE,
					"loss_forecast_at_2000": None if fc is None else [fc.lo, fc.mid, fc.hi],
					"loss_true_at_2000": 2.0 + 4.0 * 2000**-0.5,
				}
			)
		except Exception as exc:  # review, Nov 1: one model's failure fills its own row, not the whole run
			row["check_error"] = f"{type(exc).__name__}: {exc}"[:2000]
		out["models"][stem] = row
	return out


@app.local_entrypoint()
def check():
	"""modal run --env main python/preflight/corpus/lcpfn_app.py::check  (CPU, about a minute)"""
	t0 = time.monotonic()
	got = check_port.remote()
	got["wall_s"] = round(time.monotonic() - t0, 1)
	Path("out").mkdir(exist_ok=True)
	Path("out/lcpfn_check.json").write_text(json.dumps(got, indent=1))
	print(json.dumps(got, indent=1))
	if not all(m.get("matches") for m in got["models"].values()):
		raise SystemExit("the port doesn't match the package's quantiles")


# Nov 2 overnight list, item 6: LC-PFN's quantiles on held-out real loss curves (scripts/loss_coverage.py prepares
# them and scores the band); the 12-layer network, the package's default
COVERAGE_MODEL = "pfn_EPOCH1000_EMSIZE512_NLAYERS12_NBUCKETS1000"


@app.function(image=port_image, volumes={MOUNT: vol}, timeout=30 * 60, cpu=4, memory=4096)
def coverage_quantiles(doc: dict) -> dict:
	import torch

	from preflight.corpus import lcpfn

	torch.set_num_threads(4)
	meta = json.loads((Path(MOUNT) / f"{COVERAGE_MODEL}.meta.json").read_text())
	model = lcpfn.load(Path(MOUNT) / f"{COVERAGE_MODEL}.state.pt", meta)
	rows = []
	with torch.no_grad():
		for c in doc["curves"]:
			for seen in doc["seen_steps"]:
				try:
					# Nov 2 review: per-x means (q), beside the raw per-step input the first run used (q_raw)
					q = lcpfn.forecast_loss(model, c["seen"][:seen], last_step=c["steps"])
					r = lcpfn.forecast_loss(model, c["seen"][:seen], last_step=c["steps"], bin_x=False)
					rows.append(
						{
							"key": c["key"],
							"seen": seen,
							"q": None if q is None else [q.lo, q.mid, q.hi],
							"q_raw": None if r is None else [r.lo, r.mid, r.hi],
						}
					)
				except Exception as exc:  # one curve's failure is its own row, not the whole run's
					rows.append(
						{
							"key": c["key"],
							"seen": seen,
							"q": None,
							"error": f"{type(exc).__name__}: {exc}"[:300],
						}
					)
	return {"model": COVERAGE_MODEL, "port": lcpfn.MODEL_ID, "torch": str(torch.__version__), "rows": rows}


@app.local_entrypoint()
def coverage(curves: str):
	"""modal run --env main python/preflight/corpus/lcpfn_app.py::coverage --curves <curves.json.gz>  (CPU)"""
	import gzip

	t0 = time.monotonic()
	with gzip.open(curves, "rt") as f:
		doc = json.load(f)
	got = coverage_quantiles.remote(doc)
	got["wall_s"] = round(time.monotonic() - t0, 1)
	got["curves"] = curves
	Path("out").mkdir(exist_ok=True)
	Path("out/lcpfn_coverage.json").write_text(json.dumps(got))
	errs = sum(1 for r in got["rows"] if r.get("error"))
	print(f"{len(got['rows'])} forecasts, {errs} errors, {got['wall_s']} s")


@app.local_entrypoint()
def fetch():
	t0 = time.monotonic()
	got = fetch_weights.remote()
	got["wall_s"] = round(time.monotonic() - t0, 1)
	Path("out").mkdir(exist_ok=True)
	Path("out/lcpfn_fetch.json").write_text(json.dumps(got, indent=1))
	print(json.dumps(got, indent=1))

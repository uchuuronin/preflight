"""GPU spec and price table for NVIDIA and AMD accelerators.

One row per device as PyTorch sees it: an MI250X is two 64 GB devices (one per
GCD), so its row holds per-GCD figures. Compute peaks are dense (no structured
sparsity), in TFLOPS; GeForce tensor peaks use fp32 accumulate, which PyTorch
uses. Bandwidth is peak memory bandwidth in GB/s. `link_gbps` is the quoted
GPU-to-GPU bandwidth (NVLink / Infinity Fabric, or the PCIe slot), both
directions combined.

Every row says how sure it is: "datasheet" (vendor figure), "derived" (half of
a vendor with-sparsity figure, or cores / per-SM rate) or "secondary"
(a spec database). Predictions onto a "secondary" GPU carry wider bands.

Canonical names follow the schema's examples (H100-SXM-80GB, MI300X, L4): a
bare model name where there is one variant, form factor and memory where
there are several. `resolve()` maps PyTorch device names, ROCm arch names,
MIG slices and Modal strings onto them.

Pure Python, standard library only. Safe in Pyodide.

Sources (read Oct 26-27, 2026): NVIDIA datasheets and architecture
whitepapers (Volta, Turing, GA102, Ampere A100, Ada, Hopper, Blackwell),
Lenovo Press (A10, L40, B200), AMD ROCm MI100/MI300 docs and Instinct
brochures (MI210, MI250, MI355X), TechPowerUp (TITAN X Pascal), press
launch coverage (TITAN V, Maxwell TITAN X), flopper.io (RTX PRO 6000 tensor
rates). Prices: Modal list rates reported July 2026 (spheron.network), A100-40GB from
computeprices.com (Oct 2026); AMD
Developer Cloud MI300X from the Preflight contract (gap 8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Literal

from preflight.engine import prices as _prices

TFLOP = 1e12  # derived: unit
GB = 1e9  # derived: unit
GIB = 2**30  # derived: unit

Confidence = Literal["datasheet", "derived", "secondary", "guess"]

_HBM_PREFIXES = ("P100", "V100", "TITAN-V", "A100", "H100", "H200", "GH200", "B200", "B300")
# Data-centre GDDR cards ship with ECC on, which reserves part of the memory (L4: 21.45 of 24 GiB)
_ECC_GDDR = frozenset({"T4", "L4", "A10", "A16", "A40", "L40", "L40S", "RTX-PRO-6000-96GB", "M40-24GB"})
Vendor = Literal["nvidia", "amd"]


@dataclass(frozen=True)
class GpuSpec:
	name: str
	vendor: Vendor
	arch: str
	memory_gib: float
	bandwidth_gbps: float
	compute_units: int  # SMs on NVIDIA, CUs on AMD
	tdp_watts: int
	# dense peaks in TFLOPS; None means no fast path for that dtype on this GPU
	fp32: float  # what an fp32 matmul reaches without TF32 (CUDA cores; matrix cores on AMD)
	tf32: float | None
	bf16: float | None
	fp16: float | None
	fp8: float | None
	link_gbps: float  # GPU-to-GPU, both directions
	link: str  # nvlink, xgmi, pcie3, pcie4, pcie5
	confidence: Confidence = "datasheet"
	notes: str = ""
	aliases: tuple[str, ...] = field(default_factory=tuple)
	mig_slices: int = 0  # compute slices a MIG GPU splits into; 0 = no MIG
	usable_gib: float | None = None  # what torch.cuda total_memory reports, when known

	def supports(self, dtype: str) -> bool:
		return self.peak_tflops(dtype) is not None

	def peak_tflops(self, dtype: str, allow_tf32: bool = False) -> float | None:
		"""Dense peak for a dtype. fp32 uses TF32 only when asked: PyTorch leaves it off by default."""
		d = normalise_dtype(dtype)
		if d == "fp32":
			return self.tf32 if (allow_tf32 and self.tf32 is not None) else self.fp32
		if d == "tf32":
			return self.tf32
		return getattr(self, d)

	def peak_flops(self, dtype: str, allow_tf32: bool = False) -> float | None:
		tf = self.peak_tflops(dtype, allow_tf32)
		return None if tf is None else tf * TFLOP

	def bandwidth_bytes(self) -> float:
		return self.bandwidth_gbps * GB

	def link_bytes_one_way(self) -> float:
		return self.link_gbps * GB / 2

	@property
	def hbm(self) -> bool:
		return self.vendor == "amd" or self.name.startswith(_HBM_PREFIXES)

	@property
	def usable_source(self) -> str:
		return "reported" if self.usable_gib is not None else "estimated"

	def usable_capacity_gib(self) -> float:
		"""Memory a process can use, as PyTorch reports it: less than the box says (ECC, driver reservations)."""
		if self.usable_gib is not None:
			return self.usable_gib
		if self.hbm:
			return (
				self.memory_gib * 0.99
			)  # cited: published torch total_memory for the A100 and H100 rows, ~99% of HBM
		# cited: published torch total_memory for the T4 and L4 rows (ECC GDDR reserves ~8%); 0.985 is a guess for
		# other GDDR cards, replaced by one get_device_properties call per card as each joins the corpus
		return self.memory_gib * (0.92 if self.name in _ECC_GDDR else 0.985)

	def capacity_bytes(self) -> int:
		"""Usable bytes (see usable_capacity_gib). Prefer the probe's measured total when there is one."""
		return int(self.usable_capacity_gib() * GIB)

	def nominal_bytes(self) -> int:
		return int(self.memory_gib * GIB)

	def ridge_point(self, dtype: str, allow_tf32: bool = False) -> float:
		"""Arithmetic intensity (FLOP/byte) where compute and bandwidth limits meet."""
		peak = self.peak_flops(dtype, allow_tf32)
		if peak is None:
			raise ValueError(f"{self.name} has no fast path for {dtype}")
		return peak / self.bandwidth_bytes()


_DTYPE_ALIASES = {
	"float32": "fp32",
	"float": "fp32",
	"fp32": "fp32",
	"f32": "fp32",
	"tf32": "tf32",
	"bfloat16": "bf16",
	"bf16": "bf16",
	"float16": "fp16",
	"half": "fp16",
	"fp16": "fp16",
	"f16": "fp16",
	"fp8": "fp8",
	"float8": "fp8",
	"float8_e4m3fn": "fp8",
	"float8_e4m3fnuz": "fp8",
	"float8_e5m2": "fp8",
	"float8_e5m2fnuz": "fp8",
	"e4m3": "fp8",
	"e5m2": "fp8",
}


def normalise_dtype(dtype: str) -> str:
	key = str(dtype).lower().replace("torch.", "").strip()
	if key not in _DTYPE_ALIASES:
		raise ValueError(f"unknown dtype {dtype!r}")
	return _DTYPE_ALIASES[key]


def _g(name, vendor, arch, mem, bw, cu, tdp, fp32, tf32, bf16, fp16, fp8, link, link_kind, **kw) -> GpuSpec:
	return GpuSpec(name, vendor, arch, mem, bw, cu, tdp, fp32, tf32, bf16, fp16, fp8, link, link_kind, **kw)


# fmt: off
_SPECS: tuple[GpuSpec, ...] = (
	# --- Maxwell / Pascal: no tensor cores
	_g("M40-24GB", "nvidia", "maxwell", 24, 288, 24, 250, 7.0, None, None, None, None, 32, "pcie3", aliases=("tesla m40",)),
	_g("TITAN-X-Maxwell", "nvidia", "maxwell", 12, 336.5, 24, 250, 7.0, None, None, None, None, 32, "pcie3",
		confidence="secondary", aliases=("geforce gtx titan x",)),
	_g("GTX-1080-Ti", "nvidia", "pascal", 11, 484, 28, 250, 11.3, None, None, None, None, 32, "pcie3",
		aliases=("geforce gtx 1080 ti", "gtx 1080 ti")),
	_g("TITAN-X-Pascal", "nvidia", "pascal", 12, 480, 28, 250, 11.0, None, None, None, None, 32, "pcie3",
		confidence="secondary", aliases=("titan x (pascal)", "titan xp")),
	_g("P100-16GB", "nvidia", "pascal", 16, 732, 56, 250, 9.3, None, None, None, None, 32, "pcie3",
		notes="fp16 runs at 18.7 TFLOPS on CUDA cores; no tensor cores.", aliases=("tesla p100", "p100")),
	# --- Volta / Turing: fp16 tensor cores, no bf16
	_g("V100-SXM2-16GB", "nvidia", "volta", 16, 900, 80, 300, 15.7, None, None, 125.0, None, 300, "nvlink",
		aliases=("tesla v100-sxm2-16gb",), usable_gib=15.77),
	_g("V100-SXM2-32GB", "nvidia", "volta", 32, 900, 80, 300, 15.7, None, None, 125.0, None, 300, "nvlink",
		aliases=("tesla v100-sxm2-32gb", "v100-sxm2"), usable_gib=31.74),
	_g("V100-PCIe-16GB", "nvidia", "volta", 16, 900, 80, 250, 14.0, None, None, 112.0, None, 32, "pcie3",
		aliases=("tesla v100-pcie-16gb",)),
	_g("V100-PCIe-32GB", "nvidia", "volta", 32, 900, 80, 250, 14.0, None, None, 112.0, None, 32, "pcie3",
		aliases=("tesla v100-pcie-32gb", "v100s")),
	_g("TITAN-V", "nvidia", "volta", 12, 652.8, 80, 250, 14.0, None, None, 110.0, None, 32, "pcie3",
		confidence="secondary", aliases=("titan v",)),
	_g("T4", "nvidia", "turing", 16, 320, 40, 70, 8.1, None, None, 65.0, None, 32, "pcie3",
		notes="No bf16 tensor path; the datasheet says 300 GB/s, the whitepaper 320.", aliases=("tesla t4",), usable_gib=14.74),
	_g("RTX-2080", "nvidia", "turing", 8, 448, 46, 215, 10.0, None, None, 40.3, None, 32, "pcie3",
		aliases=("geforce rtx 2080",)),
	_g("RTX-2080-Ti", "nvidia", "turing", 11, 616, 68, 250, 13.4, None, None, 53.8, None, 32, "pcie3",
		aliases=("geforce rtx 2080 ti",)),
	_g("Quadro-RTX-8000", "nvidia", "turing", 48, 672, 72, 295, 16.3, None, None, 130.5, None, 32, "pcie3",
		aliases=("quadro rtx 8000",)),
	# --- Ampere
	_g("A16", "nvidia", "ampere", 16, 200, 10, 62, 4.5, 9.0, 17.9, 17.9, None, 64, "pcie4",
		confidence="derived", notes="Per GPU of the 4-GPU board; bf16 not listed, taken equal to fp16; TDP is a quarter of the board.",
		aliases=("nvidia a16",)),
	_g("RTX-A4000", "nvidia", "ampere", 16, 448, 48, 140, 19.2, 38.4, 76.7, 76.7, None, 64, "pcie4",
		confidence="derived", aliases=("rtx a4000",)),
	_g("RTX-A5000", "nvidia", "ampere", 24, 768, 64, 230, 27.8, 55.6, 111.1, 111.1, None, 64, "pcie4",
		confidence="derived", aliases=("rtx a5000",)),
	_g("RTX-A6000", "nvidia", "ampere", 48, 768, 84, 300, 38.7, 77.4, 154.8, 154.8, None, 64, "pcie4",
		aliases=("rtx a6000",)),
	_g("A40", "nvidia", "ampere", 48, 696, 84, 300, 37.4, 74.8, 149.7, 149.7, None, 64, "pcie4", aliases=("nvidia a40",)),
	_g("A10", "nvidia", "ampere", 24, 600, 72, 150, 31.2, 62.5, 125.0, 125.0, None, 64, "pcie4",
		notes="AWS A10G maps here at slightly lower clocks.", aliases=("nvidia a10", "a10g")),
	_g("RTX-3090", "nvidia", "ampere", 24, 936, 82, 350, 35.6, 35.6, 71.0, 71.0, None, 64, "pcie4",
		aliases=("geforce rtx 3090",)),
	_g("A100-PCIe-40GB", "nvidia", "ampere", 40, 1555, 108, 250, 19.5, 156.0, 312.0, 312.0, None, 64, "pcie4",
		mig_slices=7, aliases=("a100-pcie-40gb",), usable_gib=39.39),
	_g("A100-SXM-40GB", "nvidia", "ampere", 40, 1555, 108, 400, 19.5, 156.0, 312.0, 312.0, None, 600, "nvlink",
		mig_slices=7, aliases=("a100-sxm4-40gb", "a100-40gb"), usable_gib=39.39),
	_g("A100-PCIe-80GB", "nvidia", "ampere", 80, 1935, 108, 300, 19.5, 156.0, 312.0, 312.0, None, 64, "pcie4",
		mig_slices=7, aliases=("a100 80gb pcie", "a100-pcie-80gb"), usable_gib=79.15),
	_g("A100-SXM-80GB", "nvidia", "ampere", 80, 2039, 108, 400, 19.5, 156.0, 312.0, 312.0, None, 600, "nvlink",
		mig_slices=7, aliases=("a100-sxm4-80gb", "a100-80gb"), usable_gib=79.15),
	# --- Ada
	# usable_gib measured: torch total_memory 23,689,363,456 bytes on Modal (Oct 27 smoke run); the same run reached
	# 59.8 TFLOP/s on an 8192^2 bf16 matmul, 49% of the dense bf16 peak
	_g("L4", "nvidia", "ada", 24, 300, 58, 72, 30.3, 60.0, 121.0, 121.0, 242.5, 64, "pcie4", confidence="derived",
		aliases=("nvidia l4",), usable_gib=23_689_363_456 / 2**30),
	_g("L40", "nvidia", "ada", 48, 864, 142, 300, 90.5, 90.5, 181.05, 181.05, 362.0, 64, "pcie4", aliases=("nvidia l40",)),
	_g("L40S", "nvidia", "ada", 48, 864, 142, 350, 91.6, 183.0, 362.05, 362.05, 733.0, 64, "pcie4", aliases=("nvidia l40s",)),
	_g("RTX-4090", "nvidia", "ada", 24, 1008, 128, 450, 82.6, 82.6, 165.2, 165.2, 330.3, 64, "pcie4",
		aliases=("geforce rtx 4090",), usable_gib=23.65),
	# --- GeForce RTX 30 and 40, desktop and laptop (D, Nov 3: a student's own card gets memory and P(OOM)). Memory,
	# bus and SMs from NVIDIA's spec pages; peaks derived as SMs x 128 FP32 lanes x 2 x boost clock, with the GeForce
	# whitepaper ratios the RTX-3090 and RTX-4090 rows follow (TF32 = FP32; FP16 and BF16 with FP32 accumulate = 2x;
	# Ada FP8 = 4x). Laptop peaks are at the top of NVIDIA's boost range (the highest TGP), so a thinner laptop is
	# slower; laptop bandwidth is at the usual memory speed, which the laptop maker sets.
	_g("RTX-3060", "nvidia", "ampere", 12, 360, 28, 170, 12.7, 12.7, 25.4, 25.4, None, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 1.777 GHz boost clock.", aliases=("geforce rtx 3060",)),
	_g("RTX-3060-8GB", "nvidia", "ampere", 8, 240, 28, 170, 12.7, 12.7, 25.4, 25.4, None, 64, "pcie4", confidence="derived",
		notes="The 8 GB desktop 3060 (128-bit, 15 Gbps); same PyTorch name as the 12 GB card, told apart by memory.",
		aliases=("geforce rtx 3060 8gb",)),
	_g("RTX-3060-Ti", "nvidia", "ampere", 8, 448, 38, 200, 16.2, 16.2, 32.4, 32.4, None, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 1.665 GHz boost clock.", aliases=("geforce rtx 3060 ti",)),
	_g("RTX-3070", "nvidia", "ampere", 8, 448, 46, 220, 20.3, 20.3, 40.6, 40.6, None, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 1.725 GHz boost clock.", aliases=("geforce rtx 3070",)),
	_g("RTX-3070-Ti", "nvidia", "ampere", 8, 608, 48, 290, 21.7, 21.7, 43.4, 43.4, None, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 1.770 GHz boost clock.", aliases=("geforce rtx 3070 ti",)),
	_g("RTX-3080-10GB", "nvidia", "ampere", 10, 760, 68, 320, 29.8, 29.8, 59.6, 59.6, None, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 1.710 GHz boost clock.", aliases=("geforce rtx 3080",)),
	_g("RTX-3080-12GB", "nvidia", "ampere", 12, 912, 70, 350, 30.6, 30.6, 61.2, 61.2, None, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 1.710 GHz boost clock.", aliases=("geforce rtx 3080 12gb",)),
	_g("RTX-3080-Ti", "nvidia", "ampere", 12, 912, 80, 350, 34.1, 34.1, 68.2, 68.2, None, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 1.665 GHz boost clock.", aliases=("geforce rtx 3080 ti",)),
	_g("RTX-3090-Ti", "nvidia", "ampere", 24, 1008, 84, 450, 40.0, 40.0, 80.0, 80.0, None, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 1.860 GHz boost clock.", aliases=("geforce rtx 3090 ti",)),
	_g("RTX-4060", "nvidia", "ada", 8, 272, 24, 115, 15.1, 15.1, 30.2, 30.2, 60.4, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 2.460 GHz boost clock.", aliases=("geforce rtx 4060",)),
	_g("RTX-4060-Ti-8GB", "nvidia", "ada", 8, 288, 34, 160, 22.1, 22.1, 44.2, 44.2, 88.4, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 2.535 GHz boost clock.", aliases=("geforce rtx 4060 ti",)),
	_g("RTX-4060-Ti-16GB", "nvidia", "ada", 16, 288, 34, 165, 22.1, 22.1, 44.2, 44.2, 88.4, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 2.535 GHz boost clock.", aliases=("geforce rtx 4060 ti 16gb",)),
	_g("RTX-4070", "nvidia", "ada", 12, 504, 46, 200, 29.1, 29.1, 58.2, 58.2, 116.4, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 2.475 GHz boost clock.", aliases=("geforce rtx 4070",)),
	_g("RTX-4070-Super", "nvidia", "ada", 12, 504, 56, 220, 35.5, 35.5, 71.0, 71.0, 142.0, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 2.475 GHz boost clock.", aliases=("geforce rtx 4070 super",)),
	_g("RTX-4070-Ti", "nvidia", "ada", 12, 504, 60, 285, 40.1, 40.1, 80.2, 80.2, 160.4, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 2.610 GHz boost clock.", aliases=("geforce rtx 4070 ti",)),
	_g("RTX-4070-Ti-Super", "nvidia", "ada", 16, 672, 66, 285, 44.1, 44.1, 88.2, 88.2, 176.4, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 2.610 GHz boost clock.", aliases=("geforce rtx 4070 ti super",)),
	_g("RTX-4080", "nvidia", "ada", 16, 716.8, 76, 320, 48.7, 48.7, 97.4, 97.4, 194.8, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 2.505 GHz boost clock.", aliases=("geforce rtx 4080",)),
	_g("RTX-4080-Super", "nvidia", "ada", 16, 736, 80, 320, 52.2, 52.2, 104.4, 104.4, 208.8, 64, "pcie4", confidence="derived",
		notes="Peaks derived at the 2.550 GHz boost clock.", aliases=("geforce rtx 4080 super",)),
	_g("RTX-3050-Laptop", "nvidia", "ampere", 4, 192, 16, 80, 7.1, 7.1, 14.2, 14.2, None, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 1.740 GHz (80 W TGP); bandwidth at 12 Gbps on 128-bit.", aliases=("geforce rtx 3050 laptop gpu",)),
	_g("RTX-3050-Ti-Laptop", "nvidia", "ampere", 4, 192, 20, 80, 8.7, 8.7, 17.4, 17.4, None, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 1.695 GHz (80 W TGP); bandwidth at 12 Gbps on 128-bit.", aliases=("geforce rtx 3050 ti laptop gpu",)),
	_g("RTX-3060-Laptop", "nvidia", "ampere", 6, 336, 30, 115, 13.1, 13.1, 26.2, 26.2, None, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 1.703 GHz (115 W TGP); bandwidth at 14 Gbps on 192-bit.", aliases=("geforce rtx 3060 laptop gpu",)),
	_g("RTX-3070-Laptop", "nvidia", "ampere", 8, 448, 40, 125, 16.6, 16.6, 33.2, 33.2, None, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 1.620 GHz (125 W TGP); bandwidth at 14 Gbps on 256-bit.", aliases=("geforce rtx 3070 laptop gpu",)),
	_g("RTX-3070-Ti-Laptop", "nvidia", "ampere", 8, 448, 46, 125, 17.5, 17.5, 35.0, 35.0, None, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 1.485 GHz (125 W TGP); bandwidth at 14 Gbps on 256-bit.", aliases=("geforce rtx 3070 ti laptop gpu",)),
	_g("RTX-3080-Laptop-8GB", "nvidia", "ampere", 8, 448, 48, 150, 21.0, 21.0, 42.0, 42.0, None, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 1.710 GHz (150 W TGP); bandwidth at 14 Gbps on 256-bit.", aliases=("geforce rtx 3080 laptop gpu",)),
	_g("RTX-3080-Laptop-16GB", "nvidia", "ampere", 16, 448, 48, 150, 21.0, 21.0, 42.0, 42.0, None, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 1.710 GHz (150 W TGP); bandwidth at 14 Gbps on 256-bit.", aliases=("geforce rtx 3080 laptop gpu 16gb",)),
	_g("RTX-3080-Ti-Laptop", "nvidia", "ampere", 16, 512, 58, 150, 23.6, 23.6, 47.2, 47.2, None, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 1.590 GHz (150 W TGP); bandwidth at 16 Gbps on 256-bit.", aliases=("geforce rtx 3080 ti laptop gpu",)),
	_g("RTX-4050-Laptop", "nvidia", "ada", 6, 192, 20, 115, 12.1, 12.1, 24.2, 24.2, 48.4, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 2.370 GHz (115 W TGP); bandwidth at 16 Gbps on 96-bit.", aliases=("geforce rtx 4050 laptop gpu",)),
	_g("RTX-4060-Laptop", "nvidia", "ada", 8, 256, 24, 115, 14.6, 14.6, 29.2, 29.2, 58.4, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 2.370 GHz (115 W TGP); bandwidth at 16 Gbps on 128-bit.", aliases=("geforce rtx 4060 laptop gpu",)),
	_g("RTX-4070-Laptop", "nvidia", "ada", 8, 256, 36, 115, 20.0, 20.0, 40.0, 40.0, 80.0, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 2.175 GHz (115 W TGP); bandwidth at 16 Gbps on 128-bit.", aliases=("geforce rtx 4070 laptop gpu",)),
	_g("RTX-4080-Laptop", "nvidia", "ada", 12, 432, 58, 150, 33.9, 33.9, 67.8, 67.8, 135.6, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 2.280 GHz (150 W TGP); bandwidth at 18 Gbps on 192-bit.", aliases=("geforce rtx 4080 laptop gpu",)),
	_g("RTX-4090-Laptop", "nvidia", "ada", 16, 576, 76, 150, 39.7, 39.7, 79.4, 79.4, 158.8, 64, "pcie4", confidence="derived",
		notes="Laptop: peaks at the top boost clock, 2.040 GHz (150 W TGP); bandwidth at 18 Gbps on 256-bit.", aliases=("geforce rtx 4090 laptop gpu",)),
	# --- Hopper
	_g("H100-PCIe-80GB", "nvidia", "hopper", 80, 2000, 114, 350, 51.2, 378.0, 756.0, 756.0, 1513.0, 128, "pcie5",
		mig_slices=7, aliases=("h100 pcie",)),
	_g("H100-SXM-80GB", "nvidia", "hopper", 80, 3350, 132, 700, 67.0, 494.7, 989.4, 989.4, 1978.9, 900, "nvlink",
		confidence="derived", mig_slices=7, notes="All Modal H100s are SXM.", aliases=("h100 80gb hbm3", "h100-sxm", "h100"), usable_gib=79.18),
	_g("H100-NVL-94GB", "nvidia", "hopper", 94, 3900, 132, 400, 60.0, 417.5, 835.5, 835.5, 1670.5, 600, "nvlink",
		confidence="derived", mig_slices=7, aliases=("h100 nvl",)),
	_g("H200-SXM-141GB", "nvidia", "hopper", 141, 4800, 132, 700, 67.0, 494.7, 989.4, 989.4, 1978.9, 900, "nvlink",
		confidence="derived", mig_slices=7, aliases=("h200",)),
	_g("H200-NVL-141GB", "nvidia", "hopper", 141, 4800, 132, 600, 60.0, 417.5, 835.5, 835.5, 1670.5, 900, "nvlink",
		confidence="derived", mig_slices=7, aliases=("h200 nvl",)),
	_g("GH200-96GB", "nvidia", "hopper", 96, 4000, 132, 700, 67.0, 494.0, 990.0, 990.0, 1979.0, 900, "nvlink",
		mig_slices=7, notes="TDP is for the whole superchip module, 450-1000 W.", aliases=("gh200",)),
	_g("GH200-144GB", "nvidia", "hopper", 144, 4900, 132, 1000, 67.0, 494.0, 990.0, 990.0, 1979.0, 900, "nvlink",
		mig_slices=7, aliases=("gh200 144gb",)),
	# --- Blackwell
	_g("RTX-5090", "nvidia", "blackwell", 32, 1792, 170, 575, 104.8, 104.8, 209.5, 209.5, 419.0, 128, "pcie5",
		aliases=("geforce rtx 5090",)),
	_g("RTX-PRO-6000-96GB", "nvidia", "blackwell", 96, 1597, 188, 600, 120.0, 234.0, 500.0, 500.0, 1000.0, 128, "pcie5",
		confidence="secondary", mig_slices=4, notes="Server Edition; tensor rates from flopper.io, possibly a few percent high.",
		aliases=("rtx pro 6000 blackwell", "rtx pro 6000", "rtx-pro-6000")),
	_g("B200-SXM-180GB", "nvidia", "blackwell", 180, 7700, 148, 1000, 75.0, 1100.0, 2250.0, 2250.0, 4500.0, 1800, "nvlink",
		mig_slices=7, notes="HGX per-GPU figures; Modal lists 192 GB.", aliases=("b200",)),
	_g("B300-SXM-288GB", "nvidia", "blackwell", 288, 8000, 160, 1400, 75.0, 1125.0, 2250.0, 2250.0, 4500.0, 1800, "nvlink",
		confidence="derived", mig_slices=7, notes="HGX totals / 8, halved from sparse; TDP from a secondary source.",
		aliases=("b300",)),
	# --- AMD CDNA (fp32 is the matrix-core rate; PyTorch leaves xf32 off, so tf32 is None)
	_g("MI100", "amd", "cdna1", 32, 1228.8, 120, 300, 46.1, None, 92.3, 184.6, None, 276, "xgmi",
		aliases=("instinct mi100", "mi100", "gfx908")),
	_g("MI210", "amd", "cdna2", 64, 1600, 104, 300, 45.3, None, 181.0, 181.0, None, 300, "xgmi",
		aliases=("instinct mi210", "mi210")),
	_g("MI250-GCD", "amd", "cdna2", 64, 1638, 104, 250, 45.3, None, 181.0, 181.0, None, 400, "xgmi",
		confidence="derived", notes="One of the two GCDs of an MI250; PyTorch sees each as a device.",
		aliases=("instinct mi250", "mi250")),
	_g("MI250X-GCD", "amd", "cdna2", 64, 1638, 110, 280, 47.9, None, 191.5, 191.5, None, 400, "xgmi",
		confidence="derived", notes="One of the two GCDs of an MI250X; PyTorch sees each as a device.",
		aliases=("instinct mi250x", "mi250x")),
	_g("MI300X", "amd", "cdna3", 192, 5300, 304, 750, 163.4, None, 1307.4, 1307.4, 2614.9, 896, "xgmi",
		notes="SPX mode (one device); xf32 653.7 TFLOPS exists but PyTorch leaves it off by default.",
		aliases=("instinct mi300x", "mi300x")),
	_g("MI325X", "amd", "cdna3", 256, 6000, 304, 1000, 163.4, None, 1307.4, 1307.4, 2614.9, 896, "xgmi",
		aliases=("instinct mi325x", "mi325x")),
	_g("MI355X", "amd", "cdna4", 288, 8000, 256, 1400, 157.3, None, 2516.6, 2516.6, 5033.2, 1075, "xgmi",
		confidence="derived", aliases=("instinct mi355x", "mi355x", "gfx950")),
)
# fmt: on

GPUS: dict[str, GpuSpec] = {s.name: s for s in _SPECS}


def list_gpus(vendor: Vendor | None = None) -> list[str]:
	return [s.name for s in _SPECS if vendor is None or s.vendor == vendor]


# ---------------------------------------------------------------- name resolution

# "A100-SXM4-80GB MIG 3g.40gb" from PyTorch, "A100-SXM-80GB-MIG-3g.40gb" from get_gpu
_MIG = re.compile(r"\bmig[\s-]{1,3}(\d{1,2})g\.(\d{1,4})gb\b")
_SIZE = re.compile(r"(?<!\d)(\d{1,4})\s?gb?\b")  # bounded and anchored: linear on any input
MAX_NAME_LEN = (
	128  # derived: longer than any device or Modal name; visitor text never reaches a regex unbounded
)
# memory share of a MIG profile: 8 memory slices per card; a 1g profile holds 1 or 2 of them
_MIG_MEM_EIGHTHS = {
	1: (1, 2),
	2: (2,),
	3: (4,),
	4: (4,),
	7: (8,),
}  # cited: NVIDIA MIG user guide, 8 memory slices per card, a 1g profile holds 1 or 2


def _by_memory(small: str, large: str, cut_gib: float):
	return lambda mem, _cu: large if (mem or 0) > cut_gib else small


def _mi250(_mem, cu):
	return "MI250X-GCD" if (cu or 110) >= 110 else "MI250-GCD"


def _gfx942(mem, _cu):
	return "MI325X" if (mem or 192) > 240 else "MI300X"


# Most specific first: (regex on the cleaned name, table name or chooser(memory_gib, compute_units)).
_PATTERNS: list[tuple[re.Pattern[str], object]] = [
	(re.compile(p), t)
	for p, t in [
		(r"gh200", _by_memory("GH200-96GB", "GH200-144GB", 120)),
		(r"h200\s*nvl", "H200-NVL-141GB"),
		(r"h200", "H200-SXM-141GB"),
		(r"h100\s*nvl", "H100-NVL-94GB"),
		(r"h100[\s-]*pcie", "H100-PCIe-80GB"),
		(r"h100", "H100-SXM-80GB"),
		(r"b300", "B300-SXM-288GB"),
		(r"b200", "B200-SXM-180GB"),
		(r"rtx[\s-]*pro[\s-]*6000", "RTX-PRO-6000-96GB"),
		(r"rtx\s*5090(?!\s*laptop)", "RTX-5090"),  # review, Nov 3: no 5090 Laptop row; it reads as unlisted
		(r"rtx\s*4090\s*laptop", "RTX-4090-Laptop"),
		(r"rtx\s*4080\s*laptop", "RTX-4080-Laptop"),
		(r"rtx\s*4070\s*laptop", "RTX-4070-Laptop"),
		(r"rtx\s*4060\s*laptop", "RTX-4060-Laptop"),
		(r"rtx\s*4050\s*laptop", "RTX-4050-Laptop"),
		(r"rtx\s*3080\s*ti\s*laptop", "RTX-3080-Ti-Laptop"),
		(r"rtx\s*3080\s*laptop", _by_memory("RTX-3080-Laptop-8GB", "RTX-3080-Laptop-16GB", 10)),
		(r"rtx\s*3070\s*ti\s*laptop", "RTX-3070-Ti-Laptop"),
		(r"rtx\s*3070\s*laptop", "RTX-3070-Laptop"),
		(r"rtx\s*3060\s*laptop", "RTX-3060-Laptop"),
		(r"rtx\s*3050\s*ti\s*laptop", "RTX-3050-Ti-Laptop"),
		(r"rtx\s*3050\s*laptop", "RTX-3050-Laptop"),
		(r"rtx\s*4090", "RTX-4090"),
		(r"rtx\s*4080\s*super", "RTX-4080-Super"),
		(r"rtx\s*4080", "RTX-4080"),
		(r"rtx\s*4070\s*ti\s*super", "RTX-4070-Ti-Super"),
		(r"rtx\s*4070\s*ti", "RTX-4070-Ti"),
		(r"rtx\s*4070\s*super", "RTX-4070-Super"),
		(r"rtx\s*4070", "RTX-4070"),
		(r"rtx\s*4060\s*ti", _by_memory("RTX-4060-Ti-8GB", "RTX-4060-Ti-16GB", 12)),
		(r"rtx\s*4060", "RTX-4060"),
		(r"rtx\s*3090\s*ti", "RTX-3090-Ti"),
		(r"rtx\s*3090", "RTX-3090"),
		(r"rtx\s*3080\s*ti", "RTX-3080-Ti"),
		(r"rtx\s*3080", _by_memory("RTX-3080-10GB", "RTX-3080-12GB", 11)),
		(r"rtx\s*3070\s*ti", "RTX-3070-Ti"),
		(r"rtx\s*3070", "RTX-3070"),
		(r"rtx\s*3060\s*ti", "RTX-3060-Ti"),
		(r"rtx\s*3060", _by_memory("RTX-3060-8GB", "RTX-3060", 10)),
		(r"rtx\s*2080\s*ti", "RTX-2080-Ti"),
		(r"rtx\s*2080", "RTX-2080"),
		(r"rtx\s*8000", "Quadro-RTX-8000"),
		(r"rtx\s*a6000", "RTX-A6000"),
		(r"rtx\s*a5000", "RTX-A5000"),
		(r"rtx\s*a4000", "RTX-A4000"),
		(r"a100.*pci", _by_memory("A100-PCIe-40GB", "A100-PCIe-80GB", 60)),
		(r"a100", _by_memory("A100-SXM-40GB", "A100-SXM-80GB", 60)),
		(r"\ba40\b", "A40"),
		(r"\ba16\b", "A16"),
		(r"\ba10g?\b", "A10"),
		(r"l40s", "L40S"),
		(r"\bl40\b", "L40"),
		(r"\bl4\b", "L4"),
		(r"\bt4\b", "T4"),
		(r"v100.*sxm", _by_memory("V100-SXM2-16GB", "V100-SXM2-32GB", 20)),
		(r"v100", _by_memory("V100-PCIe-16GB", "V100-PCIe-32GB", 20)),
		(r"titan\s*v\b", "TITAN-V"),
		(r"titan\s*x\s*\(pascal\)|titan\s*xp", "TITAN-X-Pascal"),
		(r"titan\s*x\b", "TITAN-X-Maxwell"),
		(r"1080\s*ti", "GTX-1080-Ti"),
		(r"p100", "P100-16GB"),
		(r"m40", "M40-24GB"),
		(r"mi355x|gfx950", "MI355X"),
		(r"mi325x", "MI325X"),
		(r"mi300x|gfx942", _gfx942),
		(r"mi250x?|gfx90a", _mi250),
		(r"mi210", "MI210"),
		(r"mi100|gfx908", "MI100"),
	]
]


def _clean(name: str) -> str:
	s = name.lower().strip().rstrip("!")
	s = re.sub(r"^(nvidia|amd|tesla)\s+", "", s)
	s = re.sub(r"\s+vf$", "", s)  # SR-IOV virtual function suffix on cloud MI300X
	return s.strip()


# ---------------------------------------------------------------- GPUs the table doesn't list

UNLISTED = "unlisted:"


def unlisted_name(raw: str, memory_bytes: int, compute_units: int | None = None) -> str:
	"""A name that carries what the probe measured about a GPU the table lacks (Review, Oct 27)."""
	clean = re.sub(r"[^A-Za-z0-9 ._()-]", "", raw)[:48].strip() or "gpu"
	return f"{UNLISTED}{clean}@{int(memory_bytes)}:{int(compute_units or 0)}"


def is_unlisted(name: str | None) -> bool:
	return bool(name) and name.startswith(UNLISTED)


def _unlisted_spec(name: str) -> GpuSpec:
	"""Capacity from the probe's own report; no peaks, bandwidth or link, so nothing is scaled from it."""
	m = re.fullmatch(r"unlisted:(.{1,48})@(\d{1,16}):(\d{1,6})", name)
	if not m:
		raise KeyError(f"malformed unlisted GPU name {name!r}")
	raw, mem, cu = m.group(1), int(m.group(2)), int(m.group(3))
	vendor = "amd" if re.search(r"amd|instinct|radeon|gfx", raw, re.I) else "nvidia"
	gib = mem / GIB
	return GpuSpec(
		name, vendor, "unknown", gib, 0.0, cu, 0, 0.0, None, None, None, None, 0.0, "unknown",  # type: ignore[arg-type]
		confidence="guess",
		notes="Not in the table: capacity is what the probe's PyTorch reported; peaks, bandwidth and link unknown.",
		usable_gib=gib,
	)  # fmt: skip


def resolve(name: str, memory_bytes: int | None = None, compute_units: int | None = None) -> str:
	"""Table name for a PyTorch device name, ROCm arch, MIG slice or Modal string.

	memory_bytes and compute_units (torch's total_memory and multi_processor_count)
	split names PyTorch reports the same way, e.g. "AMD Instinct MI250X / MI250".
	"""
	if name in GPUS or is_unlisted(name):
		return name
	if len(name) > MAX_NAME_LEN:
		raise KeyError(f"unknown GPU (name longer than {MAX_NAME_LEN} characters)")
	cleaned = _clean(name)
	mig = _MIG.search(cleaned)
	if mig and (int(mig.group(1)) <= 0 or int(mig.group(2)) <= 0):
		# "1g.0gb", "0g.10gb": no real profile, and a zero size would divide by zero below
		raise KeyError(f"no MIG profile {mig.group(1)}g.{mig.group(2)}gb in {name!r}")
	s = _MIG.sub("", cleaned).strip(" -_")
	for spec in _SPECS:
		if s == spec.name.lower():
			return spec.name
	# a size in the name ("a100-80gb", "A100-SXM4-80GB"); ignore GH200's 480GB, which counts CPU memory
	sizes = [int(m) for m in _SIZE.findall(s) if int(m) <= 300]
	if mig and not sizes:
		# torch reports the slice's memory, not the card's: try each card size the profile allows
		g, slice_gb = int(mig.group(1)), int(mig.group(2))
		best = None
		for eighths in _MIG_MEM_EIGHTHS.get(g, (8 * g / 7,)):
			card_gb = slice_gb * 8 / eighths
			hit = _match(s, card_gb * GB / GIB, compute_units)
			if hit is not None:
				err = abs(GPUS[hit].memory_gib * GIB / GB - card_gb) / card_gb
				if (
					best is None or err < best[0]
				):  # ties go to the first: 1g.10gb is the 80 GB card's smallest
					best = (err, hit)
		if best:
			return best[1]
	elif mig:
		hit = _match(s, float(sizes[0]), compute_units)  # the card's size is in the name
		if hit:
			return hit
	else:
		mem_gib = memory_bytes / GIB if memory_bytes else (float(sizes[0]) if sizes else None)
		hit = _match(s, mem_gib, compute_units)
		if hit:
			return hit
	for spec in _SPECS:
		if s in spec.aliases:
			return spec.name
	raise KeyError(f"unknown GPU {name!r}")


def _match(s: str, mem_gib: float | None, compute_units: int | None) -> str | None:
	for pat, target in _PATTERNS:
		if pat.search(s):
			return target(mem_gib, compute_units) if callable(target) else target
	return None


def mig_profile(name: str) -> tuple[int, int] | None:
	"""(compute slices, memory GB) for a MIG device name such as '... MIG 3g.40gb'."""
	if len(name) > MAX_NAME_LEN:
		return None
	m = _MIG.search(name.lower())
	return (int(m.group(1)), int(m.group(2))) if m else None


def get_gpu(name: str, memory_bytes: int | None = None, compute_units: int | None = None) -> GpuSpec:
	"""The spec for a device; a MIG slice comes back scaled to its share of the GPU; an unlisted one, a fallback."""
	if is_unlisted(name):
		return _unlisted_spec(name)
	spec = GPUS[resolve(name, memory_bytes, compute_units)]
	mig = mig_profile(name)
	if mig is None or not spec.mig_slices:
		return spec
	g, mem_gb = mig
	frac = min(1.0, g / spec.mig_slices)  # compute slices
	mem_frac = min(1.0, mem_gb * GB / (spec.memory_gib * GIB))  # memory slices carry the bandwidth

	def sc(v):
		return None if v is None else v * frac

	return replace(
		spec,
		name=f"{spec.name}-MIG-{g}g.{mem_gb}gb",
		memory_gib=mem_gb * GB / GIB,
		usable_gib=mem_gb
		* GB
		/ GIB
		* 0.97,  # guess: MIG slice overhead; replaced by a MIG probe's total_memory
		bandwidth_gbps=spec.bandwidth_gbps * mem_frac,
		compute_units=round(spec.compute_units * frac),
		fp32=spec.fp32 * frac,
		tf32=sc(spec.tf32),
		bf16=sc(spec.bf16),
		fp16=sc(spec.fp16),
		fp8=sc(spec.fp8),
		link_gbps=0.0,
		notes=f"MIG slice {g}/{spec.mig_slices} of {spec.name}; compute scaled by compute slices, bandwidth by memory share.",
	)


def peak_flops(gpu_name: str | None, precision: str, memory_bytes: int | None = None) -> float | None:
	"""Dense peak FLOP/s for MFU; None when the GPU or precision is unknown.

	Drop-in for recorder/gpus.py: precision is bf16, fp16, fp32, tf32 or fp8.
	"""
	if not gpu_name:
		return None
	try:
		spec = get_gpu(gpu_name, memory_bytes)
	except KeyError:
		return None
	p = precision.lower()
	if p == "fp32":
		return spec.peak_flops("fp32", allow_tf32=False)
	if p == "tf32":
		return spec.peak_flops("tf32") or spec.peak_flops("fp32", allow_tf32=False)
	try:
		return spec.peak_flops(p)
	except ValueError:
		return None


# ---------------------------------------------------------------- prices


@dataclass(frozen=True)
class GpuPrice:
	gpu: str
	provider: str  # same ids as Env.provider: modal, amd_devcloud, ...
	usd_per_hour: float
	as_of: str
	source: str
	preemptible: bool = False


# One table (Chiya, Oct 26): the rows live in engine/prices; this is the GPU view the probes and corpus use.
_PRICES: tuple[GpuPrice, ...] = tuple(
	GpuPrice(r.gpu, r.provider, r.usd_per_hour, r.as_of, r.source, r.preemptible)
	for r in _prices.GPU_RATES
	if r.kind == "on_demand"
)

# Modal's GPU strings for the table names it offers. "!" stops Modal upgrading H100 to H200.
MODAL_GPU = {
	"T4": "T4",
	"L4": "L4",
	"A10": "A10",
	"L40S": "L40S",
	"A100-SXM-40GB": "A100-40GB",
	"A100-SXM-80GB": "A100-80GB",
	"RTX-PRO-6000-96GB": "RTX-PRO-6000",
	"H100-SXM-80GB": "H100!",
	"H200-SXM-141GB": "H200",
	"B200-SXM-180GB": "B200",
	"B300-SXM-288GB": "B300",
}


def price(gpu: str, provider: str = "modal") -> GpuPrice | None:
	key = resolve(gpu)
	for p in _PRICES:
		if p.gpu == key and p.provider == provider:
			return p
	return None


def prices(provider: str | None = None) -> list[GpuPrice]:
	return [p for p in _PRICES if provider is None or p.provider == provider]


def on_demand_reference(gpu: str) -> GpuPrice | None:
	"""Cheapest listed rate for this GPU: the grey dollar figure beside GPU-hours on own clusters."""
	key = resolve(gpu)
	rows = [p for p in _PRICES if p.gpu == key]
	return min(rows, key=lambda p: p.usd_per_hour) if rows else None

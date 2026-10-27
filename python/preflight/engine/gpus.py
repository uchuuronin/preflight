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

TFLOP = 1e12  # derived: unit
GB = 1e9  # derived: unit
GIB = 2**30  # derived: unit

Confidence = Literal["datasheet", "derived", "secondary"]

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
	_g("L4", "nvidia", "ada", 24, 300, 58, 72, 30.3, 60.0, 121.0, 121.0, 242.5, 64, "pcie4", confidence="derived",
		aliases=("nvidia l4",), usable_gib=21.45),
	_g("L40", "nvidia", "ada", 48, 864, 142, 300, 90.5, 90.5, 181.05, 181.05, 362.0, 64, "pcie4", aliases=("nvidia l40",)),
	_g("L40S", "nvidia", "ada", 48, 864, 142, 350, 91.6, 183.0, 362.05, 362.05, 733.0, 64, "pcie4", aliases=("nvidia l40s",)),
	_g("RTX-4090", "nvidia", "ada", 24, 1008, 128, 450, 82.6, 82.6, 165.2, 165.2, 330.3, 64, "pcie4",
		aliases=("geforce rtx 4090",), usable_gib=23.65),
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
		(r"rtx\s*5090", "RTX-5090"),
		(r"rtx\s*4090", "RTX-4090"),
		(r"rtx\s*3090", "RTX-3090"),
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


def resolve(name: str, memory_bytes: int | None = None, compute_units: int | None = None) -> str:
	"""Table name for a PyTorch device name, ROCm arch, MIG slice or Modal string.

	memory_bytes and compute_units (torch's total_memory and multi_processor_count)
	split names PyTorch reports the same way, e.g. "AMD Instinct MI250X / MI250".
	"""
	if name in GPUS:
		return name
	if len(name) > MAX_NAME_LEN:
		raise KeyError(f"unknown GPU (name longer than {MAX_NAME_LEN} characters)")
	cleaned = _clean(name)
	mig = _MIG.search(cleaned)
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
	"""The spec for a device; a MIG slice comes back scaled to its share of the GPU."""
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


_MODAL = (
	("T4", 0.59),
	("L4", 0.80),
	("A10", 1.10),
	("L40S", 1.95),
	("A100-SXM-80GB", 2.50),
	("RTX-PRO-6000-96GB", 3.03),
	("H100-SXM-80GB", 3.95),
	("H200-SXM-141GB", 4.54),
	("B200-SXM-180GB", 6.25),
	("B300-SXM-288GB", 7.10),
)
_PRICES: tuple[GpuPrice, ...] = (
	*(
		GpuPrice(
			g, "modal", p, "2026-07", "Modal list rate, per-second billing, base region", preemptible=True
		)
		for g, p in _MODAL
	),
	GpuPrice(
		"A100-SXM-40GB",
		"modal",
		2.10,
		"2026-10",
		"computeprices.com, updated Oct 26 2026 (lists it as PCIe; single-GPU specs are the same)",
		preemptible=True,
	),
	GpuPrice("MI300X", "amd_devcloud", 2.59, "2026-10", "Preflight contract, gap 8"),
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

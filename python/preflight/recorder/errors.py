"""Classify probe failures into stable codes.

The code drives three things: the UI's message and hint (t('error.<code>')),
whether the queue retries (retryable), and which finding the record gets.
Matching is by exception type first, then by message, most specific first.
"""

import re
from dataclasses import dataclass


@dataclass
class Classified:
	code: str
	status: str  # Probe.status: oom, error, timeout
	message: str
	retryable: bool


def _has(text: str, *needles: str) -> bool:
	t = text.lower()
	return any(n.lower() in t for n in needles)


def classify(exc: BaseException, *, steps: int, in_eval: bool) -> Classified:
	name = type(exc).__name__
	msg = str(exc)
	both = f"{name}: {msg}"

	if name == "OutOfMemoryError" or _has(msg, "out of memory", "CUBLAS_STATUS_ALLOC_FAILED"):
		if in_eval:
			return Classified("oom.eval", "oom", "Ran out of GPU memory during the eval pass", False)
		return Classified(
			"oom",
			"oom",
			"Ran out of GPU memory" + (f" at step {steps + 1}" if steps else " before the first step"),
			False,
		)
	if isinstance(exc, ModuleNotFoundError) or name == "ModuleNotFoundError":
		m = re.search(r"No module named '([^']+)'", msg)
		pkg = m.group(1) if m else "a package"
		return Classified(
			"missing_package",
			"error",
			f"The script imports {pkg}, which is not installed in the probe image",
			False,
		)
	if isinstance(exc, ImportError) or name == "ImportError":
		# libraries that check optional dependencies themselves, e.g. Trainer asking for accelerate
		m = re.search(r"requires `?([A-Za-z0-9_.\-\[\]]+?)`?(?:[<>=!~][^\s`:]*)?`?[\s:,]", msg)
		if m or _has(msg, "pip install"):
			pkg = m.group(1) if m else "a package"
			# installed, at the wrong version ("pyarrow requires NumPy 2.0 or newer, found 1.26.4")
			v = re.search(
				r"requires \S+ ([0-9]+(?:\.[0-9a-z]+)*) or newer, found ([0-9]+(?:\.[0-9a-z]+)*)", msg
			)
			if v:
				return Classified("missing_package", "error", f"A library in the script needs {pkg} "
					f"{v.group(1)} or newer; the probe image has {v.group(2)}", False)  # fmt: skip
			return Classified(
				"missing_package",
				"error",
				f"A library in the script needs {pkg}, which is not installed in the probe image",
				False,
			)
	if _has(msg, "gated repo", "401 Client Error", "Access to model", "is restricted"):
		return Classified(
			"gated_model", "error", "The model is gated on Hugging Face; the probe has no access token", False
		)
	if _has(
		msg,
		"couldn't connect to",
		"Max retries exceeded",
		"ConnectionError",
		"Temporary failure in name resolution",
		"LocalEntryNotFoundError",
		"Connection refused",
		"offline mode is enabled",
	) or name in ("ConnectionError", "LocalEntryNotFoundError", "ConnectTimeout"):
		return Classified(
			"download_blocked",
			"error",
			"A download failed; the host may be outside the probe's network allowlist",
			True,
		)
	if isinstance(exc, FileNotFoundError) or name == "FileNotFoundError":
		m = re.search(r"No such file or directory: '([^']+)'", msg)
		where = f" ({m.group(1)})" if m else ""
		return Classified("file_not_found", "error", f"A file the script reads is missing{where}", False)
	if isinstance(exc, SyntaxError) or name in ("SyntaxError", "IndentationError", "TabError"):
		return Classified("syntax_error", "error", f"The script does not parse: {msg}", False)
	if _has(msg, "NCCL", "ProcessGroup", "c10d", "Gloo"):
		return Classified("nccl", "error", "Distributed communication failed", True)
	if _has(
		msg,
		"bf16 is not supported",
		"doesn't support bf16",
		"bfloat16 is not supported",
		"not implemented for 'BFloat16'",
		"FlashAttention only supports Ampere",
	):
		return Classified(
			"dtype_unsupported",
			"error",
			"The GPU does not support a precision or kernel the script asks for",
			False,
		)
	if _has(
		msg,
		"device-side assert",
		"illegal memory access",
		"CUDA error",
		"HIP error",
		"cuDNN error",
		"CUBLAS_STATUS",
	):
		return Classified("cuda_error", "error", f"A GPU kernel failed: {msg[:200]}", False)
	if _has(msg, "DataLoader worker", "shared memory", "bus error"):
		return Classified(
			"dataloader_worker",
			"error",
			"A data loader worker crashed (often too little shared memory)",
			False,
		)
	if _has(
		msg,
		"shapes cannot be multiplied",
		"size mismatch",
		"must match the size",
		"Expected input batch_size",
		"expected scalar type",
		"dimension out of range",
	):
		return Classified(
			"shape_mismatch", "error", f"Tensor shapes or types do not line up: {msg[:200]}", False
		)
	return Classified("user_error", "error", both[:500], False)


def classify_exit(
	*, returncode: int | None, timed_out: bool, steps: int, stage: str, timeout_s: float
) -> Classified:
	when = f"after step {steps}" if steps else f"before the first optimizer step (last stage: {stage})"
	if timed_out:
		return Classified("timeout", "timeout", f"Timed out after {timeout_s:.0f} s {when}", False)
	if returncode in (-9, 137):
		return Classified(
			"oom.host", "oom", f"The probe was killed, most likely out of host memory, {when}", False
		)
	if returncode is not None and returncode < 0:
		return Classified("killed", "error", f"The probe process died from signal {-returncode} {when}", True)
	return Classified(
		"script_exit", "error", f"The probe process exited with code {returncode} {when}", False
	)

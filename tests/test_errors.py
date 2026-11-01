"""Failure classification: every probe failure gets a stable code, a plain message and a retry decision."""

import pytest

from preflight.recorder.errors import classify, classify_exit


def exc(cls, msg):
	try:
		raise cls(msg)
	except cls as e:
		return e


class OutOfMemoryError(RuntimeError):
	pass


@pytest.mark.parametrize(
	("error", "code", "status", "retryable"),
	[
		(exc(OutOfMemoryError, "CUDA out of memory. Tried to allocate 2.00 GiB"), "oom", "oom", False),
		(exc(RuntimeError, "HIP out of memory. Tried to allocate 1.5 GiB"), "oom", "oom", False),
		(exc(ModuleNotFoundError, "No module named 'flash_attn'"), "missing_package", "error", False),
		(
			exc(OSError, "We couldn't connect to 'https://huggingface.co' to load this file"),
			"download_blocked",
			"error",
			True,
		),
		(
			exc(
				OSError,
				"You are trying to access a gated repo. Make sure to have access to it at https://huggingface.co/meta-llama/x. 401 Client Error",
			),
			"gated_model",
			"error",
			False,
		),
		(
			exc(FileNotFoundError, "[Errno 2] No such file or directory: 'data/train.bin'"),
			"file_not_found",
			"error",
			False,
		),
		(exc(RuntimeError, "CUDA error: device-side assert triggered"), "cuda_error", "error", False),
		(
			exc(RuntimeError, "CUDA error: an illegal memory access was encountered"),
			"cuda_error",
			"error",
			False,
		),
		(
			exc(RuntimeError, "NCCL error in: ProcessGroupNCCL.cpp:1191, unhandled system error"),
			"nccl",
			"error",
			True,
		),
		(exc(RuntimeError, "Bf16 is not supported on this GPU"), "dtype_unsupported", "error", False),
		(
			exc(ValueError, "Your setup doesn't support bf16/gpu. You need torch>=1.10, using Ampere GPU"),
			"dtype_unsupported",
			"error",
			False,
		),
		(
			exc(RuntimeError, "mat1 and mat2 shapes cannot be multiplied (4x10 and 12x8)"),
			"shape_mismatch",
			"error",
			False,
		),
		(
			exc(
				RuntimeError,
				"DataLoader worker (pid 4123) is killed by signal: Bus error. It is possible that dataloader's workers are out of shared memory.",
			),
			"dataloader_worker",
			"error",
			False,
		),
		(
			exc(
				ImportError,
				"Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install transformers[torch]`",
			),
			"missing_package",
			"error",
			False,
		),
		(exc(SyntaxError, "invalid syntax"), "syntax_error", "error", False),
		(exc(ValueError, "labels and logits have different lengths"), "user_error", "error", False),
	],
)
def test_classify(error, code, status, retryable):
	c = classify(error, steps=0, in_eval=False)
	assert (c.code, c.status, c.retryable) == (code, status, retryable)
	assert c.message


def test_oom_during_eval_is_its_own_code():
	c = classify(exc(OutOfMemoryError, "CUDA out of memory"), steps=3, in_eval=True)
	assert c.code == "oom.eval" and c.status == "oom"
	assert "eval" in c.message


def test_missing_package_names_the_package():
	c = classify(exc(ModuleNotFoundError, "No module named 'flash_attn'"), steps=0, in_eval=False)
	assert "flash_attn" in c.message


def test_file_not_found_names_the_path():
	c = classify(
		exc(FileNotFoundError, "[Errno 2] No such file or directory: 'data/train.bin'"),
		steps=0,
		in_eval=False,
	)
	assert "data/train.bin" in c.message


@pytest.mark.parametrize(
	("returncode", "timed_out", "code", "status", "retryable"),
	[
		(None, True, "timeout", "timeout", False),
		(-9, False, "oom.host", "oom", False),
		(137, False, "oom.host", "oom", False),
		(-11, False, "killed", "error", True),
		(3, False, "script_exit", "error", False),
	],
)
def test_classify_exit(returncode, timed_out, code, status, retryable):
	c = classify_exit(returncode=returncode, timed_out=timed_out, steps=0, stage="first batch", timeout_s=60)
	assert (c.code, c.status, c.retryable) == (code, status, retryable)


def test_import_error_that_asks_for_a_package_names_it():
	c = classify(
		exc(
			ImportError,
			"Using the `Trainer` with `PyTorch` requires `accelerate>=0.26.0`: Please run `pip install x`",
		),
		steps=0,
		in_eval=False,
	)
	assert "accelerate" in c.message


def test_a_package_at_the_wrong_version_says_so_rather_than_not_installed():
	# the Nov 1 serving smoke: datasets' pyarrow wanted numpy 2 in an image pinned to numpy 1
	c = classify(ImportError("pyarrow requires NumPy 2.0 or newer, found 1.26.4"), steps=0, in_eval=False)
	assert c.code == "missing_package"
	assert "2.0 or newer" in c.message and "1.26.4" in c.message and "not installed" not in c.message

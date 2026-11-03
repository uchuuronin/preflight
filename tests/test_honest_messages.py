"""The re-validation's honest messages (Nov 3): a map refused for size isn't a syntax error and names the script the
job runs; a repository link is never told to point Preflight at a smaller directory."""

from preflight.harness.static import SourceTooLarge
from preflight.runner.failures import describe, with_step
from preflight.runner.worker import error_code


def test_a_map_refused_for_size_is_its_own_code_and_names_the_entry_point():
	exc = SourceTooLarge("the source is 2.40 MB of Python in 700 files, more than Preflight maps (2 MB)")
	assert error_code(exc) == "map.too_large"
	msg = with_step(str(exc), error_code(exc), entrypoint="recipes/lora_finetune_single_device.py")
	assert "syntax" not in msg.lower() and "train.py" not in msg and "<your script>" not in msg
	assert "preflight run recipes/lora_finetune_single_device.py" in msg


def test_the_worker_names_the_entry_point_the_pipeline_attached(monkeypatch):
	from preflight.runner import pipeline as P
	from preflight.runner.sources import Fetched

	f = Fetched(files={"recipes/x.py": "x = (\n"}, entrypoint="recipes/x.py", args=[], source=None)
	monkeypatch.setattr(P, "_flag", lambda n: False)
	try:
		P._map_or_skip(f)
	except Exception as exc:
		assert exc.entrypoint == "recipes/x.py"
		assert error_code(exc) == "syntax_error"
		assert "py_compile recipes/x.py" in with_step(str(exc), error_code(exc), entrypoint=exc.entrypoint)
	else:
		raise AssertionError("expected a refusal")


def test_the_size_advice_works_for_a_repository_link():
	_step, by_hand = describe("submission.too_large")
	assert "smaller directory" not in by_hand and "repository link" in by_hand and "entry point" in by_hand
	from preflight.runner.sources import _too_big

	assert "smaller directory" not in str(_too_big(10, 2**30)) and "entry point" in str(
		_too_big(10, 2**30, repo=True)
	)

"""probes.entry_requirements (Nov 3, the re-validation: F1's word_language_model/requirements.txt says torch>=2.6, and
only the repo top's requirements.txt was read): the requirements file nearest the entry point."""

from pathlib import Path

from preflight.recorder import requirements as R


def test_the_nearest_requirements_file_up_from_the_entry_point():
	names = {"requirements.txt", "word_language_model/requirements.txt", "a/b/main.py", "a/requirements.txt"}
	assert (
		R.entry_requirements(names, "word_language_model/main.py") == "word_language_model/requirements.txt"
	)
	assert R.entry_requirements(names, "a/b/main.py") == "a/requirements.txt"  # one folder up
	assert R.entry_requirements(names, "main.py") == "requirements.txt"
	assert (
		R.entry_requirements({"x/main.py"}, "x/main.py") == "requirements.txt"
	)  # none nearer: the top's, as before
	assert R.entry_requirements(names, "../evil/main.py") == "requirements.txt"
	assert R.entry_requirements(names, None) == "requirements.txt"


def test_f1s_torch_ask_is_read_from_its_own_folder():
	files = {"word_language_model/main.py": "import torch\n", "word_language_model/requirements.txt": "torch>=2.6\n",
		"word_language_model/base.txt": "x\n"}  # fmt: skip
	assert R.torch_asks(files) == []  # off: only the top's, which F1 doesn't have
	req = R.entry_requirements(files, "word_language_model/main.py")
	assert R.torch_asks(files, req) == ["torch>=2.6"]
	assert R.torch_too_old(R.torch_asks(files, req), R.IMAGE_TORCH)
	# -r is read relative to the file that names it
	files["word_language_model/requirements.txt"] = "-r base2.txt\n"
	files["word_language_model/base2.txt"] = "torch>=2.7\n"
	assert R.torch_asks(files, req) == ["torch>=2.7"]


def test_install_reads_the_named_file_and_never_one_outside_the_work_dir(tmp_path, monkeypatch):
	seen = []
	monkeypatch.setattr(R, "installed", lambda: {"torch": "2.5.1"})
	monkeypatch.setattr(R, "plan", lambda text, have: seen.append(text) or ([], []))
	(tmp_path / "sub").mkdir()
	(tmp_path / "sub" / "requirements.txt").write_text("tqdm\n")
	assert (
		R.install(tmp_path, tmp_path / "deps", requirements="sub/requirements.txt", log=lambda m: None)[0]
		is None
	)
	assert seen == ["tqdm\n"]
	assert R.install(tmp_path, tmp_path / "deps", requirements="../x.txt", log=lambda m: None) == (None, [])
	assert seen == ["tqdm\n"]  # treated as the top's requirements.txt, which doesn't exist: nothing read


def test_the_pipeline_picks_the_newer_image_for_f1_only_with_the_flag(monkeypatch):
	import preflight.runner.pipeline as P
	from preflight.runner.sources import Fetched

	files = {
		"word_language_model/main.py": "import torch\n",
		"word_language_model/requirements.txt": "torch>=2.6\n",
	}
	f = Fetched(files=files, entrypoint="word_language_model/main.py", args=[], source=None)
	sm = type("SM", (), {"lens": "pretrain"})()
	monkeypatch.setattr(P, "_uses_vllm", lambda sm, files: False)
	on = {"probes.torch_new"}
	monkeypatch.setattr(P, "_flag", lambda n: n in on)
	assert P._image(sm, files, P._requirements(f)) == "probe"
	on.add("probes.entry_requirements")
	assert P._requirements(f) == "word_language_model/requirements.txt"
	assert P._image(sm, files, P._requirements(f)) == "torch27"


def test_the_flag_is_off_by_default():
	import json

	assert json.loads(Path("flags.json").read_text())["probes.entry_requirements"] is False

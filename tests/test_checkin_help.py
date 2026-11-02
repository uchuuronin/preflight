"""Check-in help's job fields (contract 0.6.3, C's request, Nov 2): extra packages from Preflight's list,
starting from the entry point, and the stops said in fields (map.too_large, Probe.missing_module)."""

import subprocess

import pytest

from preflight.probes.local import InvalidSubmission
from preflight.recorder import requirements as R
from preflight.runner import record_jobs, sources
from preflight.runner.sources import extra_lines, fetch
from preflight.schema import JobRequest, Source

UPLOAD = {
	"train.py": "from pkg import model\nimport helpers\nbatch_size = 8\n",
	"pkg/__init__.py": "",
	"pkg/model.py": "from . import layers\n",
	"pkg/layers.py": "x = 1\n",
	"helpers.py": "y = 2\n",
	"unrelated/other.py": "import torch\n",
	"requirements.txt": "einops\n",
}


def _upload(**kw):
	return {"kind": "upload", "files": dict(UPLOAD), "entrypoint": "train.py", **kw}


# ---------------------------------------------------------------- extra packages


def test_extra_packages_are_names_on_the_list_installed_with_the_lists_own_pin():
	assert R.extra_requirements(["torchvision", "timm"]) == ["torchvision", "timm"]
	assert R.extra_requirements(["torchao"]) == ["torchao<0.8"]  # the list's pin, never the visitor's
	assert R.extra_requirements(["Sentence_Transformers"]) == ["sentence-transformers"]  # PyPI's rules
	assert R.extra_requirements(["sklearn", "scikit-learn"]) == ["scikit-learn"]  # a module names its package
	assert R.extra_requirements([]) == [] and R.extra_requirements(None) == []


@pytest.mark.parametrize(
	"names", [["not-a-real-pkg"], ["timm==1.0"], ["https://evil/x.whl"], ["timm", "-e ."], [""], [7]]
)
def test_a_name_off_the_list_is_refused_naming_it(names):
	with pytest.raises(R.PackageRefused) as exc:
		R.extra_requirements(names)
	assert exc.value.code == "missing_package.not_allowed"
	assert "isn't on it" in str(exc.value)


def test_a_package_with_no_wheels_says_so_and_more_than_ten_are_refused():
	with pytest.raises(R.PackageRefused) as exc:
		R.extra_requirements(["torch-sparse", "timm"])
	assert exc.value.code == "missing_package.no_wheels" and exc.value.names == ["torch-sparse"]
	assert R.extra_requirements(["timm"] * 11) == ["timm"]  # counted once each
	eleven = sorted(R.known_packages())[:11]
	with pytest.raises(R.PackageRefused, match="at most 10") as exc:
		R.extra_requirements(eleven)
	assert exc.value.code == "submission.invalid"
	with pytest.raises(InvalidSubmission) as exc:  # a record's ten and a request's one more
		fetch(_upload(extra_packages=eleven))
	assert exc.value.code == "submission.invalid" and "at most 10" in str(exc.value)
	pair = [*[n for n in eleven if n != "scikit-learn"][:9], "sklearn", "scikit-learn"]
	assert len(R.extra_requirements(pair)) == 10  # an alias pair is one package


def test_the_list_c_serves_is_every_known_package_by_pypi_name():
	known = R.known_packages()
	assert known["torch-geometric"] == "torch-geometric" and known["torchao"] == "torchao<0.8"
	assert "torch-sparse" not in known and len(known) == len(set(R.KNOWN_MODULES.values()))


def test_fetch_keeps_the_extra_packages_on_the_source_and_refuses_off_list_ones():
	got = fetch(_upload(extra_packages=["torchvision", "sklearn"]))
	assert got.source.extra_packages == ["torchvision", "scikit-learn"]
	assert extra_lines(got.source) == ("torchvision", "scikit-learn")
	assert fetch(_upload()).source.extra_packages is None  # absent stays absent on the record
	with pytest.raises(InvalidSubmission) as exc:
		fetch(_upload(extra_packages=["leftpad"]))
	assert exc.value.code == "missing_package.not_allowed" and "leftpad" in str(exc.value)
	with pytest.raises(InvalidSubmission) as exc:
		fetch(_upload(extra_packages=["torch_scatter"]))
	assert exc.value.code == "missing_package.no_wheels"


def test_install_adds_the_extra_packages_with_or_without_a_requirements_file(tmp_path, monkeypatch):
	seen = []

	def fake_run(cmd, **kw):
		seen.append(cmd)
		return subprocess.CompletedProcess(cmd, 0, "", "")

	monkeypatch.setattr(R.subprocess, "run", fake_run)
	monkeypatch.setattr(R, "installed", lambda: {"torch": "2.5.1+cu121"})
	monkeypatch.delenv("PREFLIGHT_DEPS_DIR", raising=False)
	(tmp_path / "run.py").write_text("import timm\n")
	err, _ = R.install(tmp_path, tmp_path / "d1", log=lambda m: None, extra=["timm", "torchao<0.8"])
	assert err is None and seen[-1][-2:] == ["timm", "torchao<0.8"] and "--only-binary" in seen[-1]
	(tmp_path / "requirements.txt").write_text("einops\n")
	err, _ = R.install(tmp_path, tmp_path / "d2", log=lambda m: None, extra=["timm"])
	assert err is None and seen[-1][-2:] == ["timm", "einops"]  # the extra packages first
	monkeypatch.setattr(R, "installed", lambda: {"timm": "1.0", "einops": "0.8"})
	n = len(seen)  # the image's own copy wins: nothing left to install
	assert (
		R.install(tmp_path, tmp_path / "d3", log=lambda m: None, extra=["timm"])[0] is None and len(seen) == n
	)


def test_a_record_read_again_installs_the_same_packages_and_reads_the_same_files():
	src = Source(kind="repo", uri="https://github.com/a/b", commit="0" * 40, entrypoint="train.py",
		scope="entrypoint", extra_packages=["timm"])  # fmt: skip
	again = record_jobs.refetch(src, None)
	assert again["scope"] == "entrypoint" and again["extra_packages"] == ["timm"]
	plain = record_jobs.refetch(Source(kind="repo", uri="https://github.com/a/b", commit="0" * 40), None)
	assert "scope" not in plain and "extra_packages" not in plain


def test_the_probe_spec_carries_the_lines_and_the_cache_key_changes_only_when_set():
	from preflight.probes.local import ProbeSpec
	from preflight.runner.pipeline import cache_key

	base = ProbeSpec(probe_id="p1", files={"t.py": "x"}, entrypoint="t.py", install_requirements=True)
	with_pkgs = ProbeSpec(probe_id="p1", files={"t.py": "x"}, entrypoint="t.py", install_requirements=True,
		extra_packages=("timm",))  # fmt: skip
	assert cache_key(base, "L4", "img") != cache_key(with_pkgs, "L4", "img")
	assert cache_key(base, "L4", "img") == cache_key(
		ProbeSpec(probe_id="p1", files={"t.py": "x"}, entrypoint="t.py", install_requirements=True),
		"L4",
		"img",
	)


def test_the_job_request_takes_up_to_ten_names():
	rec = {"kind": "upload", "files": {"t.py": "x"}}
	assert JobRequest(source=rec).extra_packages is None
	# absent stays absent: a queued request's body (and its idempotency hash) is as before
	assert "extra_packages" not in JobRequest(source=rec).model_dump(exclude_none=True)
	assert JobRequest(source=rec, extra_packages=["timm"]).extra_packages == ["timm"]
	with pytest.raises(ValueError):
		JobRequest(source=rec, extra_packages=["timm"] * 11)


# ---------------------------------------------------------------- starting from the entry point


def test_scope_entrypoint_reads_the_entry_point_what_it_imports_and_the_requirements():
	got = fetch(_upload(scope="entrypoint"))
	assert set(got.files) == {"train.py", "pkg/__init__.py", "pkg/model.py", "pkg/layers.py", "helpers.py",
		"requirements.txt"}  # fmt: skip
	assert got.source.scope == "entrypoint" and got.partial["asked"] is True
	whole = fetch(_upload())
	assert "unrelated/other.py" in whole.files and whole.source.scope is None and whole.partial is None


def test_scope_entrypoint_needs_a_python_entry_point_and_scope_is_checked():
	with pytest.raises(InvalidSubmission) as exc:
		fetch({"kind": "upload", "files": dict(UPLOAD), "scope": "entrypoint"})
	assert exc.value.code == "submission.invalid"
	with pytest.raises(InvalidSubmission) as exc:
		fetch(_upload(scope="everything"))
	assert exc.value.code == "submission.invalid"
	with pytest.raises(InvalidSubmission) as exc:
		fetch({"kind": "upload", "files": {**UPLOAD, "run.sh": "python train.py\n"}, "entrypoint": "run.sh",
			"scope": "entrypoint"})  # fmt: skip
	assert exc.value.code == "submission.invalid"


def test_the_finding_says_the_visitor_asked_for_it():
	from preflight.runner.pipeline import _why_partial

	p = {"repo_files": 7, "repo_bytes": 100, "asked": True}
	assert _why_partial(p).startswith("You asked Preflight to start from the entry point")
	assert _why_partial({**p, "asked": False}).startswith("The repository has 7 text files")


@pytest.mark.skipif(not sources.shutil.which("git"), reason="needs git")
def test_a_repo_with_scope_entrypoint_is_read_as_its_closure_even_when_small(tmp_path, monkeypatch):
	env = {
		"GIT_TERMINAL_PROMPT": "0",
		"GIT_CONFIG_NOSYSTEM": "1",
		"PATH": "/usr/bin:/bin",
		"HOME": str(tmp_path),
	}
	src = tmp_path / "src"
	for rel, text in UPLOAD.items():
		(src / rel).parent.mkdir(parents=True, exist_ok=True)
		(src / rel).write_text(text)

	def git(*a, cwd=src):
		subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True, env=env)

	git("init", "-q", "-b", "main")
	git("add", "-A")
	git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
	dest = tmp_path / "clone"
	git(
		"-c",
		"protocol.file.allow=always",
		"clone",
		"-q",
		"--no-checkout",
		f"file://{src}",
		str(dest),
		cwd=tmp_path,
	)
	base = ["git", "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=never"]
	monkeypatch.setattr("preflight.runner.flags.flag", lambda name: False)  # read_closure off: not needed
	files, info = sources._git_read(dest, "HEAD", base=base, env=env, entry="train.py", only_entry=True)
	assert "unrelated/other.py" not in files and "pkg/layers.py" in files and info["asked"] is True
	whole, none = sources._git_read(dest, "HEAD", base=base, env=env, entry="train.py")
	assert "unrelated/other.py" in whole and none is None


# ---------------------------------------------------------------- the stops, said in fields


def test_a_missing_import_names_the_module_and_the_package_on_the_list():
	from preflight.probes.local import _missing

	msg = "The script imports torchvision, which is not installed in the probe image"
	assert _missing("missing_package", msg) == {
		"missing_module": "torchvision",
		"missing_package": "torchvision",
	}
	assert (
		_missing("missing_package", msg.replace("torchvision", "sklearn"))["missing_package"]
		== "scikit-learn"
	)
	# off the list: the module only; a submodule: the package is there at another version
	assert _missing("missing_package", msg.replace("torchvision", "leftpad")) == {"missing_module": "leftpad"}
	assert _missing("missing_package", msg.replace("torchvision", "torchao.quantization")) == {
		"missing_module": "torchao.quantization"
	}
	assert _missing("oom", msg) == {} and _missing("missing_package", "A library needs x") == {}


def test_the_maps_limit_has_its_own_code_and_every_new_code_its_own_step():
	from preflight.harness.static import SourceTooLarge
	from preflight.runner.failures import STEPS

	assert SourceTooLarge.code == "map.too_large"
	for code in ("map.too_large", "missing_package.not_allowed", "missing_package.no_wheels"):
		assert code in STEPS


# ---------------------------------------------------------------- review fixes (Nov 2)


def test_the_extra_packages_go_before_a_long_requirements_file(tmp_path, monkeypatch):
	seen = []
	monkeypatch.setattr(
		R.subprocess, "run", lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", "")
	)
	monkeypatch.setattr(R, "installed", lambda: {})
	monkeypatch.delenv("PREFLIGHT_DEPS_DIR", raising=False)
	(tmp_path / "requirements.txt").write_text("".join(f"pkg{i}\n" for i in range(60)))
	assert R.install(tmp_path, tmp_path / "d", log=lambda m: None, extra=["timm"])[0] is None
	assert "timm" in seen[-1]  # the requirements' cap (50) drops the file's tail, never them


def test_a_library_asking_for_a_package_names_it_too():
	from preflight.probes.local import _missing

	msg = "A library in the script needs accelerate, which is not installed in the probe image"
	assert _missing("missing_package", msg) == {"missing_module": "accelerate"}  # not on the list
	assert _missing("missing_package", msg.replace("accelerate", "timm")) == {
		"missing_module": "timm", "missing_package": "timm"}  # fmt: skip


def test_the_upload_limits_are_the_read_limit_code():
	from preflight.probes.local import MAX_FILES, validate_files

	with pytest.raises(InvalidSubmission) as exc:
		validate_files({f"f{i}.py": "" for i in range(MAX_FILES + 1)}, entrypoint="f0.py")
	assert exc.value.code == "submission.too_large"


def test_a_sample_record_read_again_with_scope_keeps_working():
	src = Source(kind="sample", uri="sample://zoo/x", entrypoint="sub/train.py", scope="entrypoint")
	assert "scope" not in record_jobs.refetch(src, None)  # no entry point to start from: read whole


def test_the_pipeline_installs_them_keeps_them_and_a_later_job_adds_to_them(tmp_path):
	from pathlib import Path

	from preflight.probes.local import ProbeResult
	from preflight.runner.pipeline import Deps, _Ctx, run_pipeline
	from preflight.runner.records import LocalRecordWriter
	from preflight.runner.remote_store import DirBlobs, externalise_files
	from preflight.schema import Env, Probe

	specs = []

	def runner(spec, gpu, *, should_stop):
		specs.append(spec)
		probe = Probe(id=spec.probe_id, knobs=spec.knobs, status="ok", steps=7, duration_s=30.0)
		return ProbeResult(probe=probe, env=Env(backend="cuda", provider="modal", gpu={"name": gpu}), summary={},
			kernels=None, arch=None, trace_file=None, log_tail="", workdir=Path("."))  # fmt: skip

	store = LocalRecordWriter(tmp_path / "records")
	deps = Deps(
		records=store, runner=runner, reader=store, install_requirements=True, blobs=DirBlobs(tmp_path / "up")
	)
	script = "import torch\nbatch_size = 8\nmodel = torch.nn.Linear(4, 4)\nopt = torch.optim.SGD(model.parameters(), 0.1)\n"
	req = externalise_files({"source": {"kind": "upload", "files": {"train.py": script}}, "extra_packages": ["timm"]},
		deps.blobs)  # fmt: skip
	rec = run_pipeline(req, _Ctx(), deps)
	assert (
		rec.job.source.extra_packages == ["timm"]
		and specs
		and all(s.extra_packages == ("timm",) for s in specs)
	)
	store.put(rec)
	specs.clear()
	ref = {"kind": "record", "record_ref": rec.share_token}
	again = run_pipeline({"source": ref, "mode": "probe", "knobs": {"batch_size": 4}, "extra_packages": ["einops"]},
		_Ctx(), deps)  # fmt: skip
	assert again.job.source.extra_packages == ["timm", "einops"]  # the record's first, then the new
	assert specs and all(s.extra_packages == ("timm", "einops") for s in specs)
	specs.clear()
	plain = run_pipeline({"source": ref, "mode": "probe", "knobs": {"batch_size": 2}}, _Ctx(), deps)
	assert plain.job.source.extra_packages == ["timm"] and all(s.extra_packages == ("timm",) for s in specs)

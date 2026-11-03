"""Reading a big repository as its entry point and the files it imports (D, Nov 2)."""

import json

import pytest

from preflight.harness.closure import import_closure
from preflight.harness.static import MAX_SOURCE_BYTES, ScriptParseError, map_static, mapped_files


@pytest.fixture
def closure_on(monkeypatch):
	from preflight.runner import flags

	monkeypatch.setenv("PREFLIGHT_FLAGS", json.dumps({"runner.read_closure": True}))
	flags.reload()
	yield
	monkeypatch.delenv("PREFLIGHT_FLAGS")
	flags.reload()


REPO = {
	"train.py": "import torch\nfrom pkg.models import net\nfrom pkg import data\nimport pkg.util as u\nbatch_size = 8\n",
	"pkg/__init__.py": "",
	"pkg/models/__init__.py": "from .net import Net\n",
	"pkg/models/net.py": "from ..util import helper\nclass Net: pass\n",
	"pkg/data.py": "import json\n",
	"pkg/util.py": "def helper(): pass\n",
	"pkg/unused.py": "x = 1\n",
	"other/big.py": "y = 2\n",
	"requirements.txt": "torch\n",
	"src/lib/__init__.py": "",
	"README.md": "hi",
}


def read(paths):
	return {p: REPO[p] for p in paths}


def test_it_follows_imports_absolute_relative_and_from_forms_and_nothing_else():
	got, info = import_closure(REPO, read, "train.py", max_files=100, max_bytes=10_000)
	assert set(got) == {"train.py", "requirements.txt", "pkg/__init__.py", "pkg/models/__init__.py",
		"pkg/models/net.py", "pkg/data.py", "pkg/util.py"}  # fmt: skip
	assert (
		info["files"] == 7
		and info["imported"] == 5
		and info["skipped"] == 0
		and info["repo_files"] == len(REPO)
	)
	assert next(iter(got)) == "train.py"  # nearest first


def test_the_budget_keeps_the_nearest_files_and_counts_the_rest():
	sizes = {p: len(t) for p, t in REPO.items()}
	got, info = import_closure(REPO, read, "train.py", max_files=3, max_bytes=10_000, sizes=sizes)
	assert len(got) == 3 and "train.py" in got and info["skipped"] > 0


def test_a_file_that_doesnt_parse_is_kept_but_not_followed():
	repo = {"main.py": "import a\n", "a.py": "def (:\nimport b\n", "b.py": ""}
	got, _ = import_closure(repo, lambda ps: {p: repo[p] for p in ps}, "main.py", max_files=9, max_bytes=999)
	assert set(got) == {"main.py", "a.py"}


def test_the_entry_points_folder_and_src_are_import_roots():
	repo = {
		"recipes/run.py": "import helper\nimport lib\n",
		"recipes/helper.py": "",
		"src/lib/__init__.py": "",
	}
	got, _ = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "recipes/run.py", max_files=9, max_bytes=999
	)
	assert set(got) == {"recipes/run.py", "recipes/helper.py", "src/lib/__init__.py"}


def test_the_mapper_maps_the_closure_when_the_repo_is_over_its_budget(closure_on):
	pad = "# " + "x" * 1000 + "\n"
	big = {f"vendor/m{i}.py": pad * 100 for i in range(25)}  # about 2.5 MB nobody imports
	files = {**REPO, **big}
	kept, info = mapped_files(files, "train.py")
	assert info is not None and info["python_bytes"] > MAX_SOURCE_BYTES
	assert not any(p.startswith("vendor/") for p in kept) and "README.md" in kept
	sm = map_static(files, entrypoint="train.py")  # no "too large" refusal
	assert any(k.id == "batch_size" for k in sm.knobs)
	assert mapped_files(REPO, "train.py") == (REPO, None)  # under the budget: everything, as before


def test_an_entry_point_over_the_budget_alone_is_refused_accurately():
	with pytest.raises(
		ScriptParseError, match=r"big\.py alone is .* MB of Python, more than Preflight maps"
	) as exc:
		mapped_files({"big.py": "x = 1\n" * 600_000}, "big.py")
	assert exc.value.code == "map.too_large"  # 0.6.3: the map's limit, not the read limit (C)


def test_the_record_says_what_was_read(closure_on):
	from preflight.runner.pipeline import _reading_findings
	from preflight.runner.sources import Fetched

	whole = Fetched(files=REPO, entrypoint="train.py")
	assert _reading_findings(whole) == []
	part = Fetched(files=REPO, entrypoint="train.py",
		partial={"files": 7, "imported": 5, "bytes": 300, "skipped": 4, "repo_files": 15449, "repo_bytes": 12_300_182})  # fmt: skip
	(f,) = _reading_findings(part)
	assert f.code == "source.partial" and "15,449 text files" in f.detail and "5 Python files" in f.detail
	assert "at least 4 more imported files were left out" in f.detail
	pad = "# " + "x" * 1000 + "\n"
	big = Fetched(files={**REPO, **{f"vendor/m{i}.py": pad * 100 for i in range(25)}}, entrypoint="train.py")
	(g,) = _reading_findings(big)
	assert g.id == "source.partial.map" and "the map reads train.py and the 5 files" in g.detail


def test_with_the_flag_off_over_the_budget_is_refused_saying_how_big():
	pad = "# " + "x" * 1000 + "\n"
	files = {**REPO, **{f"vendor/m{i}.py": pad * 100 for i in range(25)}}
	with pytest.raises(ScriptParseError, match=r"the source is 2\.\d\d MB of Python in 34 files") as exc:
		map_static(files, entrypoint="train.py")
	assert exc.value.code == "map.too_large"  # 0.6.3: the map's limit, not the read limit (C)


def test_a_hostile_dotted_name_stays_cheap():
	import time

	name = ".".join(["a"] * 60_000)
	repo = {"main.py": f"import {name}\nfrom {name} import " + ", ".join(f"n{i}" for i in range(5000)) + "\n"}
	t = time.monotonic()
	got, _ = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "main.py", max_files=9, max_bytes=10**7
	)
	assert set(got) == {"main.py"} and time.monotonic() - t < 2
	# capped names: a long but parseable import list costs little
	names = ", ".join(f"n{i}" for i in range(20_000))
	repo = {
		"main.py": "\n".join(f"from pkg import {n}" for n in names.split(", ")) + "\n",
		"pkg/__init__.py": "",
	}
	t = time.monotonic()
	import_closure(repo, lambda ps: {p: repo[p] for p in ps}, "main.py", max_files=9, max_bytes=10**7)
	assert time.monotonic() - t < 10


def test_type_checking_imports_aren_t_followed_and_the_first_root_wins():
	repo = {
		"recipes/run.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import heavy\nimport utils\n",
		"recipes/utils.py": "",
		"utils.py": "",
		"heavy.py": "",
	}
	got, _ = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "recipes/run.py", max_files=9, max_bytes=999
	)
	assert set(got) == {"recipes/run.py", "recipes/utils.py"}


def test_project_files_are_read_but_not_followed_or_counted():
	repo = {"train.py": "import a\n", "a.py": "", "setup.py": "import b\n", "b.py": ""}
	got, info = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "train.py", max_files=9, max_bytes=999
	)
	assert set(got) == {"train.py", "setup.py", "a.py"} and info["imported"] == 1


def test_the_entry_point_is_read_even_over_the_budget():
	repo = {"train.py": "x = 1\n" * 100, "a.py": ""}
	sizes = {p: len(t) for p, t in repo.items()}
	got, _ = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "train.py", max_files=9, max_bytes=10, sizes=sizes
	)
	assert "train.py" in got


def test_a_non_python_entry_point_over_the_budget_is_refused_saying_why(closure_on):
	pad = "# " + "x" * 1000 + "\n"
	files = {"launch.sh": "python train.py\n", **{f"m{i}.py": pad * 100 for i in range(25)}}
	with pytest.raises(ScriptParseError, match=r"launch\.sh isn't a Python file") as exc:
		mapped_files(files, "launch.sh")
	assert exc.value.code == "map.too_large"  # 0.6.3: the map's limit, not the read limit (C)


def test_a_module_beats_a_folder_of_the_same_name_without_init():
	repo = {"train.py": "import data\nimport ns.mod\n", "data.py": "", "data/vocab.txt": "", "ns/mod.py": ""}
	got, _ = import_closure(repo, lambda ps: {p: repo[p] for p in ps}, "train.py", max_files=9, max_bytes=999)
	assert set(got) == {"train.py", "data.py", "ns/mod.py"}


def test_continuation_lines_dont_hide_a_hostile_name():
	import time

	repo = {"main.py": "import a" + "\\\n.a" * 30_000 + "\n"}
	t = time.monotonic()
	got, _ = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "main.py", max_files=9, max_bytes=10**7
	)
	assert set(got) == {"main.py"} and time.monotonic() - t < 2


def test_no_imported_files_reads_as_none():
	from preflight.runner.pipeline import _n_files

	assert _n_files(0, "file") == "none of the files" and _n_files(1, "file") == "the 1 file"
	assert _n_files(2_000, "file") == "the 2,000 files"


def test_the_cut_is_by_lookups_not_a_clock_so_two_reads_match():
	"""A's fix (Nov 3): the 30 s clock could read different files on a slow day, changing a record's
	content hash; the walk now stops on a lookup count, the same on every read."""
	import preflight.harness.closure as C

	repo = {
		"main.py": "".join(f"import m{i}\n" for i in range(50)),
		**{f"m{i}.py": f"import n{i}\n" for i in range(50)},
		**{f"n{i}.py": "" for i in range(50)},
	}
	read = lambda ps: {p: repo[p] for p in ps}  # noqa: E731
	assert not hasattr(C, "time")  # no clock in the walk
	a, ia = import_closure(repo, read, "main.py", max_files=999, max_bytes=10**7, max_lookups=60)
	b, ib = import_closure(repo, read, "main.py", max_files=999, max_bytes=10**7, max_lookups=60)
	assert a == b and ia == ib and ia["cut"] == "lookups" and ia["skipped"] > 0
	full, info = import_closure(repo, read, "main.py", max_files=999, max_bytes=10**7)
	assert len(full) == 101 and info["cut"] is None and info["skipped"] == 0


def test_a_very_long_path_isnt_followed_and_a_cut_is_said_as_a_cut():
	from preflight.harness.closure import MAX_PATH

	long = "d/" * (MAX_PATH // 2) + "m.py"
	repo = {"main.py": "import d\n", long: "", "d/__init__.py": ""}
	got, info = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "main.py", max_files=9, max_bytes=999
	)
	assert long not in got and "d/__init__.py" in got and info["repo_files"] == 2
	from preflight.runner.pipeline import _reading_findings
	from preflight.runner.sources import Fetched

	partial = {
		"files": 3,
		"imported": 2,
		"bytes": 10,
		"skipped": 4,
		"cut": "lookups",
		"repo_files": 9,
		"repo_bytes": 99,
	}
	f = Fetched(files={"main.py": ""}, entrypoint="main.py", source=None, partial=partial)
	assert "when the import lookups ran out" in _reading_findings(f)[0].detail


def test_files_the_code_may_open_are_read_but_not_followed():
	"""B's must-fix 3 (Nov 3): `--config configs/a.yaml`, a vocab file named in the code, a file beside a module."""
	repo = {
		"train.py": "import mypkg.data\nopen('configs/base.yaml')\n",
		"mypkg/__init__.py": "",
		"mypkg/data.py": "VOCAB = 'vocab.json'\nimport os\n",
		"mypkg/vocab.json": "{}",
		"configs/a.yaml": "lr: 1\n",
		"configs/base.yaml": "x: 1\n",
		"configs/other.yaml": "unused: 1\n",
		"README.md": "# docs",
		"tools/evil.py": "import os\n",
	}
	got, info = import_closure(repo, lambda ps: {p: repo[p] for p in ps}, "train.py", max_files=99,
		max_bytes=10**6, args=["--config", "configs/a.yaml"])  # fmt: skip
	assert {"configs/a.yaml", "configs/base.yaml", "mypkg/vocab.json"} <= set(got)
	assert "configs/other.yaml" not in got and "README.md" not in got and "tools/evil.py" not in got
	assert info["data_read"] == 3 and info["data_skipped"] == 0
	got2, _ = import_closure(repo, lambda ps: {p: repo[p] for p in ps}, "train.py", max_files=99,
		max_bytes=10**6, args=["--config", "configs/a.yaml"])  # fmt: skip
	assert got == got2  # still the same files on every read
	_, small = import_closure(repo, lambda ps: {p: repo[p] for p in ps}, "train.py", max_files=4, max_bytes=10**6,
		args=["--config", "configs/a.yaml"])  # fmt: skip
	assert small["data_skipped"] > 0
	repo2 = {"a/train.py": "open('../../etc/passwd')\n", "etc/passwd": "x"}
	escaped, _ = import_closure(
		repo2, lambda ps: {p: repo2[p] for p in ps}, "a/train.py", max_files=9, max_bytes=999
	)
	assert "etc/passwd" not in escaped  # a path that leaves the repository is never read


def test_a_lazy_package_s_type_checking_imports_are_followed():
	"""B's must-fix 4 (Nov 3): a transformers-style lazy __init__ names its real modules under TYPE_CHECKING."""
	repo = {
		"train.py": "from mylib import Trainer\n",
		"mylib/__init__.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    from .trainer import Trainer\n",
		"mylib/trainer.py": "class Trainer: ...\n",
		"app.py": "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import heavy\n",
		"heavy.py": "",
	}
	got, _ = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "train.py", max_files=99, max_bytes=10**6
	)
	assert "mylib/trainer.py" in got
	got, _ = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "app.py", max_files=99, max_bytes=10**6
	)
	assert "heavy.py" not in got  # a module's own TYPE_CHECKING body still isn't followed


def test_the_data_pass_keeps_the_budget_without_sizes_and_doesn_t_count_as_imports():
	repo = {"train.py": "import util\n", "util.py": "", "big.json": "x" * 50, "b.json": "y"}
	got, info = import_closure(repo, lambda ps: {p: repo[p] for p in ps}, "train.py", max_files=99, max_bytes=30,
		sizes={"train.py": 12, "util.py": 0}, args=["util.py"])  # fmt: skip
	assert "big.json" not in got and info["bytes"] <= 30 and "big.json" in info["data_left_out"]
	assert info["imported"] == 1  # util.py, imported; naming it in the arguments doesn't count it twice


def test_a_follow_up_on_a_record_read_before_the_data_pass_still_matches_its_code():
	from preflight.runner.sources import content_hash

	repo = {"train.py": "import util\n", "util.py": "", "notes.txt": "x"}
	got, info = import_closure(
		repo, lambda ps: {p: repo[p] for p in ps}, "train.py", max_files=99, max_bytes=999
	)
	assert info["data_files"] == ["notes.txt"]
	before = content_hash({k: v for k, v in got.items() if k not in info["data_files"]})
	assert before == content_hash({"train.py": "import util\n", "util.py": ""})  # what the old read hashed

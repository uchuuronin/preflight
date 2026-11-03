"""probes.dataset_cache (off; the PM, Nov 3): a torchvision download whose md5 the volume has is a copy."""

import hashlib
import types

from preflight.recorder import dataset_cache as C


def _utils(calls):
	def download_url(url, root, filename=None, md5=None, max_redirect_hops=3):
		calls.append(url)

	def check_integrity(path, md5=None):
		try:
			return hashlib.md5(open(path, "rb").read()).hexdigest() == md5
		except OSError:
			return False

	return types.SimpleNamespace(download_url=download_url, check_integrity=check_integrity)


def test_a_cached_file_is_copied_and_anything_else_downloads(tmp_path):
	body = b"cifar bytes"
	md5 = hashlib.md5(body).hexdigest()
	(tmp_path / "cache" / "by-md5").mkdir(parents=True)
	(tmp_path / "cache" / "by-md5" / md5).write_bytes(body)
	calls, said = [], []
	u = _utils(calls)
	C.patch(u, tmp_path / "cache", log=said.append)
	u.download_url("https://x/cifar-10-python.tar.gz", str(tmp_path / "data"), md5=md5)
	assert (tmp_path / "data" / "cifar-10-python.tar.gz").read_bytes() == body and calls == [] and said
	u.download_url("https://x/other.tar.gz", str(tmp_path / "data"), md5="0" * 32)  # not in the volume
	u.download_url("https://x/nomd5.tar.gz", str(tmp_path / "data"))  # no md5: never from the cache
	assert calls == ["https://x/other.tar.gz", "https://x/nomd5.tar.gz"]


def test_a_hostile_md5_never_reads_outside_the_cache(tmp_path):
	assert C.cached(tmp_path, "../../etc/passwd") is None and C.cached(tmp_path, None) is None


def test_the_spec_carries_it_only_when_set():
	from preflight.probes.local import ProbeSpec
	from preflight.runner.pipeline import cache_key

	a = ProbeSpec(probe_id="p", files={"t.py": ""}, entrypoint="t.py")
	b = ProbeSpec(probe_id="p", files={"t.py": ""}, entrypoint="t.py", dataset_cache=True)
	assert a.dataset_cache is False and cache_key(a, "L4", "i") != cache_key(b, "L4", "i")


def test_only_code_that_imports_torchvision_is_keyed_with_it():
	from preflight.runner.pipeline import _uses_torchvision

	assert _uses_torchvision({"t.py": "import torch\nfrom torchvision import datasets\n"})
	assert not _uses_torchvision(
		{"t.py": "import torch\n# torchvision later\n", "notes.md": "import torchvision"}
	)


def test_ogb_s_archive_is_copied_by_its_url_and_its_size_check_skipped(tmp_path, monkeypatch):
	import hashlib
	import types

	url = C.URL_FILES["ogbn-arxiv"]
	(tmp_path / "cache" / "by-url").mkdir(parents=True)
	(tmp_path / "cache" / "by-url" / C.url_key(url)).write_bytes(b"zip bytes")
	monkeypatch.setitem(C.URL_SHA256, url, hashlib.sha256(b"zip bytes").hexdigest())
	opened = []
	mod = types.SimpleNamespace(decide_download=lambda u: opened.append(u) or True,
		download_url=lambda u, folder, log=True: opened.append(u) or "fetched")  # fmt: skip
	C.patch_ogb(mod, tmp_path / "cache", log=lambda m: None)
	assert mod.decide_download(url) is True and opened == []  # no HTTP request for the cached one
	got = mod.download_url(url, str(tmp_path / "data" / "arxiv"))
	assert (
		got.endswith("arxiv.zip") and (tmp_path / "data" / "arxiv" / "arxiv.zip").read_bytes() == b"zip bytes"
	)
	other = "http://snap.stanford.edu/ogb/data/nodeproppred/products.zip"
	assert mod.download_url(other, str(tmp_path)) == "fetched" and opened == [other]
	assert C.cached_url(tmp_path / "cache", "http://evil/../../x") is None  # only the URLs the cache names


def test_ogb_code_gets_the_cache_too():
	from preflight.runner.pipeline import _uses_torchvision

	assert _uses_torchvision({"gnn.py": "from ogb.nodeproppred import PygNodePropPredDataset\n"})


def test_an_archive_whose_hash_isn_t_the_pinned_one_is_never_served(tmp_path, monkeypatch):
	url = C.URL_FILES["ogbn-arxiv"]
	(tmp_path / "by-url").mkdir()
	(tmp_path / "by-url" / C.url_key(url)).write_bytes(b"<html>error page</html>")
	assert C.cached_url(tmp_path, url) is None  # nothing pinned: never served
	monkeypatch.setitem(C.URL_SHA256, url, "0" * 64)
	assert C.cached_url(tmp_path, url) is None  # pinned, but not this file

"""probes.dataset_cache (off; the PM, Nov 3): torchvision's common datasets from a read-only Modal volume,
before any download.

Unseen 9 (airbench, Nov 2) spent its whole probe downloading CIFAR-10 from torchvision's host at about
37 kB/s. With the flag on, the probe sandbox mounts the `preflight-datasets` volume read-only at
CACHE_DIR, and the recorder wraps torchvision.datasets.utils.download_url: a file the dataset asks for
by its md5 is copied from the volume (CACHE_DIR/by-md5/<md5>) into the script's own root, and torchvision
then checks and extracts it as if it had downloaded it. A file the volume doesn't have, or one asked for
without an md5, downloads as before. The volume holds the archives torchvision itself names
(RESOURCES), each verified by its md5 when the ops job fills it (runner/ops_app.py fill_dataset_cache).
"""

import os
import shutil
from pathlib import Path

CACHE_DIR = "/datasets"  # the volume's mount in the probe sandbox (runner/modal_app.py)
DATASETS = ("CIFAR10", "CIFAR100", "MNIST", "FashionMNIST", "ogbn-arxiv")
# archives with no published md5, kept by their URL (CACHE_DIR/by-url/<sha256 of the URL>): OGB's, which it fetches over
# plain HTTP, which the probe sandbox's allowlist doesn't open (a-fresh-live-ogb-2-checkin, Nov 3)
URL_FILES = {"ogbn-arxiv": "http://snap.stanford.edu/ogb/data/nodeproppred/arxiv.zip"}
# each URL_FILES archive's sha256, pinned from the fill job's own download (OGB publishes no checksum, and its host is
# plain HTTP): the cache serves an archive only when its copy matches, so a tampered or error-page file never is
URL_SHA256: dict[str, str] = {}


def resources() -> list[tuple[str, list[str], str]]:
	"""(dataset, urls to try in order, md5) for every archive the cache holds: CIFAR-10/100, MNIST,
	FashionMNIST, as the image's torchvision names them, every mirror it lists (MNIST's first, yann.lecun.com,
	answered 404 on Nov 3)."""
	from torchvision import datasets as D

	out = [
		("CIFAR10", [D.CIFAR10.url], D.CIFAR10.tgz_md5),
		("CIFAR100", [D.CIFAR100.url], D.CIFAR100.tgz_md5),
	]
	for cls in (D.MNIST, D.FashionMNIST):
		out += [(cls.__name__, [f"{m}{name}" for m in cls.mirrors], md5) for name, md5 in cls.resources]
	return out + [(name, [url], None) for name, url in URL_FILES.items()]


def url_key(url: str) -> str:
	import hashlib

	return hashlib.sha256(url.encode()).hexdigest()[:32]


def cached_url(cache_dir: str | os.PathLike, url: str) -> Path | None:
	"""The volume's copy of an archive kept by its URL (URL_FILES only), when it has one whose sha256 is the
	pinned one (none pinned: never served)."""
	import hashlib

	want = URL_SHA256.get(url)
	if url not in URL_FILES.values() or not want:
		return None
	p = Path(cache_dir) / "by-url" / url_key(url)
	if not p.is_file():
		return None
	h = hashlib.sha256()
	with open(p, "rb") as f:
		while chunk := f.read(1 << 20):
			h.update(chunk)
	return p if h.hexdigest() == want else None


def patch_ogb(url_mod, cache_dir: str | os.PathLike, log=print) -> None:
	"""Wrap ogb.utils.url's decide_download (which opens the URL to read its size) and download_url, so an OGB
	archive the volume has is copied, not fetched. The dataset modules import both names from it, after it has
	run, so the wrappers are what they get."""
	orig_decide, orig_download = url_mod.decide_download, url_mod.download_url

	def decide_download(url, *a, **kw):
		return True if cached_url(cache_dir, url) is not None else orig_decide(url, *a, **kw)

	def download_url(url, folder, *a, **kw):
		src = cached_url(cache_dir, url)
		if src is None:
			return orig_download(url, folder, *a, **kw)
		dst = os.path.join(folder, url.rpartition("/")[2])
		os.makedirs(folder, exist_ok=True)
		if not (os.path.exists(dst) and os.path.getsize(dst) > 0):
			shutil.copyfile(src, dst)
		log(f"[preflight] dataset cache: {os.path.basename(dst)} from Preflight's copy, not downloaded")
		return dst

	decide_download.__wrapped__, download_url.__wrapped__ = orig_decide, orig_download
	url_mod.decide_download, url_mod.download_url = decide_download, download_url


def cached(cache_dir: str | os.PathLike, md5: str | None) -> Path | None:
	"""The volume's copy of the file with this md5, when it has one."""
	if not md5 or not all(c in "0123456789abcdef" for c in md5.lower()) or len(md5) != 32:
		return None
	p = Path(cache_dir) / "by-md5" / md5.lower()
	return p if p.is_file() else None


def patch(utils, cache_dir: str | os.PathLike, log=print) -> None:
	"""Wrap utils.download_url (torchvision.datasets.utils) so a cached file is copied, not downloaded.
	download_and_extract_archive calls download_url through the module, so every dataset's download goes
	through it."""
	orig = utils.download_url

	def download_url(url, root, filename=None, md5=None, *args, **kwargs):
		src = cached(cache_dir, md5)
		if src is None:
			return orig(url, root, filename, md5, *args, **kwargs)
		root = os.path.expanduser(root)
		name = filename or os.path.basename(url)
		dst = os.path.join(root, name)
		os.makedirs(root, exist_ok=True)
		if not utils.check_integrity(dst, md5):
			shutil.copyfile(src, dst)
		log(f"[preflight] dataset cache: {name} from Preflight's copy, not downloaded")
		return None

	download_url.__wrapped__ = orig
	utils.download_url = download_url

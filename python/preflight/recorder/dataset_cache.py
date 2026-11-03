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
DATASETS = ("CIFAR10", "CIFAR100", "MNIST", "FashionMNIST")


def resources() -> list[tuple[str, str, str]]:
	"""(dataset, url, md5) for every archive the cache holds: CIFAR-10/100, MNIST, FashionMNIST, as the
	image's torchvision names them (its first mirror for MNIST)."""
	from torchvision import datasets as D

	out = [("CIFAR10", D.CIFAR10.url, D.CIFAR10.tgz_md5), ("CIFAR100", D.CIFAR100.url, D.CIFAR100.tgz_md5)]
	for cls in (D.MNIST, D.FashionMNIST):
		mirror = cls.mirrors[0]
		out += [(cls.__name__, f"{mirror}{name}", md5) for name, md5 in cls.resources]
	return out


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

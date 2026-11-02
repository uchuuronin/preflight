"""scripts/check_unseen_hash.py: a published list matches the committed hash, a changed one doesn't."""

import hashlib
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("check_unseen_hash", ROOT / "scripts" / "check_unseen_hash.py")
chk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(chk)


def test_the_committed_hash_file_holds_one_hash():
	h = chk.committed_hash(chk.HASH_FILE.read_text())
	assert h == "be178f20fa8bb14f3b841b6f444a21ec2fc08db98ddddc1673d36b18de99b232"


def test_a_list_matches_only_byte_for_byte(tmp_path, capsys):
	body = b"salt 1234\n1. some/script@abc\n"
	lst = tmp_path / "unseen-scripts.md"
	lst.write_bytes(body)
	hf = tmp_path / "hash.txt"
	hf.write_text(f"SHA-256:\n\n{hashlib.sha256(body).hexdigest()}\n")
	assert chk.main([str(lst), "--hash-file", str(hf)]) == 0
	assert "match: this is the list" in capsys.readouterr().out
	lst.write_bytes(body.replace(b"\n", b"\r\n"))  # re-saved with other line endings
	assert chk.main([str(lst), "--hash-file", str(hf)]) == 1


def test_a_missing_list_is_not_a_mismatch(tmp_path):
	assert chk.main([str(tmp_path / "nope.md")]) == 2

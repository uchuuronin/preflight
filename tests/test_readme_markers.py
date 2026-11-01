"""The README keeps the block B's scorer writes the validation table into (scripts/score_validation.py
--readme). Without both markers, in order and once each, the scorer writes nothing and the submitted
README shows no results; a rewrite of the README (D's draft, the PM's merge) must keep them."""

from __future__ import annotations

from pathlib import Path

import pytest

README = Path(__file__).resolve().parents[1] / "README.md"
START, END = "<!-- validation:start -->", "<!-- validation:end -->"


def test_the_readme_has_one_validation_block_in_order():
	text = README.read_text()
	assert text.count(START) == 1, f"README needs exactly one {START}"
	assert text.count(END) == 1, f"README needs exactly one {END}"
	assert text.index(START) < text.index(END)


def test_the_markers_are_the_scorers_own():
	validate = pytest.importorskip("preflight.corpus.validate")
	if not hasattr(validate, "START"):
		pytest.skip("the scorer isn't on this branch yet (B's ws-b)")
	assert (validate.START, validate.END) == (START, END)

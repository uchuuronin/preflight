"""The final README's validation table is B's scorer output, written between two markers (Chiya, Nov 1).

These keep that hook working: the markers are there once each and in order, and, once B's scorer is on the
branch, writing a table through it changes only what is between them and the block starts with its header.
"""

from pathlib import Path

import pytest

DRAFT = (
	Path(__file__).resolve().parents[1] / "README.md"
)  # readme-final.md retired (PM, Nov 3): README.md is final
START, END = "<!-- validation:start -->", "<!-- validation:end -->"
SEED_STAMP = "Not scored yet: no full runs (seeded from the scorer with an empty manifest, Nov 1)."


def test_markers_are_there_once_each_and_in_order():
	text = DRAFT.read_text()
	assert text.count(START) == 1
	assert text.count(END) == 1
	assert text.index(START) < text.index(END)
	# The block sits under the Validation heading, before the next section.
	section = text.split("## Validation", 1)[1].split("\n## ", 1)[0]
	assert START in section and END in section


def test_the_scorer_rewrites_only_the_block(tmp_path):
	validate = pytest.importorskip("preflight.corpus.validate")
	assert (validate.START, validate.END) == (START, END)
	text = DRAFT.read_text()
	head, rest = text.split(START, 1)
	_, tail = rest.split(END, 1)
	copy = tmp_path / "README.md"
	copy.write_text(text)
	rows, _ = validate.build({"runs": []}, tmp_path, None)
	table = validate.markdown(rows, "test stamp")
	assert validate.into_readme(copy, table)
	assert copy.read_text() == f"{head}{START}\n{table}{END}{tail}"
	# Until Monday's run the block is exactly the scorer's output for no runs, so the first run only
	# changes what it measured. Fails when B changes the writer: reseed it from the scorer then.
	block = text.split(START, 1)[1].split(END, 1)[0]
	if SEED_STAMP in block:
		assert block == "\n" + validate.markdown(rows, SEED_STAMP)

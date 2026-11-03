"""The web offers Muse Code's fix on exactly the samples A's runner has a script for."""

import re
from pathlib import Path

from preflight.runner.record_jobs import SAMPLE_REFS

ROOT = Path(__file__).resolve().parents[2]


def test_the_web_list_matches_the_runners():
	text = (ROOT / "web/src/fix/candidates.ts").read_text()
	listed = re.search(r"FIXABLE_SAMPLES: readonly string\[\] = \[([^\]]*)\]", text).group(1)
	assert sorted(re.findall(r'"([^"]+)"', listed)) == sorted(SAMPLE_REFS)

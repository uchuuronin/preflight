"""The web's list of commands (web/src/app/command.ts) is A's (runner.sources.COMMANDS)."""

import re
from pathlib import Path

from preflight.runner.sources import COMMANDS, MAX_ARG_LEN, MAX_ARGS, MAX_COMMAND_CHARS

ROOT = Path(__file__).resolve().parents[2]


def test_the_web_offers_the_commands_the_runner_runs_with_its_limits():
	text = (ROOT / "web/src/app/command.ts").read_text()
	listed = re.search(r"export const COMMANDS = \[([^\]]*)\]", text).group(1)
	assert sorted(re.findall(r'"([^"]+)"', listed)) == sorted(COMMANDS)
	assert f"MAX_WORDS = {MAX_ARGS};" in text
	assert f"MAX_WORD_CHARS = {MAX_ARG_LEN};" in text
	assert f"MAX_COMMAND_CHARS = {MAX_COMMAND_CHARS};" in text

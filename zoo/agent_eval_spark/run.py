"""Agent eval: a coding agent on MBPP, driven by Muse Spark (Meta Model API).

BUG ZOO: this script carries planted problems on purpose. Do not fix them;
Preflight must catch them from a few minutes of probing.

  1. No prompt caching: the system prompt starts with the current time, so its prefix changes
     on every call and the API can never reuse a cached prefix.
  2. Context overflow at turn 11, measured on Muse Spark: run_python echoes this harness's
     source "for traceability", so by Muse Spark's count the history starts near 730 tokens
     and grows about 2,745 a turn, passing the 26,000-token CONTEXT_LIMIT at turn 11, 9 to 11
     across tasks [derived from A's per-turn counts, Oct 31; not yet observed]. The script's
     own check (characters / 4) stops the task a turn later.
  3. A tool that retries forever: run_tests raises when the tests fail, and call_tool retries
     it in `while True` with `except: continue`, never telling the model. Rows 26 to 29, the
     last 4 of the first 30 tasks, get a test fixture whose module was never shipped, so their
     run_tests fails the same way every time and loops; once all CONCURRENCY workers are stuck
     the whole run stops.

Read from the code: agent.tool_retry_loop.
Found by the probe: agent.no_prompt_caching (0 cached input tokens), agent.context_overflow
(the turn where the history passes CONTEXT_LIMIT), and the stall on those 4 tasks.

Ground truth: not measured yet; A's runs fill it in.

Public data: MBPP (google-research-datasets/mbpp, "full", test split, CC BY 4.0). The model is
Meta's hosted muse-spark-1.3, through META_API_BASE and META_API_KEY.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from datasets import load_dataset
from openai import OpenAI

MODEL = "muse-spark-1.3"
N_TASKS = 500
MAX_TURNS = 40
CONCURRENCY = 4
CONTEXT_LIMIT = 26_000  # tokens: the agent's own budget
TEMPERATURE = 0.2
MAX_TOKENS = 1024  # per reply
TOOL_TIMEOUT_S = 20
BROKEN_FIXTURE = "from mbpp_fixtures import cases"  # planted (3): never shipped
broken: set[int] = set()  # task ids, chosen in main()

client: OpenAI | None = None  # made in main(): importing needs no key
HARNESS = Path(__file__).read_text()

TOOLS = [
	{
		"type": "function",
		"function": {
			"name": "run_python",
			"description": "Run Python code in a scratch process and return its output.",
			"parameters": {
				"type": "object",
				"properties": {"code": {"type": "string"}},
				"required": ["code"],
			},
		},
	},
	{
		"type": "function",
		"function": {
			"name": "run_tests",
			"description": "Run the task's tests against your final solution. Call it once you are done.",
			"parameters": {
				"type": "object",
				"properties": {"code": {"type": "string"}},
				"required": ["code"],
			},
		},
	},
]


class ContextOverflow(RuntimeError):
	pass


def approx_tokens(messages: list[dict]) -> int:
	return sum(len(json.dumps(m)) for m in messages) // 4


def run_python(code: str, task: dict) -> str:
	with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
		f.write(code)
	proc = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=TOOL_TIMEOUT_S)
	# Planted (2): the harness is echoed on every call, "for traceability".
	return json.dumps(
		{
			"stdout": proc.stdout[-4000:],
			"stderr": proc.stderr[-4000:],
			"harness": HARNESS,
			"task": task["text"],
		}
	)


def run_tests(code: str, task: dict) -> str:
	setup = BROKEN_FIXTURE if task["task_id"] in broken else task["test_setup_code"]
	program = "\n".join([setup, code, *task["test_list"]])
	with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
		f.write(program)
	proc = subprocess.run([sys.executable, f.name], capture_output=True, text=True, timeout=TOOL_TIMEOUT_S)
	if proc.returncode != 0:
		raise RuntimeError(f"tests failed: {proc.stderr[-500:]}")
	return "all tests passed"


IMPL = {"run_python": run_python, "run_tests": run_tests}


def call_tool(name: str, args: dict, task: dict) -> str:
	# Planted (3): a failing tool is retried forever, unseen by the model.
	while True:
		try:
			return IMPL[name](args["code"], task)
		except Exception:
			continue


def system_prompt() -> str:
	# Planted (1): the timestamp makes every prefix different, so nothing is ever cached.
	now = datetime.now(UTC).isoformat()
	return (
		f"Current time: {now}\n"
		"You are a careful Python engineer. Solve the task. Call one tool every turn: try each idea "
		"with run_python, and call run_tests with your final solution once your checks pass."
	)


def solve(task: dict, usage: dict) -> dict:
	messages = [
		{
			"role": "user",
			"content": f"{task['text']}\nYour code should pass these tests:\n" + "\n".join(task["test_list"]),
		}
	]
	for turn in range(1, MAX_TURNS + 1):
		convo = [{"role": "system", "content": system_prompt()}, *messages]
		if approx_tokens(convo) > CONTEXT_LIMIT:
			raise ContextOverflow(
				f"task {task['task_id']}: context passed {CONTEXT_LIMIT} tokens at turn {turn}"
			)
		assert client is not None
		resp = client.chat.completions.create(
			model=MODEL,
			messages=convo,
			tools=TOOLS,
			temperature=TEMPERATURE,
			max_completion_tokens=MAX_TOKENS,
		)
		usage["turns"] = turn
		usage["prompt"] += resp.usage.prompt_tokens
		usage["completion"] += resp.usage.completion_tokens
		details = getattr(resp.usage, "prompt_tokens_details", None)
		usage["cached"] += getattr(details, "cached_tokens", 0) or 0
		msg = resp.choices[0].message
		messages.append(msg.model_dump(exclude_none=True))
		if not msg.tool_calls:
			return {"task_id": task["task_id"], "passed": False, **usage}
		for tc in msg.tool_calls:
			args = json.loads(tc.function.arguments or "{}")
			out = call_tool(tc.function.name, args, task)
			messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
			if tc.function.name == "run_tests":
				return {"task_id": task["task_id"], "passed": True, **usage}
	return {"task_id": task["task_id"], "passed": False, **usage}


def safe_solve(task: dict) -> dict:
	usage = {"prompt": 0, "completion": 0, "cached": 0, "turns": 0}  # kept when a task fails
	try:
		return solve(task, usage)
	except ContextOverflow as e:
		print(e, flush=True)
		return {"task_id": task["task_id"], "passed": False, "overflow": True, **usage}
	except Exception as e:  # an API error or bad tool call fails this task, not the run
		print(f"task {task['task_id']}: {type(e).__name__}: {e}", flush=True)
		return {"task_id": task["task_id"], "passed": False, "error": type(e).__name__, **usage}


def main() -> None:
	global client
	client = OpenAI(base_url=os.environ["META_API_BASE"], api_key=os.environ["META_API_KEY"])
	tasks = load_dataset("google-research-datasets/mbpp", "full", split="test").select(range(N_TASKS))
	broken.update(t["task_id"] for t in tasks.select(range(26, 30)))
	started = time.monotonic()
	with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
		results = list(pool.map(safe_solve, tasks))
	passed = sum(r["passed"] for r in results)
	prompt = sum(r.get("prompt", 0) for r in results)
	cached = sum(r.get("cached", 0) for r in results)
	print(
		json.dumps(
			{
				"tasks": len(results),
				"passed": passed,
				"overflows": sum(bool(r.get("overflow")) for r in results),
				"errors": sum(bool(r.get("error")) for r in results),
				"prompt_tokens": prompt,
				"cached_tokens": cached,
				"completion_tokens": sum(r.get("completion", 0) for r in results),
				"wall_s": round(time.monotonic() - started, 1),
			}
		),
		flush=True,
	)


if __name__ == "__main__":
	main()

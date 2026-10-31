"""A stand-in agent eval: 100 tasks, three turns each, one process, through OPENAI_BASE_URL.

Each turn appends the reply and a tool result, as agent loops do. A refused call (429) ends the
task, as a client that gives up would. Reads PREFLIGHT_TASK_INDICES when FAKE_AGENT_READS_INDICES
is set; FAKE_AGENT_THREADS runs that many tasks at once; FAKE_AGENT_ENV_OUT writes the
environment's variable names there.
"""

import json
import os
import urllib.error
import urllib.request

base = os.environ["OPENAI_BASE_URL"]
indices = os.environ.get("PREFLIGHT_TASK_INDICES")
tasks = (
	[int(i) for i in indices.split(",")]
	if indices and os.environ.get("FAKE_AGENT_READS_INDICES")
	else range(100)
)


def chat(messages):
	body = json.dumps({"model": "muse-spark-1.3", "messages": messages, "max_tokens": 50}).encode()
	req = urllib.request.Request(
		f"{base}/chat/completions",
		body,
		{
			"Content-Type": "application/json",
			"Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
		},
	)
	with urllib.request.urlopen(req, timeout=10) as r:
		return json.loads(r.read())


def system():
	if os.environ.get("FAKE_AGENT_TIMESTAMP_SYSTEM"):  # rebuilt every call, as D's agent script does
		import time

		return {"role": "system", "content": f"Current time: {time.time_ns()}"}
	return {"role": "system", "content": "you are a coder"}


def run(t):
	messages = [system(), {"role": "user", "content": f"task {t}"}]
	try:
		for turn in range(3):
			messages[0] = system()
			reply = chat(messages)["choices"][0]["message"]
			messages += [
				{"role": "assistant", "content": reply.get("content") or ""},
				{"role": "tool", "content": f"ran step {turn}"},
			]
	except urllib.error.HTTPError:
		pass


if os.environ.get("FAKE_AGENT_ENV_OUT"):
	with open(os.environ["FAKE_AGENT_ENV_OUT"], "w") as f:
		f.write("\n".join(sorted(os.environ)))
threads = int(os.environ.get("FAKE_AGENT_THREADS", "1"))
if threads > 1:
	from concurrent.futures import ThreadPoolExecutor

	with ThreadPoolExecutor(threads) as pool:
		list(pool.map(run, tasks))
else:
	for t in tasks:
		run(t)

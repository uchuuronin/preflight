"""A stand-in agent eval: 100 tasks, three turns each, one process, through OPENAI_BASE_URL.

Each turn appends the reply and a tool result, as agent loops do. A refused call (429) ends the
task, as a client that gives up would. Reads PREFLIGHT_TASK_INDICES when given.
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


for t in tasks:
	messages = [{"role": "system", "content": "you are a coder"}, {"role": "user", "content": f"task {t}"}]
	try:
		for turn in range(3):
			reply = chat(messages)["choices"][0]["message"]
			messages += [
				{"role": "assistant", "content": reply.get("content") or ""},
				{"role": "tool", "content": f"ran step {turn}"},
			]
	except urllib.error.HTTPError:
		continue

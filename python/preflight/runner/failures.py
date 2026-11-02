"""Which step a failure happened in, and how to get past it by hand (plan item 43, harness side).

	step, by_hand = describe("oom.eval")
	message = with_step("Ran out of GPU memory during the eval pass", "oom.eval")

Every failed job and every failed probe carries an error code (Probe.error_code's families and the
job codes in worker.error_code). This maps each family to the step it happened in and a manual
entry point: the command or the change that gets the visitor past it without Preflight's help.
Codes match by their longest known prefix; an unknown code falls back to its family, then to a
generic line, so a new code never goes out with no step.
"""

STEPS: dict[str, tuple[str, str]] = {
	# the job, before any probe
	"submission": (
		"checking the submission",
		"fix what the message says and submit again, or probe it on "
		"your machine: `pip install git+https://github.com/uchuuronin/preflight`, then "
		"`preflight run train.py`",
	),
	"submission.files_gone": ("reading the uploaded files", "upload the files again"),
	"submission.code_changed": (
		"checking the record's code",
		"check the record in again from the code as it is now",
	),
	"submission.knob_value": (
		"checking the setting",
		"pick a value inside the knob's range, or set it in your script and probe that",
	),
	"submission.too_deep": (
		"reading the script",
		"split the deeply nested code, or probe it on your machine with `preflight run train.py`",
	),
	"syntax_error": (
		"reading the script",
		"run `python -m py_compile train.py` (or `preflight map train.py`) to "
		"see the line, fix it, and submit again",
	),
	"override.unsupported": (
		"applying the setting to the script",
		"set the value in your script by hand and probe the script as it is",
	),
	"patch": ("applying the fix", "apply the patch by hand (`git apply fix.patch`) and probe the result"),
	"unsupported": (
		"choosing how to probe the script",
		"probe it on your machine with `preflight run train.py -- <your args>`",
	),
	"unsupported.agent_sampling": (
		"sampling the agent's tasks",
		"sample it on your machine with your key: "
		"`OPENAI_API_KEY=... preflight run agent.py --agent-max-usd 5`",
	),
	"probe.no_agent_calls": (
		"sampling the agent's tasks",
		"check the script calls the model through "
		"OPENAI_BASE_URL (or META_API_BASE), then `preflight run agent.py --agent-max-usd 5` on your machine",
	),
	"budget": (
		"staying inside the job's budget",
		"probe fewer settings, or probe on your machine with `preflight run train.py`",
	),
	"gpu.unavailable": ("waiting for a GPU", "submit again in a few minutes, or pick another GPU"),
	"gpu.fallback": ("moving to another GPU", "pin one GPU to keep every probe on it"),
	# inside a probe
	"missing_package.install": (
		"installing requirements.txt",
		"pin versions that install together, or install "
		"them yourself and run `preflight run train.py` on your machine",
	),
	"missing_package": (
		"importing the script's packages",
		"add the package (with its version) to "
		"requirements.txt, or run `preflight run train.py` where it's installed",
	),
	"gated_model": (
		"downloading the model",
		"run `preflight run train.py` on your machine, where your Hugging Face token can read it",
	),
	"download_blocked": ("downloading", "fetch it on your machine and run `preflight run train.py` there"),
	"file_not_found": (
		"opening a file the script reads",
		"add the file to the upload, or make the path relative to the script",
	),
	"oom.eval": (
		"the eval pass",
		"probe a smaller eval batch by hand: `preflight run train.py -- "
		"--eval_batch_size <n>`, or wrap eval in torch.no_grad()",
	),
	"oom": (
		"the training steps",
		"probe a smaller batch by hand: `preflight run train.py -- --batch_size <n>`",
	),
	"killed": (
		"the training steps (the process was killed, most often host memory)",
		"lower the loader's workers or the batch, then `preflight run train.py`",
	),
	"timeout": (
		"the probe's time limit",
		"run it yourself for a few steps (`python train.py <your args>`) to see "
		"where it's slow, or probe it on your machine with `preflight run train.py`",
	),
	"cuda_error": (
		"running on the GPU",
		"run it yourself with `CUDA_LAUNCH_BLOCKING=1 python train.py` to see the failing op",
	),
	"nccl": ("starting the GPUs' communication", "run it on one GPU first: `preflight run train.py`"),
	"dtype_unsupported": (
		"running on the GPU",
		"use a dtype this GPU supports (bf16 needs an A10, L4 or newer), or pick another GPU",
	),
	"user_error": (
		"running the script",
		"run it yourself (`python train.py <your args>`) to see the traceback",
	),
	"error": ("running the script", "run it yourself (`python train.py <your args>`) to see what happened"),
	"oom.host": (
		"host memory (the machine's RAM, not the GPU's)",
		"lower the loader's workers or load the data lazily, then `preflight run train.py`",
	),
	"dataloader_worker": (
		"loading data (a loader worker died)",
		"set num_workers=0 in the script to see the worker's own error, then `preflight run train.py`",
	),
	"shape_mismatch": (
		"the training steps (a tensor's shape doesn't fit)",
		"run it yourself (`python train.py <your args>`) to see which layer; a changed setting may not fit the model",
	),
	"script_exit": (
		"running the script (it exited before training)",
		"run it yourself (`python train.py <your args>`); arguments it needs go after -- in `preflight run train.py -- ...`",
	),
	"recorder": (
		"starting Preflight's recorder",
		"check the Python that runs probes imports torch (`python -c 'import torch'`), then `preflight run train.py`",
	),
	"too_few_steps": (
		"counting the training steps (the script ended after fewer than two)",
		"give the script more steps (raise max_steps or the epochs), then `preflight run train.py`",
	),
	"nan_divergence": (
		"the training steps (the loss went NaN or infinite)",
		"lower the learning rate or turn on gradient clipping, then `preflight run train.py`",
	),
	"cancelled": ("the probe, which was cancelled", "submit again when you want it measured"),
	# Preflight's own failure
	"internal": (
		"Preflight itself",
		"submit again; if it fails twice, probe it on your machine with `preflight run train.py`",
	),
}

FALLBACK = ("an unnamed step", "probe it on your machine with `preflight run train.py` to see what happens")


def describe(code: str | None) -> tuple[str, str]:
	"""(the step, the manual entry point) for an error code, by its longest known prefix."""
	parts = (code or "").split(".")
	for n in range(len(parts), 0, -1):
		hit = STEPS.get(".".join(parts[:n]))
		if hit:
			return hit
	return FALLBACK


def with_step(message: str, code: str | None, *, entrypoint: str | None = None) -> str:
	"""The message, naming its step and the way past it by hand (the script's own file name in the
	commands when known). Idempotent: a message that already carries them is returned as it is."""
	if "By hand:" in message:
		return message
	step, by_hand = describe(code)
	if entrypoint and entrypoint != "train.py":
		by_hand = by_hand.replace("train.py", entrypoint)
	return f"{message.rstrip('. ')}. Step: {step}. By hand: {by_hand}."

"""B's logging asks (Nov 3): a trainer's dataset rows (GRPO's run length), and an eval's whole request count and
scoring tokens (behind probes.eval_units)."""

import types

import pytest

pytest.importorskip("torch")

from preflight.recorder.runtime import Recorder


def test_a_trainer_s_dataset_rows_are_logged_with_its_planned_steps():
	rec = Recorder()

	class Trainer:
		def __init__(self):
			self.args = types.SimpleNamespace(num_train_epochs=1.0, max_steps=-1)
			self.train_dataset = list(range(7473))  # gsm8k's train split, as GRPO's demo holds it

		def create_scheduler(self, num_training_steps, optimizer=None):
			return "scheduler"

		def log(self, logs):
			return None

	rec._hook_trainer(types.SimpleNamespace(Trainer=Trainer))
	assert Trainer().create_scheduler(1868) == "scheduler"
	assert rec.run_length == {"planned_steps": 1868, "epochs": 1.0, "trainer_dataset_len": 7473.0}


def test_lm_eval_s_requests_and_tokens_are_counted_before_the_first_batch():
	rec = Recorder(eval_units=8)

	class HFLM:
		def _loglikelihood_tokens(self, requests, disable_tqdm=False):
			return ["scored"] * len(requests)

		def generate_until(self, requests, disable_tqdm=False):
			return ["text"] * len(requests)

	rec._hook_lm_eval_requests(types.SimpleNamespace(HFLM=HFLM))
	m = HFLM()
	reqs = [(("q", "a"), [1, 2, 3], [4]), (("q", "b"), [1, 2, 3], [5, 6])]
	assert m._loglikelihood_tokens(reqs) == ["scored", "scored"]
	m.generate_until(["p1", "p2", "p3"])
	assert rec.eval_requests == {"loglikelihood": 2, "tokens": 9, "generate": 3}


def test_off_nothing_is_hooked():
	rec = Recorder()
	assert rec.eval_units == 0 and rec.eval_requests == {}

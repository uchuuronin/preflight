"""A stand-in for TRL 0.14.0's GRPOTrainer with the seams the recorder hooks, shaped as 0.14 is:
__init__ keeps reward_funcs, num_generations and ref_model; the per-token log-probs are a closure
inside compute_loss (no method to patch); the reference runs under torch.inference_mode, through
ref_model or, with no ref_model (PEFT), the policy itself; generation goes through transformers'
generate; batches are fetched for every accumulation micro-step up front, as transformers 4.46
does. Small enough for a CPU test."""

import torch


class GRPOTrainer:
	def __init__(self, model, ref_model, reward_funcs, num_generations, prompts, accumulation=1):
		self.model, self.ref_model = model, ref_model
		self.reward_funcs = reward_funcs
		self.num_generations = num_generations
		self.prompts = prompts
		self.accumulation = accumulation
		self.opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

	def compute_loss(self, model, batch):
		g = self.num_generations
		ids = torch.tensor(batch).repeat_interleave(g, 0)
		seqs = model.generate(ids, max_new_tokens=6, do_sample=True, pad_token_id=0)

		def get_per_token_logps(m, input_ids):
			logits = m(input_ids).logits[:, :-1]
			return torch.log_softmax(logits.float(), -1).gather(-1, input_ids[:, 1:, None]).squeeze(-1)

		logps = get_per_token_logps(model, seqs)
		with torch.inference_mode():
			ref = get_per_token_logps(self.ref_model if self.ref_model is not None else model, seqs)
		ref = ref.clone()  # out of inference mode, as TRL's arithmetic does
		comps = seqs[:, ids.shape[1] :].tolist()
		rewards = torch.tensor(
			[
				sum(vals)
				for vals in zip(
					*[f(prompts=batch, completions=comps) for f in self.reward_funcs], strict=True
				)
			]
		)
		adv = rewards - rewards.view(-1, g).mean(1).repeat_interleave(g)
		return -(adv[:, None] * logps).mean() + 0.04 * (logps - ref).mean()

	def train(self, steps=1000):
		loader = torch.utils.data.DataLoader(self.prompts, batch_size=2, collate_fn=lambda b: b)
		it = iter(loader)
		for _ in range(steps):
			micro = [next(it) for _ in range(self.accumulation)]  # fetched up front
			for batch in micro:
				(self.compute_loss(self.model, batch) / self.accumulation).backward()
			self.opt.step()
			self.opt.zero_grad()

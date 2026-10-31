"""A stand-in for TRL's GRPOTrainer with the same seams the recorder hooks: __init__ keeps
reward_funcs and num_generations, _get_per_token_logps runs a model, generation goes through
transformers' generate. Small enough for a CPU test."""

import torch


class GRPOTrainer:
	def __init__(self, model, ref_model, reward_funcs, num_generations, prompts):
		self.model, self.ref_model = model, ref_model
		self.reward_funcs = reward_funcs
		self.num_generations = num_generations
		self.prompts = prompts
		self.opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

	def _get_per_token_logps(self, model, ids):
		logits = model(ids).logits[:, :-1]
		return torch.log_softmax(logits.float(), -1).gather(-1, ids[:, 1:, None]).squeeze(-1)

	def train(self, steps=1000):
		g = self.num_generations
		for step in range(steps):
			batch = self.prompts[(step * 2) % len(self.prompts) :][:2]
			ids = torch.tensor(batch).repeat_interleave(g, 0)
			seqs = self.model.generate(ids, max_new_tokens=6, do_sample=True, pad_token_id=0)
			with torch.no_grad():
				ref = self._get_per_token_logps(self.ref_model, seqs)
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
			logps = self._get_per_token_logps(self.model, seqs)
			loss = -(adv[:, None] * logps).mean() + 0.04 * (logps - ref).mean()
			loss.backward()
			self.opt.step()
			self.opt.zero_grad()

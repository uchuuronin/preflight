"""Ops check (Nov 2): lm_eval's own response lengths beside the recorder's new-token count for the same calls.

lm_eval's HF model generates through HFLM.generate_until (input_ids=, max_length=, its own stopping criteria);
this prints the length, in tokens, of each response lm_eval keeps, so the probe's generation stats, measured on
the same generate calls, can be checked against them (the gsm8k probe read 1.0 new tokens per sequence)."""

from lm_eval.api.instance import Instance
from lm_eval.models.huggingface import HFLM

FEW = (
	"Question: Tom has 3 apples and buys 2 more. How many apples does he have?\nAnswer: 3 + 2 = 5. The answer is 5.\n\n"
	"Question: A box holds 4 rows of 6 eggs. How many eggs are in the box?\nAnswer: 4 * 6 = 24. The answer is 24.\n\n"
)
QS = [f"{FEW}Question: Sara reads {n} pages a day for {d} days. How many pages does she read?\nAnswer:"
	for n, d in [(5, 3), (12, 4), (7, 7), (20, 2), (9, 6), (15, 3), (8, 8), (11, 5)]]  # fmt: skip

lm = HFLM(pretrained="HuggingFaceTB/SmolLM2-1.7B-Instruct", dtype="bfloat16", batch_size=8)
reqs = [Instance(request_type="generate_until", doc={}, idx=i,
	arguments=(q, {"until": ["Question:", "</s>", "<|im_end|>"], "do_sample": False, "max_gen_toks": 64}))
	for i, q in enumerate(QS)]  # fmt: skip
outs = lm.generate_until(reqs)
lengths = [len(lm.tok_encode(o)) for o in outs]
print(
	f"[lmeval-check] lm_eval's responses, tokens each: {lengths}; mean {sum(lengths) / len(lengths):.1f}",
	flush=True,
)
print(f"[lmeval-check] first response: {outs[0][:80]!r}", flush=True)

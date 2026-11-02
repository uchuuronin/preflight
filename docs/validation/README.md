# Validation tables

`table-nov2.md` is the scorer's output on every full run we had before Monday (Nov 2 overnight, B), and the same table sits between the validation markers in `README.md` and `docs/readme-final.md`. Monday's runs fill the rows marked "not yet run".

To regenerate it:

```
uv run --extra probe python scripts/score_validation.py tests/fixtures/validation/manifest_nov2.json \
	--corpus <kernel store> --out docs/validation/table-nov2.md --readme README.md
```

The kernel store is the cross-GPU top-up export (`b-topup-export-2`, five GPU types), which lives outside the repo. Without `--corpus` the cross-GPU row says "no corpus store given".

What each run in the manifest contributes:

- `finetune_lora_hf`: A's Oct 31 L4 LoRA probe record and its full run (`a-gt-lora-l4-5`). It calibrated the step band and the intervals, so it scores only the OOM and planted-problems rows.
- `agent_eval_spark`: the 30-task MBPP sample (Oct 31) against the whole 500-task run, which is those 30 plus the other 470 (`a-mbpp-full`, Nov 2, $6.2288 by the proxy; all 470 ended done).
- `grpo_trl`: the 50-step A10 run. Its record's only probe is that run, so it is calibration for every row and scores none.
- Knob response: C's stranger test at 4x CPU slowdown on a local build of main `cfcf83c` (the deployed site isn't reachable from B's container).

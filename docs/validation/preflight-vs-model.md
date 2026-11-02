# Preflight vs asking a model (Nov 2)

**The question:** would a strong model, given the same code, the command and the GPU but no probes, predict a job's peak memory, step time and out-of-memory as well as Preflight does?

**How it was asked.** Muse Spark (`muse-spark-1.3`) got exactly what Preflight's script map gets: the repo's files with the same suffixes, entrypoint first, up to the same 60,000 characters (`harness/mapper.py`), plus the command and the GPU, and nothing measured. It answered peak memory (max reserved, GiB), median step time after warm-up, and whether the job runs out of memory with a probability, each number with a 90% interval, as structured output. The prompt (`corpus/ask_model.py`, `ask-model@1`, hash `211e2e084350`) was committed at 11:17 ET (`d975cd4`) and pushed with its runner before A posted the unseen results (about 11:40, ed74e35). A's first full-run attempts had started at 11:07, but none of them measured anything (every one stopped before its first step), and the model only ever saw the repos' own code. Its answers were scored by the validation scorer's own rows on the same manifest as Preflight's (`corpus/ask_model_score.py`): the OOM row uses its probability against the same 0.5 alarm level, and a missing or malformed answer counts like Preflight's abstain. Run `b-ask-model-2`, 11 cases, $0.32.

## Scored with the validation rows (the frozen scorer, both sides)

| Metric | Target | Preflight | 95% interval | Hit | Asking Muse Spark | 95% interval | Hit |
|---|---|---|---|---|---|---|---|
| Peak memory error | Within 10% | not measured |  |  | not measured |  |  |
| OOM prediction | No missed OOMs; at most 1 false alarm per 10 scripts | 1 of 1 | 20% to 100% | hit, few runs | 1 of 1 | 20% to 100% | hit, few runs |
| Step time, same GPU | Within 15% | not measured |  |  | not measured |  |  |
| Interval coverage | 90% intervals cover at least 88% | 1 of 1 | 20% to 100% | hit, few runs | not measured |  |  |

Both sides are scored on the same manifest (`tests/fixtures/validation/manifest_unseen_nov2.json`), by the frozen scorer at `56c2141`. **There is almost nothing to score.** No unseen full run reached its first step, so only one run anywhere measured what these rows need: the zoo LoRA run that ran out of memory at eval. Both caught it: Preflight at 100% (a probe measured the OOM), the model at 97% from the code alone. Its memory isn't scored (a run that ran out never showed its full need) and its step time is calibration. Preflight's interval-coverage hit is the agent cost row, which the model wasn't asked. **So this table can't say which is more accurate: 1 run each, Wilson 20% to 100%.**

## What each said, case by case

| Case | GPU | Muse Spark: peak GiB (90%) | step s (90%) | P(OOM) | What the full run did | What Preflight's first attempt said |
|---|---|---|---|---|---|---|
| unseen-1a: torchtune LoRA 1.5B | L4 | 11.5 (8 to 17) | 0.9 (0.35 to 2.2) | 2.0% | stopped at import (torchao), 0 steps | failed at mapping (over 2 MB of Python), no record |
| unseen-1b: torchtune LoRA 7B | A100-SXM-80GB | 22 (17 to 30) | 2 (0.8 to 5) | 1.5% | stopped at import (torchao), 0 steps | failed at mapping, no record |
| unseen-2: LitGPT pretrain debug | L4 | 3.2 (1.5 to 7) | 0.09 (0.03 to 0.3) | 2% | stopped at import (lightning), 0 steps | no L4 in 20 min, no probe (retry: lightning missing) |
| unseen-3: open-r1 GRPO | A100-SXM-80GB | 42 (28 to 68) | 32 (15 to 70) | 10% | stopped at import (open_r1), 0 steps | probes: open_r1 missing (src/ layout) |
| unseen-5: lm-evaluation-harness (narrowed read: the runner refuses the repo) | L4 | 5.5 (4 to 8.5) | no training step | 1% | not run: refused | refused at submission (too large to read) |
| unseen-6: vLLM throughput (narrowed read: the runner refuses the repo) | L4 | 20.5 (19 to 22) | no training step | 5% | not run: refused | refused at submission |
| unseen-7: BEIR dense retrieval | L4 | 14 (10 to 19) | no training step | 15% | stopped at import (sentence_transformers), 0 steps | no L4 in 20 min, no probe (retry: sentence_transformers missing) |
| unseen-8: OGB arxiv GraphSAGE | L4 | 1.4 (0.8 to 2.5) | 0.08 (0.03 to 0.2) | 0.2% | stopped at import (torch_geometric), 0 steps | no L4 in 20 min, no probe (retry: torch_geometric missing) |
| unseen-9: airbench94 Muon | L4 | 4.2 (2.5 to 7) | 0.025 (0.014 to 0.045) | 1% | stopped at import (torchvision), 0 steps | probes: torchvision missing |
| finetune_lora_hf: zoo LoRA (Qwen2.5-1.5B), the eval OOM | L4 | 48 (32 to 90) | 2.8 (1.8 to 4.5) | 97% | ran out of memory at eval (step 200) | eval OOM measured by a probe: P(OOM) 100% |
| grpo_trl: zoo GRPO (TRL) (saw its own record: contaminated) | A10 | 21.59 (20.7 to 22.2) | 27.6 (21 to 34) | 6% | ran 50 steps, no OOM | calibration for every row; not compared |

**What this does show.** The model was asked what happens when each job runs, not whether it would run, so this is no miss of its own: it gave confident numbers, and a low chance of running out of memory, for nine unseen jobs that could not run as sent: every one stopped at a missing package or was refused before its first step. Preflight's first attempts, where a probe ran (3 and 9; the retries of 2, 4, 7 and 8 too), stopped at the same missing import the full run hit, and said so; where none ran (1, 5 and 6 refused or unmapped, 2, 4, 7 and 8 waiting for an L4), it gave no numbers at all. Neither side's numbers can be checked until the scripts run, which is the "after freeze" re-run with `probes.repo_install`.

## After the freeze: it rests on 2 runs

A re-ran the nine with `probes.repo_install` on in each ops run only (A's status file, "After freeze"). So far one reached its own work: **BEIR (unseen-7) ran out of memory on the A10 before its first step**, 21.5 GiB reserved on a 22 GiB card (`a-after-7-gt`, GitHub run 41491792689; ground truth `tests/fixtures/validation/after/ground_truth_unseen-7.json`). The others stopped before a step, so they measure nothing either side could be scored on. Scored the same way as above, by the frozen scorer, on `tests/fixtures/validation/after/manifest_after_freeze_nov2.json` (the zoo runs plus BEIR):

| Metric | Target | Preflight | 95% interval | Hit | Asking Muse Spark | 95% interval | Hit |
|---|---|---|---|---|---|---|---|
| Peak memory error | Within 10% | not measured |  |  | not measured |  |  |
| OOM prediction | No missed OOMs; at most 1 false alarm per 10 scripts | 1 of 2 | 9% to 91% | miss | 1 of 2 | 9% to 91% | miss |
| Step time, same GPU | Within 15% | not measured |  |  | not measured |  |  |
| Interval coverage | 90% intervals cover at least 88% | 1 of 1 | 20% to 100% | hit, few runs | not measured |  |  |

**Both missed BEIR's OOM, for different reasons.** The model read the code and gave it a 15% chance (peak 14 GiB, 90% 10 to 19, asked for the L4, a 24 GB card like the A10 it ran on); the run needed more than 21.5 GiB. Preflight's only record of BEIR is its first attempt, which waited 20 minutes for an L4 that never came and ran no probe, so it abstained, and an abstain counts as a miss. Preflight has no after-freeze check-in of BEIR, because production doesn't install a repo's dependencies until `probes.repo_install` goes on. **So this rests on 2 runs, one of them the zoo LoRA run already scored above, and it still can't say which is more accurate.** A Preflight check-in of BEIR with the flag on would be the fair test of Preflight's side. A's `runner/gt_record.py` can build a record from the ground-truth run's own recorder probe, but that isn't scored as Preflight's side: it is a probe from A's ops rig, run outside the product's path (with `--install-repo`, in the image the rig chose), and `gt_record.py` predicts nothing. That probe did run out of memory before its first step too (`recorder.probe.status` "oom" in the ground truth file), so a check-in's first probe with the flag on would most likely see the same OOM, but no check-in has shown it yet.

## With the zoo scripts' full runs: it rests on 6 scripts for OOM, 4 for memory and step time

Muse Spark was asked the same way about the zoo scripts B ran to their end after the freeze (`ask-model@1`, run `b-ask-model-zoo`, $0.11; each sample's own measured outputs left out, as the runner now does), for the GPU the full run got. Both answer files are scored together (`ask_model_nov2.json` for the zoo LoRA and BEIR, `ask_model_zoo_nov2.json` for the rest) with `corpus/ask_model_score.py` on the frozen scorer's rules, on `tests/fixtures/validation/zoo_gt/manifest_zoo_after_freeze_nov2.json`. Compared **by script on Modal's "A10"** (gnn and retrieval on an A10, vision and nanoGPT on an A10G; Preflight's `@L4` moved-GPU runs aside):

| Row | Scripts | Preflight | Asking Muse Spark |
|---|---|---|---|
| Peak memory within 10% | 4 (gnn, retrieval, vision, nanoGPT) | 2 of 4 (vision, nanoGPT) | 0 of 4 |
| Step time within 15%, same GPU | 4 (the same) | 1 of 4 (gnn) | 0 of 4 |
| 90% intervals that cover, on these four | 8 (memory and step) | 3 of 8 | 4 of 8 |
| OOM | 6 (the four, the zoo LoRA, BEIR); 2 ran out | LoRA caught, BEIR missed (no probe) | LoRA caught (97%), BEIR missed (15%) |

**Too few to rank.** On 4 scripts the hit counts' intervals overlap (Wilson 2 of 4: 15% to 85%; 0 of 4: 0% to 49%), the model's intervals cover slightly more often (4 of 8 against 3 of 8), the step-time difference is one script, and two of the four full runs got a different card from the one named. What the model said: nanoGPT 11.5 GiB (it peaked at 18.9 GiB) and 1.15 s a step (0.70); gnn 5.5 GiB (8.0) and 1.1 s (0.061); retrieval 1.3 GiB (2.3) and 0.06 s (0.031); vision 3.8 GiB (3.3) and 0.85 s (0.150). Its memory guesses land closer than its step times, which run 1.6 to 18 times too slow. Answers: `python/preflight/corpus/reports/ask_model_zoo_nov2.json`.

The model's answers: `python/preflight/corpus/reports/ask_model_nov2.json` (prompt `ask-model@1`, hash `211e2e084350`, $0.32). The `grpo_trl` case is left out of the scoring: its sample directory holds that run's own record, which the model read (its 27.6 s step is the measured p50).


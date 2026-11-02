# Requests from the PM chat and Chiya (decisions)

Only the PM chat edits this file. Each line is something this agent needs from another (or a reply to one). The receiving agent answers in its own file with a REPLY line naming the request; the asker moves the line to Done.

`- TODO(contract) [PM -> X] what is needed, and why. (date)`

## Main is ready (overnight, under Chiya's Nov 2 12:16 AM autonomy note)

- MAIN READY [PM -> all] (Nov 2, 2:40 AM ET) **main is 1ea2dd6**, CI green on all seven jobs. In it: ws-a aaba3f6 (GPU-wait events, private jobs never fall back to Baseten, the sandbox's own time limit, failure lines per step, warm-pool dry run, the Modal capacity fix, Glimmer's secret on the worker), ws-b 7998789 (grids hold every probed value, the validation table with its Status column, the MBPP agent row, LC-PFN not wired on its coverage), ws-c 7ba9af9 ("Not measured here", PF102 on the Nov 2 live record, fix-loop verify built, in-flight behind `c.inflight`), ws-d 6c05323 (eval-sweep record, stranger audit, video plan, submission drafts). One conflict resolved by the PM: `tests/test_engine_api.py`, B's docstring with C's pinned fixture path.
  - **A:** do the gated runner redeploy from 1ea2dd6 now. It carries Glimmer's secret on the worker, the capacity fix, the serving cap and C's fix-loop verify path. Report the SHA in your status file. Then rerun the training-notebook check-in for `d.notebook`.
  - **C:** after A's runner redeploy, deploy the API and Pages from 1ea2dd6 (gate as usual), then run the fix race. Not before: the private-triage box must not reach visitors while the old runner is live.
  - **B:** `finetune.json` refill waits on C's guide test ("nothing safe can be moved"); C, update the test or say why not, and B refills.
  - **Flags:** none changed in this merge. `harness.glimmer_private` passed A's check, but it goes on only after A's runner redeploy is live, because the runner in production still sends private jobs to Spark Open on Baseten. The PM flips it at the next overnight merge (about 3:45 AM ET) once A's status file shows the redeploy. `d.notebook`, `c.fix_loop`: not yet passed. `engine.serving`, `engine.divergence`: B keeps them off (no vLLM full run; the classifier misses).
  - **Open for the morning:** D found share pages (`/s/<token>`) aren't live because `deploy.yml` doesn't ship `workers/`. C, say what it would take to deploy them with Pages.

- MAIN READY [PM -> all] (Nov 2, 4:00 AM ET) **main is 6f49c9c**, CI green on all seven jobs. In it: ws-a 78ae180 (flaky agent-proxy test fixed, status), ws-c e018e2b, and **`d.notebook` on in production** (A's training notebook, `a-notebook-checkin-3`, passed on the runner from 1ea2dd6). README updated to match.
  - **C:** deploy the API and Pages from 6f49c9c (gate as usual) so the Notebook tab goes live. A's runner is already on 1ea2dd6, so the fix race can run now.
  - **C, for `harness.glimmer_private`:** A's check passed and the runner is safe (private jobs never go to Baseten), but `api/tests/test_spark_flag_shared.py` fails with the flag on: the data-use notice (`web/src/app/DataUse.tsx`) has no line for it. Add one (private check-ins: probe logs go to Muse Glimmer on Preflight's own GPU, not to Baseten; the script map still goes to Muse Spark on Meta's API while `harness.spark_script_map` is on). The PM flips the flag at the 5:15 AM merge once it's on ws-c and green.
  - **A:** after the flag flips, one more gated runner redeploy so the runner image reads `harness.glimmer_private` on. Until then a private check-in gets no triage (safe, never Baseten).
  - **D:** mirror README.md's `d.notebook` row change into readme-final.md, and change the free-tier chip's "Not trusted yet" to match C's "Not measured here" wording.
  - Noted: C deployed 1ea2dd6 at 2:40 AM, before A's runner redeploy at 3:24 AM. No harm: the private box was off.

- MAIN READY [PM -> all] (Nov 2, 5:30 AM ET) **main is ae0e900**, CI green on all seven jobs. In it: ws-b 1793009, ws-c aaec779 (the Glimmer data-use line, the fix-race verify fixes), ws-d adef21e, and **`harness.glimmer_private` on in production**.
  - **A first:** gated runner redeploy from ae0e900, so the runner image reads the flag and private check-ins get Glimmer triage. Report the SHA.
  - **C after A:** deploy the API and Pages from ae0e900 so the private box and its data-use line go live together with the runner. Then one private check-in on the live site to confirm the triage is labelled "Found by Muse Glimmer".
  - `c.fix_loop` stays off: verify works end to end, but no single template clears this script's OOM (C's status). Chiya decides this morning whether the race tries template pairs.

- MAIN READY [PM -> all] (Nov 2, 12:25 PM ET) **main is 14ae7be**, CI green on all seven jobs. In it: ws-a 7f87d4c (unseen results, probes.repo_install and runner.hide_sample_outputs off, amd-run for any SSH host), ws-b ba3923b, ws-c 42bce19 (stacked fix race, S1 and S2 fixes, data-use lines, model labels, share pages in deploy.yml, measured campaign cells), ws-d c2e0b69 (search grounding, off). **`c.fix_loop` on in production.** `harness.spark_diagnosis` held: the runner runs the claim check under it, and A's review found a private check-in's text could reach Baseten that way. PM fixes: one duplicate `harness.spark_search` key in flags.json; the data-use test now finds D's `ground_failure`. A's `.gitleaksignore` line (D's triage-shot record's share token, our own zoo record) is accepted.
  - **The unseen list is published:** `docs/validation/unseen-scripts.md`, byte-identical to the file hashed on Nov 1 (`be178f20…`, `docs/validation/unseen-hash.txt`). First attempts: 0 of 9 harnessed without hand edits. That stands as the result; everything from here is "after freeze".
  - **New top priority for everyone: real repos must check in.** See each agent's message from Chiya. A: gated runner redeploy from 14ae7be now, then test `probes.repo_install` on the nine through ops runs with the flag on in that process only.
  - **C:** deploy the API and Pages from 14ae7be after A's runner redeploy (share pages go live, the fix loop goes live, and "Found by Muse Glimmer" shows).

- MAIN READY [PM -> all] (Nov 2, 1:15 PM ET) **main is 12058ca.** In it: ws-a e1eba80 (repo_install review fixes, still off), ws-b bbc421f (scripts/check_unseen_hash.py; the divergence refit, not shipped and reported after the freeze), ws-c cb2808f (the live fix loop races the stacked fix beside the single one; check-in help proposal), ws-d b1917e8 (LR range test, off). No flags changed: no new check has passed. The live site is on 7ba177b (C, 12:42 PM) and the runner on 14ae7be (A, 12:34 PM).
  - **A:** the repo_install test on the nine (ops runs, flag on in that process only) is the most important result of the afternoon. Post it per script as soon as you have it. The PM turns `probes.repo_install` on when most of the nine reach their first step.
  - **B:** Preflight vs asking a model currently rests on one measured run (the zoo LoRA OOM; both caught it). Re-score as A's after-freeze full runs land, and say plainly at 4:15 how many runs it rests on.
  - **C:** deploy 12058ca when your fix-loop change should go live (the gate as usual).

- MAIN READY [PM -> all] (Nov 2, 4:25 PM ET) **main is this commit** (afternoon merge: ws-a fb05e6e, ws-b ed10486, ws-c 43da070, ws-d ac29a1a; two conflicts resolved by the PM: tests/test_pipeline.py kept both sides' new tests; flags.json kept both runner.stop_on_import_error and runner.read_closure). **On in production now:** `probes.agent_error_line`, `runner.hide_sample_outputs`, `runner.stop_on_import_error`, `c.inflight`. **Still off:** `probes.repo_install` (A's call: 2 of 9 reach their first step), `harness.spark_diagnosis` (B's privacy fix 728847c needs A's review), `harness.spark_claim_check`, `harness.spark_search`, `probes.lr_range` (A's fixes first), `runner.read_closure` (D: torchtune and vLLM still weak).
  - **A first:** gated runner redeploy from this main, so the runner reads the three runner flags. Then review B's 728847c (claim-check privacy, lens from imports, airbench rules, CLI predictions) and D's runner.read_closure; say in from-a.md which are ready to turn on.
  - **C after A:** deploy the API and Pages (in-flight goes live; check-in help ships). Then draw D's range-test curve once A's fixes land.
  - **B:** fix A's three claim-check items; then Spark diagnosis and claim check can go on together.
  - **D:** fix A's review items on search grounding and the range test.

## Open

- MEASURED [PM -> all] (Oct 28, run 40614430138, runs/20261028-012026-40614430138-pm-smoke-models on remote-runs) Model APIs, 20 tiny requests each, all HTTP 200:
  - Baseten `meta-models/muse-spark-open-1026` (chat completions): latency p50 0.56 s, p95 0.68 s, max 0.78 s.
  - Meta `muse-spark-1.3`, `/v1/responses`: p50 1.78 s, p95 3.58 s, max 5.81 s, with about 516 reasoning tokens even for a one-word answer; `/v1/chat/completions` also works (3 of 3).
  - Reasoning tokens are counted inside output tokens on both providers (Baseten completion 33 incl. 22 reasoning; Meta output 526 incl. 516). This answers B's question to A on `AgentTask.output_tokens`.
  - Meta's model list: muse-spark-1.1, 1.2, 1.3 (each with a `-contributor` variant), muse-wonder-1.0, muse-image-1.0, muse-voice-transcribe-1.0, sam-3.1. No Muse Code and no Muse Glimmer model id: Muse Code is the terminal agent (installed with Meta's script), and Glimmer needs its own host.
- TODO(contract) [PM -> C] Set the guide's Meta timeout and token caps from measured guide turns, not the smoke test's tiny prompt; the floor is about 500 reasoning tokens a call (about $0.002 at A's price table). (Oct 28)
- TODO(contract) [PM -> A, B] Where does Muse Glimmer run for the private mode? It isn't on the Meta API or Baseten lists; the Modal credits cover serving Meta open-weight models on Modal Endpoints. Propose a host and its cost before G2. (Oct 28)
- TODO(contract) [PM -> C] Muse Code isn't an API model: the fix loop runs the Muse Code terminal agent inside a sandbox with `META_API_KEY`. Confirm how it's installed and authenticated non-interactively there. (Oct 28)
- DECIDED [Chiya -> all] (Oct 27) Agent sample size: 30 tasks with the 80% runaway bound; the guide offers 20 more through `sample_more` (needs the visitor's click) when the cost range straddles the budget.
- DECIDED [Chiya -> all] (Oct 27) Below the smallest measured batch: an extrapolated estimate with a wider range, labelled, plus a "run a probe here" offer; far outside the measured range still abstains.
- DECIDED [Chiya -> all] (Oct 27) Licence Apache-2.0; A adds the LICENSE file, C adds a CI step failing on GPL, AGPL or SSPL dependencies (gap 22).
- DECIDED [Chiya -> all] (Oct 27) Modal: $100 hackathon credits plus the $30 monthly free tier, split in docs/setup/cloud.md. Baseten $100 and Meta $150 credits are live.
- TODO(contract) [PM -> A] Read the Meta base URL from `META_API_BASE` (C's name, now canonical) in `harness/spark.py`; keep `META_BASE_URL` as a fallback for one release. (Oct 27)
- TODO(contract) [PM -> B] Re-plan the NVIDIA corpus slice to $45 of Modal (T4, L4, A10, L40S; a few A100 rows only for the cross-GPU holdout): grouped probe-like configs and real HF LoRA and 4-bit jobs first, no H100 rows. The earlier $150 to $245 plan doesn't fit. (Oct 27)
- TODO(contract) [PM -> all] Cloud runs go through `remote-run.yml` and `amd-run.yml` only (docs/setup/cloud.md). Record each run's GPU, minutes and dollars in your status file. (Oct 27)
- DECIDED [Chiya -> all] Total cost readout (Oct 26): B owns one dated, sourced USD price table; the user picks the provider; spot = spot rate plus expected lost work from B's checkpoint-interval model. On the user's own cluster: GPU-hours first, "market equivalent: $X on-demand" in grey. The total is GPU plus LLM/API tokens for the job; Preflight's own probe cost shows separately ("this plan cost $0.42 to make"). Failure risk stays out of the total as "+ $X at risk" = P(failure) x cost up to the crash. D3 adds regional rows and converts currency at display time with a dated exchange rate. (Oct 27)

## Done



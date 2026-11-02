### The unseen scripts, first attempts (Nov 2)

Nine public scripts, one per job type, picked and hashed before the runs, each run once as a visitor would send it, on the frozen commit `56c2141` (A's status file, "Unseen validation runs (Nov 2)", has each run). The table above is the frozen scorer's output on those first attempts with the zoo runs we already had. In plain words:

- **Hit:** OOM prediction (1 of 1: the zoo LoRA run's eval OOM, predicted at 100%), agent run total cost (1 of 1: the 500-task MBPP run's $6.66 inside the 30-task sample's $5.90 to $13.71), planted problems caught (3 of 3), knob response (9 of 9 under 400 ms, p95 198 ms), interval coverage (1 of 1). All of these are **"hit, few runs"**: one to nine runs each, so none is a confident hit.
- **Missed:**
  - **Unseen scripts harnessed: 0 of 9 without hand edits** (Wilson 95% 0% to 30%), against a target of 8 of 9. Three were refused before any probe (lm-evaluation-harness and vLLM as larger than Preflight reads; torchtune as too much Python to map), and the other six stopped at an import the probe image lacks (lightning, open_r1's src/ layout, smolagents, sentence_transformers, torch_geometric, torchvision): none ships a requirements.txt. The scorer's row reads 0 of 6, because it counts only scripts that produced a record; the three refused before a record existed are misses too (a scorer gap these runs found, logged for after scoring).
  - **Step time on a GPU the corpus never ran:** 57% within 20% (55% to 59%), against 90%. Simulated on held-out corpus pairs.
  - **Divergence warning:** AUROC 0.66 (0.56 to 0.76), against 0.85. Simulated on held-out short runs; not used in the product.
  - **GNN + GBDT vs GBDT alone:** the paired gain's interval includes zero, so the GBDT alone is used. Simulated.
- **Not measured:** peak memory, same-GPU step time, rollout time, serving throughput and the cost of one Preflight run. **No unseen full run reached its first step**: each stopped at the same missing import as its check-in, or was refused, so there is no measured peak memory, step time or out-of-memory to score them against. The zoo full runs we have were used to calibrate these rows, so they don't score them either.

What happens next is scored apart, as "after freeze": A has built `probes.repo_install` (off), which installs a repo's declared dependencies and known imports when it has no requirements.txt; the nine are re-run with it after the runner is redeployed, and those results never change the rows above.

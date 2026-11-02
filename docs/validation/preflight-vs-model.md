# Preflight vs asking a model (Nov 2)

**The question:** would a strong model, given the same code, the command and the GPU but no probes, predict a job's peak memory, step time and out-of-memory as well as Preflight does?

**How it was asked.** Muse Spark (`muse-spark-1.3`) got exactly what Preflight's script map gets: the repo's files with the same suffixes, entrypoint first, up to the same 60,000 characters (`harness/mapper.py`), plus the command and the GPU, and nothing measured. It answered peak memory (max reserved, GiB), median step time after warm-up, and whether the job runs out of memory with a probability, each number with a 90% interval, as structured output. The prompt (`corpus/ask_model.py`, `ask-model@1`, hash `211e2e084350`) was committed at 11:17 ET (`d975cd4`) and pushed with its runner before A posted the unseen results (about 11:50). A's first full-run attempts had started at 11:07, but none of them measured anything (every one stopped before its first step), and the model only ever saw the repos' own code. Its answers were scored by the validation scorer's own rows on the same manifest as Preflight's (`corpus/ask_model_score.py`): the OOM row uses its probability against the same 0.5 alarm level, and a missing or malformed answer counts like Preflight's abstain. Run `b-ask-model-2`, 11 cases, $0.32.

<!-- side-by-side:start -->
(filled from the scorer once the unseen first attempts are scored)
<!-- side-by-side:end -->

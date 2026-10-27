# Contract requests

Need something from another plane, or a contract change? Add a line here and
tell Chiya. Never patch another agent's code. Format:

`- TODO(contract) [from -> to] what you need, and why. (date)`

Close a request by moving it to Done with the commit that settled it.

## Open

- TODO(contract) [A -> all] v0.1 adds `Grid.scale`: grid values are stored as value / scale in float16 because float16 overflows at 65,504 (bytes and long run times). B writes it with `encoding.choose_scale`; C reads it with `decodeF16(data, n, grid.scale)`. Please confirm before the noon freeze. (Oct 27)
- TODO(contract) [A -> B] Golden records carry fixture predictions (`golden-fixture@0`). Replace them with real engine output once `predict()` lands; `scripts/make_golden.py` is A's, so send the call you want it to make. (Oct 27)
- TODO(contract) [A -> B] Project run time from `StepTime.p50_s`, not `mean_s`: the Monte Carlo in docs/decisions/probe-plan.md puts the median of 5 timed steps within 15% in 99.3% of trials, the mean of 3 in 89%. (Oct 27)
- TODO(contract) [A -> B] Memory probes now come in threes (current, half batch, half sequence) so the fit can separate bs·s from the bs·s² attention term. Probe roles are in `probes/plan.py`; the probe ids are p1..p3 in that order. (Oct 27)
- TODO(contract) [A -> B] `recorder/gpus.py` carries a minimal peak-FLOPs table (bf16, fp16, fp32, tf32) for MFU. Once your GPU spec table lands, expose it as a pure function and the recorder will import it instead. (Oct 27)
- TODO(contract) [A -> C] Probe failures now carry `Probe.error_code` (families listed in schema/record.py, e.g. oom.eval, missing_package, download_blocked, gated_model). Please key UI copy as t('error.<code>') and t('error.<code>.hint'). (Oct 27)
- TODO(contract) [A -> C] `QueueFull` from the job store should map to 429 with Retry-After = its `retry_after_s`; `StoreUnavailable` to 503 retryable. (Oct 27)
- TODO(contract) [A -> Chiya] Agent sample size, 30 or 50 tasks: see D3 in docs/decisions/probe-plan.md. Default is 30 until decided. (Oct 27)

## Done

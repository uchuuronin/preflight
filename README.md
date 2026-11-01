# Preflight

A flight simulator for ML jobs. Preflight measures a few minutes of your real
job, builds a model of how it behaves, and lets you fly the whole run, or a
whole campaign of runs, before you spend real GPU hours or API dollars.

Built during the Meta Global AI Developer Hackathon (Oct 27 to Nov 4, 2026).

## Validation

The table between the markers below scores each validation target, hit or miss, and says so when
a row has nothing to score yet. It is written by `scripts/score_validation.py --readme README.md`
from one manifest of full runs; edit the manifest, not the table. The knob response row's timings
come from a stranger test run at 4x CPU slowdown (`KNOB_MS_OUT=out/knob_ms.json node
web/e2e/stranger.mjs`), copied into the manifest's `knob_ms` list.

<!-- validation:start -->
Not scored yet: the full runs are scored on Mon Nov 2.
<!-- validation:end -->

## Develop

```bash
uv sync                                   # Python 3.11 env with dev tools
npm ci                                    # schema codegen (json-schema-to-typescript)
uv run pytest -q                          # tests
uv run ruff check . && uv run ruff format --check .
uv run python -m preflight.schema.generate   # after any schema change
```

Probes need PyTorch: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`
for CPU, or the CUDA/ROCm wheel on a GPU box.

## Verify a fix in your own CI

The verify GitHub Action fails a job unless a fix improved what it was for and left the
training run's meaning alone: the same effective batch, tokens seen and learning-rate
schedule, throughput no worse than 90% of before, and the same fixed-seed loss within a small tolerance. It runs
Preflight's own verify code (`api/preflight_api/verify.py`) on two flight records, with
Python's standard library only and no keys.

```yaml
# .github/workflows/preflight-verify.yml in your repository
on: pull_request
jobs:
  verify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: uchuuronin/preflight/.github/actions/verify@main  # pin a commit hash for production
        with:
          before: preflight/before.record.json  # the run before the fix
          after: preflight/after.record.json    # the verify run after it
          metric: memory.peak                   # what the fix was for
          # may-change: throughput_floor        # checks this fix is allowed to move
          # keep-eval: "true"                   # the fix touches evaluation; it must still run
          # no-grad: "true"                     # the fix runs evaluation without gradients
```

The `status` output is `passed`, `failed` or `error` (a check couldn't be measured, which never
counts as a pass), and the job summary lists every check with its before and after values. The
records are flight record JSON files: the API returns any record by its share link at
`GET /v1/records/{share_token}`. Another repository can use the action only while this one is public, or, while
it is private, from repositories of the same owner with Actions access to it allowed.

## Layout

See [CONTRACT.md](CONTRACT.md) for the shared interfaces and who owns which path.

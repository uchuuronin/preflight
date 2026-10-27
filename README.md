# Preflight

A flight simulator for ML jobs. Preflight measures a few minutes of your real
job, builds a model of how it behaves, and lets you fly the whole run, or a
whole campaign of runs, before you spend real GPU hours or API dollars.

Built during the Meta Global AI Developer Hackathon (Oct 27 to Nov 4, 2026).

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

## Layout

See [CONTRACT.md](CONTRACT.md) for the shared interfaces and who owns which path.

# Relabelled copies

The frozen scorer (56c2141) reads records up to contract 0.6.2. These inputs are of a later contract, so they are copied with `schema_version` set to 0.6.2. Contracts 0.6.3 to 0.6.5 are additive (check-in help's job fields, `Grid.independent_of`, `resend_of`), and the frozen schema ignores fields it doesn't know, so what the scorer reads is unchanged; the original is at its repo path.

| Original | Its version | Copy |
| --- | --- | --- |
| `tests/fixtures/validation/amd/pretrain_nanogpt_mi300x.record.json` | 0.6.3 | `files/tests/fixtures/validation/amd/pretrain_nanogpt_mi300x.record.json` |

# Validation freeze

- **Frozen commit:** `56c2141` (main, Nov 2 2026, 05:32 ET), also on branch `freeze/validation`.
- **Declared:** Mon Nov 2, 06:42 ET, on Chiya's go, before any unseen script was opened by an agent.
- **What is frozen:** the engine, the calibration table `50ea0fb14a`, the harness and the runner as of that commit. Unseen scripts are predicted and harnessed from this commit only. Fixes made after the list is opened are logged as findings and scored separately as "after freeze"; they never change the first-attempt rows.
- **Unseen list:** its SHA-256 `be178f20fa8bb14f3b841b6f444a21ec2fc08db98ddddc1673d36b18de99b232` was committed Nov 1, 01:46 ET (`4ca3704`, `docs/validation/unseen-hash.txt`). The list itself is published after the runs, so anyone can check it hashes to that value.
- A `validation-freeze` git tag on the same commit is added from Chiya's machine (the session's proxy refuses tag pushes).

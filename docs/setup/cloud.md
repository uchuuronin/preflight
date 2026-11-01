# Cloud setup: Modal, Baseten, the Meta Model API and AMD

The agents' sandboxes can reach GitHub and package registries, nothing else. So no agent holds a key
or calls a cloud directly. Keys live in GitHub Actions secrets; an agent pushes code, starts a
workflow with `gh`, and reads the results back from the `remote-runs` branch with `git`. Keys never
appear in a chat, a commit or a flight record.

| Service | Used for | Where the key lives | Budget |
| --- | --- | --- | --- |
| Modal | Probes, zoo ground truth, the NVIDIA corpus slice, the API | GitHub secrets `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | $100 hackathon credits + $30 a month free |
| Baseten | Muse Spark Open (triage, labelling) | GitHub secret `BASETEN_API_KEY`, copied into Modal secret `preflight-baseten` | $100 credits |
| Meta Model API | Muse Spark (script map, guide), Muse Code, Muse Glimmer | GitHub secrets `META_API_KEY`, `META_API_BASE`, copied into Modal secret `preflight-meta` | $150 credits |
| The API's own keys | Signing session cookies; proving traffic came through the Cloudflare site | GitHub secrets `PREFLIGHT_SESSION_KEYS`, `PREFLIGHT_EDGE_SECRET`, copied into Modal secret `preflight-api` | none |
| AMD Developer Cloud | ROCm probes, the AMD corpus slice, the AMD holdout | GitHub secrets `AMD_SSH_KEY`, optional `DO_API_TOKEN` | $40, about 20 MI300X-hours at $1.99 (the droplet page, Nov 1) |

Probe containers never mount `preflight-meta` or `preflight-baseten` (gap 6). Only the API and the
pipeline's own functions do.

## Chiya: one-time setup

Run these one after another. Every `gh` command needs the GitHub CLI signed in (`gh auth login`);
the web alternative is the repo's Settings > Secrets and variables > Actions > New repository secret.

**1. Modal token and spending limit**

1. On modal.com, open Settings > API Tokens > New token. It shows a token id (`ak-...`) and secret (`as-...`).
2. In Modal's usage or billing settings, set a workspace spending limit of $130 if offered, so nothing is charged past the credits.
3. Save both values as secrets (each command asks you to paste the value):

```
gh secret set MODAL_TOKEN_ID -R uchuuronin/preflight
gh secret set MODAL_TOKEN_SECRET -R uchuuronin/preflight
```

**2. Baseten and Meta keys**

```
gh secret set BASETEN_API_KEY -R uchuuronin/preflight
gh secret set META_API_KEY -R uchuuronin/preflight
gh secret set META_API_BASE -R uchuuronin/preflight
```

`META_API_BASE` is `https://api.meta.ai/v1` (from the dashboard's cURL example; Meta uses the Responses API format).

**3. Two random keys for the API's sessions**

C's API signs visitors' session cookies, and only accepts traffic that comes through the Cloudflare
site, checked with a shared secret. Make two random values in your conda env (run each, copy its output):

```
python -c "import secrets; print(secrets.token_urlsafe(48))"
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Save the first as `PREFLIGHT_SESSION_KEYS` and the second as `PREFLIGHT_EDGE_SECRET`, as repository
secrets like the others. When the Cloudflare Pages project is created, set `PREFLIGHT_EDGE_SECRET`
there too, with the same value.

**4. Copy the keys into Modal, then check everything works**

`main` must be the repo's default branch first (Settings > Branches), because GitHub only offers
manual workflows that exist on the default branch.

```
gh workflow run modal-setup.yml -R uchuuronin/preflight
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=pm-smoke-models -f cmd="python scripts/cloud/smoke_models.py --n 20"
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=pm-smoke-l4 -f cmd="modal run scripts/cloud/smoke_modal.py --gpu L4"
```

The model smoke test lists the model ids each provider offers and measures latency over 20 tiny
requests (p50, p95, max). That grounds the guide's timeout and fixes the Muse Spark model id. The L4
smoke test records the memory PyTorch actually reports on an L4, which B needs, and an achieved bf16
matmul rate. Both cost cents.

**5. AMD droplet (only when a ROCm run is planned; it bills every hour it exists)**

On Windows PowerShell, make a key with no passphrase (press Enter at both prompts):

```
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\preflight_amd
Get-Content $env:USERPROFILE\.ssh\preflight_amd -Raw | gh secret set AMD_SSH_KEY -R uchuuronin/preflight
Get-Content $env:USERPROFILE\.ssh\preflight_amd.pub
```

Then in the AMD Developer Cloud panel:

1. Add the printed public key as an SSH key.
2. Create a GPU droplet: 1x MI300X, the image with ROCm and PyTorch preinstalled, your SSH key.
3. Note its public IPv4 and its droplet id (the number in the droplet's page URL).
4. Optional: if the panel offers API tokens, create one with write access and save it, so a run can destroy the droplet when it finishes:

```
gh secret set DO_API_TOKEN -R uchuuronin/preflight
```

Destroy the droplet when a session ends; a powered-off droplet still bills.

## Agents: running something in the cloud

**Ask first.** Don't use the keys recklessly. Before running anything that spends tokens or GPU
time (Modal, Muse Spark, Muse Code, Baseten, the AMD droplet), run it by Chiya and the PM chat in
your chat: what it runs, on which GPU or model, for how long, and the expected cost. Start it only
after Chiya says yes. This holds for every run, smoke tests included.

Name runs `<agent>-<what>` (for example `a-zoo-lora`, `b-corpus-t4`). Use the cheapest GPU that
fits (T4 $0.59/h, L4 $0.80/h); anything above L40S needs Chiya's OK in chat first.

Start a run on Modal or a model API, from your own branch:

```
gh workflow run remote-run.yml -R uchuuronin/preflight -f name=a-smoke-t4 -f ref=ws-a -f timeout=15 -f cmd="modal run scripts/cloud/smoke_modal.py --gpu T4"
```

`cmd` must start with `modal run `, `modal deploy `, `python scripts/` or `python -m preflight.`,
with no shell operators. A `modal deploy` leaves the app running and billing until it is stopped, so it
always needs Chiya's yes first, with how and when it will be stopped. Anything your command writes to `./out/` comes back with the log.

Wait for it and read the results:

```
gh run list -R uchuuronin/preflight --workflow remote-run.yml -L 5
gh run watch <run id> -R uchuuronin/preflight --exit-status
git fetch origin remote-runs
git ls-tree --name-only origin/remote-runs runs/ | tail -5
git show origin/remote-runs:runs/<folder>/log.txt
git show origin/remote-runs:runs/<folder>/result.json
```

Run on the AMD droplet (Chiya gives you the IP and droplet id when it exists):

```
gh workflow run amd-run.yml -R uchuuronin/preflight -f name=b-amd-slice -f host=<ip> -f ref=ws-b -f timeout=120 -f cmd="python -m preflight.corpus.sweep --backend rocm" -f destroy_droplet_id=<id>
```

Leave `destroy_droplet_id` empty when more runs follow in the same session.

## Budget for Modal: $130

$100 of hackathon credits (valid for the hackathon; Chiya redeemed them with Modal's form) plus the
$30 monthly free tier. Check the expiry date of the $100 on Modal's billing page: if it ends before
judging (Nov 5 to 8), the demo runs on November's free $30 instead.

| Use | Budget | Roughly, at T4 $0.59/h, L4 $0.80/h, A100-80GB $2.50/h |
| --- | --- | --- |
| Smoke tests | $1 | a few L4 minutes |
| Bug-zoo ground truth (full runs of the planted-bug scripts) | $15 | 19 L4-hours |
| A's end-to-end pipeline and probe development | $15 | about 150 probe jobs of 5 minutes |
| B's NVIDIA corpus slice | $45 | T4, L4, A10 and L40S; a few A100 rows only for the cross-GPU holdout |
| Validation full runs (Mon Nov 2) | $20 | the unseen scripts on L4, 7B jobs on A100 |
| Public demo, Nov 5 to 8 | $25 | about $6 a day of visitors' own probes |
| Reserve | $9 | |

The corpus plan assumed $150 to $245, so B re-plans it to $45: grouped probe-like configs and real
Hugging Face LoRA and 4-bit jobs on T4 and L4 first, then A10 and L40S; no H100 rows. Other GPUs
come from the cross-GPU model with wider intervals, labelled as such. If outputs look good by G1
(Thu Oct 29), Chiya decides whether to add money for a larger corpus.

## Rules

- Ask Chiya and the PM chat before any run that spends tokens or GPU time (see "Ask first" above).
- Never paste a key into a chat, a commit, a test or a record. `.env.example` lists the names only.
- `remote-runs` is written by workflows only; never push to it by hand.
- Every run records its spend in your `docs/status/<agent>.md` (GPU, minutes, dollars), so the
  budget above is checked from data, not memory.

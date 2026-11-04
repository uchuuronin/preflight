# Shot 9b on Chiya's laptop: the exact steps

By D, Tue Nov 3, for filming on Wednesday (the PM, Nov 3, 09:03 and 09:38). Shot 9b is in
`docs/demo/video-plan.md`: `preflight watch` runs a real script in a terminal, the cockpit follows
its live record beside it, and an alert appears in both when the run leaves its predicted band.

**The machine:** Chiya's laptop, Windows, NVIDIA GeForce RTX 4070 Laptop GPU (8 GB).
**The rules:** every path is on E:, nothing on C:; every command is typed in one PowerShell window,
one after another, no `.ps1`, `.bat` or `.sh` scripts.

## Native Windows: yes (decided Nov 3)

On Tue Nov 3 morning, `preflight run` **didn't work on native Windows**: D's checks on a GitHub
Windows runner stopped at `os.geteuid`, then the time-zone database, then `os.killpg`; with those
and four more stubbed, every probe failed with `WinError 10106`, because the probe's cleaned-up
environment dropped Windows' own variables. Nine gaps in all (the list is in
`docs/requests/from-d.md`). A fixed them behind guards, and **with A's fix on `main` (`6d40cd40`),
the same runner with nothing stubbed measured all three probes, wrote a record with its memory range
(a CPU gives no step-time range), and `preflight watch` followed the script for 2,000 steps, then
400 with step 11's change (GitHub run 41696495753) [measured]. So native Windows it is.** CUDA on
Windows is the part no one has run yet.

Step 4 installs whatever is on `main`, so it gets the fix. If step 9 still fails on Wednesday (see
its failure cases), the WSL2 route at the end of this page keeps everything on E: too.

## What D checked, and where it differs from your laptop

| Where | What | Result |
| --- | --- | --- |
| An A10 GPU on Modal, Linux, Nov 3 (`d-watch-gpu-9b-2`, GitHub run 41650324681) | The same flow from `runner/watch_gpu_app.py`: `preflight run` on the unchanged script, then `preflight watch` on it unchanged and with step 11's change. Differences: 200,000 steps, not 60,000; the repository's `flags.json` (a pip install reads none, so every flag is off; the run and watch steps use none of them); each watch stopped after 150 s | With the change, both alerts fired: `step 201: peak memory 0.40 GiB, above the predicted 0.27–0.36 GiB`, then `step 401: 0.0033 s a step, above the predicted 0.00125–0.00275 s`. Unchanged, only a note ("faster than the predicted"), no alert. About 1,240 steps a second unchanged, 300 with the change [measured] |
| A CPU, Linux, Nov 3 (D's sandbox) | Steps 2 to 13 from a fresh venv, with 18,000 steps, the changed run at 400 | Every step works; no alert, because a CPU record has no step-time range ("the probes ran without a known GPU") [measured] |
| A GitHub Windows runner, CPU, Nov 3 (`d-windows-watch.yml`, seven runs) | The same flow in PowerShell 7 (not 5.1) on drive D:, with CPU torch, the script copied from the checkout, 2,000 steps and 400 changed | `preflight run` failed as above until all nine gaps were stubbed; `preflight watch` ran. Then, with nothing stubbed, D's own copy of the fixes (since withdrawn; run 41683645239) and A's (run 41684996973, both copies side by side) each measured all three probes, and `preflight watch` followed 2,000 steps [measured] |
| Your card | An RTX 4070 Laptop isn't in Preflight's GPU table | Step time still gets its range from the probes on that card (checked by relabelling an L4 record as it) [measured]; memory's range on it wasn't checked (that L4 record's eval ran out of memory, so its memory abstains anyway); moving the GPU knob to or from the card can't be predicted until B adds it to the table |

Not checked anywhere: CUDA on Windows, and Chrome on Windows following a file as it changes.
Both are the first things to look at if a step misbehaves.

## Once: what to have ready

- **Power:** plug in the charger and set Windows' power mode to Best performance (a laptop GPU
  slows down on battery).
- **NVIDIA's driver.** In PowerShell, `nvidia-smi` should list the RTX 4070 Laptop GPU and a
  "CUDA Version" of 12.8 or higher. If it's lower, update the driver from NVIDIA's site.
- **E: is NTFS and plugged in** (if it's an external drive). In Explorer, E:'s Properties show
  the file system.
- **Python 3.12.10 or 3.11.9 on E:** (the last versions with a Windows installer; Preflight needs
  3.11 or 3.12). From python.org, the "Windows installer (64-bit)". Choose "Customize
  installation"; untick "py launcher" and "for all users" (both install to C:); on the next page
  set "Customize install location" to `E:\Python312` (or `E:\Python311`). The installer still keeps
  its own cache (tens of MB) and Start-menu shortcuts in your user folder on C:; nothing else of
  Python goes there. Below, change `E:\Python312` to the folder you chose.
- **Chrome or Edge.** They follow the live record as it changes; other browsers read it once.
- **About 12 GB free on E:** CUDA torch is about 3 GB to download and 4 to 5 GB installed.

## The steps

Open **PowerShell** (not as administrator). Type each line, press Enter, and wait for it to finish
before the next.

### 1. Keep everything on E:

```powershell
New-Item -ItemType Directory -Force E:\pf9b\tmp, E:\pf9b\pip-cache, E:\pf9b\cuda-cache, E:\pf9b\sandbox, E:\pf9b\film
```

```powershell
$env:TEMP = 'E:\pf9b\tmp'; $env:TMP = 'E:\pf9b\tmp'; $env:PIP_CACHE_DIR = 'E:\pf9b\pip-cache'; $env:CUDA_CACHE_PATH = 'E:\pf9b\cuda-cache'; $env:PREFLIGHT_SANDBOX_DIR = 'E:\pf9b\sandbox'; $env:PYTHONUTF8 = '1'
```

```powershell
Set-Location E:\pf9b
```

**You should see:** a list of the five new folders, then nothing, then the prompt `PS E:\pf9b>`.
These settings last only for this window: **if you close it, type all three lines again** in the
new one.

### 2. A fresh venv

```powershell
E:\Python312\python.exe -m venv E:\pf9b\venv
```

```powershell
E:\pf9b\venv\Scripts\python.exe --version
```

**You should see:** `Python 3.12.10` (or `3.11.9`). Every later command calls the venv's own
`python.exe` or `preflight.exe` by its full path, so you never "activate" the venv (Windows blocks
the activation script by default).

**If it fails:** "not recognized" means Python isn't in that folder; find `python.exe` on E: and use
that path.

### 3. torch with CUDA

```powershell
E:\pf9b\venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
```

```powershell
E:\pf9b\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
```

**You should see:** a download of about 3 GB, then (it can take a minute or two the first time) a
line like `2.x.x+cu128 True NVIDIA GeForce RTX 4070 Laptop GPU`.

**If it fails:**
- `False no GPU`, or a version ending `+cpu`: the CPU build got in. Type
  `E:\pf9b\venv\Scripts\python.exe -m pip uninstall -y torch` and repeat the install line.
- "No matching distribution": go to pytorch.org, "Get Started", pick Stable, Windows, Pip, Python,
  and the highest CUDA no newer than the "CUDA Version" `nvidia-smi` showed; use the
  `--index-url` it shows in place of `.../whl/cu128` (only torch is needed, not torchvision or
  torchaudio).
- "No space left": E: needs about 12 GB free.

### 4. Preflight, from the public repository

```powershell
E:\pf9b\venv\Scripts\python.exe -m pip install "preflight[probe] @ https://github.com/uchuuronin/preflight/archive/refs/heads/main.zip" tzdata
```

```powershell
E:\pf9b\venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

**You should see:** `Successfully installed ... preflight-0.1.0 ...` (pip leaves step 3's CUDA torch
alone: "Requirement already satisfied: torch"), then the same `+cu128 True` as before. `tzdata`
is the time-zone database Windows lacks; A's fix doesn't need it, and it's harmless.

**If it fails:**
- A 404: the repository isn't public yet. In the browser, signed in to GitHub, right-click
  `https://github.com/uchuuronin/preflight/archive/refs/heads/main.zip`, choose "Save link as", and
  save it in `E:\pf9b` (not Downloads, which is on C:). Then type
  `E:\pf9b\venv\Scripts\python.exe -m pip install "preflight[probe] @ file:///E:/pf9b/preflight-main.zip" tzdata`.
- The second line now says `+cpu` or `False`: pip replaced torch. Repeat step 3's install line.

### 5. Check the CLI

```powershell
E:\pf9b\venv\Scripts\preflight.exe --help
```

**You should see:** `usage: preflight [-h] {map,run,watch,demo} ...`.

**If it fails** (Windows or antivirus blocks `preflight.exe`): use
`E:\pf9b\venv\Scripts\python.exe -m preflight.cli.main` in its place, here and below.

### 6. The script

```powershell
Invoke-WebRequest -UseBasicParsing https://raw.githubusercontent.com/uchuuronin/preflight/main/zoo/notebook_tiny/tiny_train.py -OutFile E:\pf9b\film\tiny_train.py
```

(If the repository isn't public: open `E:\pf9b\preflight-main.zip` in Explorer and copy
`preflight-main\zoo\notebook_tiny\tiny_train.py` to `E:\pf9b\film\`.)

### 7. More steps, and fewer loss lines, so the shot reads well

```powershell
(Get-Content E:\pf9b\film\tiny_train.py) -replace 'max_steps = 200$', 'max_steps = 60000' -replace 'step % 50 == 0', 'step % 5000 == 0' | Set-Content E:\pf9b\film\tiny_train.py
```

```powershell
Select-String -Path E:\pf9b\film\tiny_train.py -Pattern 'max_steps =|batch_size =|nn.Linear|step %'
```

**You should see** four lines: `batch_size = 64`, `max_steps = 60000`,
`model = nn.Sequential(nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 1)).to(device)` and
`if step % 5000 == 0:`. The script prints its loss every 5,000 steps now instead of every 50, so
Preflight's alert lines stay on screen rather than scrolling away.

**How long it runs:** with step 11's change an A10 did about 300 steps a second, so 60,000 steps is
about 3 minutes there [derived]; a laptop's RTX 4070 has less than half an A10's memory
bandwidth, so expect roughly twice that [guess]. Press Ctrl+C whenever you have the shot. For a
different length, type step 7's first line again with `max_steps = 60000$` as the old text.

### 8. Make sure the records folder starts empty

```powershell
Get-ChildItem E:\pf9b\records -ErrorAction SilentlyContinue
```

**You should see:** nothing the first time. On a retake, move old files out of `E:\pf9b\records`
first, so you pick the right file in step 13.

### 9. Measure the script (Preflight's probes)

```powershell
E:\pf9b\venv\Scripts\preflight.exe run film\tiny_train.py
```

**You should see**, after up to a few minutes the first time (Windows loads CUDA slowly once), a
block of JSON starting:

```text
{
  "record": "records\\<share>.json",
  "lens": "pretrain",
  "backend": "cuda",
  "gpu": "NVIDIA GeForce RTX 4070 Laptop GPU",
```

then three probes with `"status": "ok"`. Note the record's name (`<share>` stands for it).

**If it fails:**
- `AttributeError: module 'os' has no attribute ...` (`geteuid`, `killpg`, `O_NOFOLLOW`), `No module
  named 'resource'`, or every probe's error says `WinError 10106`: A's Windows fix isn't in this
  install. Check `main` has it, repeat step 4 with `--force-reinstall --no-deps` added before the
  quoted package, or use the WSL2 route below.
- `ZoneInfoNotFoundError`: `tzdata` is missing; type
  `E:\pf9b\venv\Scripts\python.exe -m pip install tzdata` and repeat.
- `"backend": "cpu"`: torch can't see the GPU; go back to step 3's check.
- A probe with `"status": "oom"`: something else is using the GPU's memory; close it and repeat.

### 10. Check the record has its ranges

Type this with your record's name in place of `<share>`:

```powershell
E:\pf9b\venv\Scripts\python.exe -c "import sys; from preflight.schema import load_record; r = load_record(open(sys.argv[1], encoding='utf-8').read()); print({k: (e.status, e.lo, e.hi) for k, e in r.predictions.estimates.items() if k in ('time.step', 'memory.peak')})" records\<share>.json
```

**You should see** both keys with two numbers each, for example
`{'time.step': ('measured', 0.0012..., 0.0027...), 'memory.peak': ('extrapolated', ..., ...)}`
(those are the A10's; yours differ).

**If one says `'abstain'` with `None`:** that alert can't fire. Without `time.step` there's no
step-time alert; without `memory.peak`, no memory alert. One is enough to film. If both abstain,
stop and tell D, with the reasons: the same line with `(e.status, e.reason)` in place of
`(e.status, e.lo, e.hi)`.

### 11. The change the prediction didn't see

```powershell
(Get-Content E:\pf9b\film\tiny_train.py) -replace 'nn.Linear\(32, 256\), nn.ReLU\(\), nn.Linear\(256, 1\)', 'nn.Linear(32, 8192), nn.ReLU(), nn.Linear(8192, 1)' -replace 'batch_size = 64$', 'batch_size = 4096' | Set-Content E:\pf9b\film\tiny_train.py
```

```powershell
Select-String -Path E:\pf9b\film\tiny_train.py -Pattern 'max_steps =|batch_size =|nn.Linear|step %'
```

**You should see:** `batch_size = 4096`, `max_steps = 60000`, `nn.Linear(32, 8192), nn.ReLU(),
nn.Linear(8192, 1)` and `if step % 5000 == 0:`. A bigger layer and batch than the record measured:
on the A10 that moved both memory and step time out of their bands. (A small change, such as the
batch alone, may not: memory is reserved in blocks, and launch time dominates on a model this
small.)

### 12. The cockpit, ready beside the terminal

In Chrome or Edge, open `https://preflight-cqn.pages.dev`, go to **Logbook**, and scroll to
**Watch a real run**. Leave it there, beside the PowerShell window, ready to press
**Choose the live record**.

### 13. Watch the run (film from here)

Start recording, then type, with your record's name:

```powershell
E:\pf9b\venv\Scripts\preflight.exe watch records\<share>.json film\tiny_train.py --window 200
```

**You should see,** in the terminal:

```text
[preflight] watching film\tiny_train.py against record <share>; live record records\<new>.json
0 <loss>
[preflight] step 201: peak memory 0.40 GiB, above the predicted 0.27–0.36 GiB
[preflight] step 401: 0.0033 s a step, above the predicted 0.00125–0.00275 s
```

(The A10's numbers; yours differ. The alerts come within a few seconds; after them, a loss line
every 5,000 steps.)

Then in the browser press **Choose the live record**, open `E:\pf9b\records`, and choose
`<new>.json`: the new one, **not** `<share>.json` and **not** `<new>.live.jsonl`. The cockpit
opens and says "Following <new>.json: it refreshes as the run writes it.", and shows the run's
step, its step time and peak memory against the predicted ranges, and the same two alerts in its
own words (for example "Step 201: peak memory reached 0.40 GiB, above the predicted ..."). It
rereads the file every 3 seconds.

When you have the shot, press **Ctrl+C** in PowerShell (the record says the run was interrupted),
or let it finish. The last line is `[preflight] <n> steps, <k> alerts; record records\<new>.json`.

**After filming, close the PowerShell window** (or end any `python.exe` left in Task Manager's
Details tab): on Windows, a process the script started and left running isn't ended when the
probe or the run exits, and it keeps the GPU's memory until it is.

**If it fails:**
- The browser says "That isn't a live record from preflight watch": you chose `<share>.json` or the
  `.live.jsonl`; choose `<new>.json`.
- It says "Read <new>.json once": the browser can't follow files (not Chrome or Edge); use Chrome
  or Edge, or choose the file again to refresh.
- The run stops with `PermissionError: [WinError 5]` while the browser follows it: Windows
  wouldn't let Preflight replace the file while the browser or antivirus was reading it. With A's
  fix Preflight retries, skips a running write that still can't land, and retries the last write
  for about 9 seconds before it says so; if it still fails, move `<new>.json` and
  `<new>.live.jsonl` out of `records` and redo step 13.
- No alert in the terminal: check step 11's lines took, and that step 10 had a range.
- The Logbook shows no "Watch a real run": tell C (`c.inflight` is on in production since Nov 2).

## A changed script, watched live

Since C's D87 (web only; live with C's deploy of `main` `ce764b41`, D88), a followed run fills the
dials where the record has no prediction at the run's own settings: time to finish (projected from
the steps so far), run cost (that time at the record's GPU rate) and peak memory (the peak so far),
each labelled **Measured live**, one reading with no range (`web/src/workspace/live/liveFill.ts`).
Where the record does predict, the dials keep its prediction and the live panel compares the run
with it. So for a changed script's own predictions, with ranges, measure it first: step 9's
`preflight run film\tiny_train.py`, then step 13's `preflight watch` against the record it prints.

## To redo a take

The record from step 9 stays good. Move `E:\pf9b\records\<new>.json` and `<new>.live.jsonl` out
of the folder, and repeat step 13. Only repeat steps 9 to 11 if you changed the script back.

## Only if native Windows still fails on Wednesday: WSL2, kept on E:

WSL2 runs Ubuntu beside Windows and sees the GPU through the Windows driver; it's the Linux code
path D checked on a CPU and an A10. WSL itself is part of Windows (on C:, like the driver); the
Ubuntu disk and every file below go on E:. It needs virtualisation on in the laptop's BIOS (if
WSL says `0x80370102`, it's off).

In PowerShell **as administrator**, once (it may ask to restart):

```powershell
wsl --install --no-distribution
```

Then, in a normal PowerShell, first type step 1's three lines (so the download lands in
`E:\pf9b\tmp`, not on C:), then:

```powershell
wsl --install -d Ubuntu-24.04 --location E:\wsl\ubuntu
```

**You should see:** Ubuntu's setup asking for a new user name and password. `--location` puts the
distribution's disk at `E:\wsl\ubuntu` (Microsoft's documented option for `wsl --install`). If your
WSL doesn't know `--location`, update it with `wsl --update` (as administrator) and repeat.

In the Ubuntu window that opens, one line at a time:

```bash
nvidia-smi
sudo apt update && sudo apt install -y python3-venv
mkdir -p /mnt/e/pf9b/film && cd /mnt/e/pf9b
python3 -m venv ~/pf9b-venv
~/pf9b-venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
~/pf9b-venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
~/pf9b-venv/bin/pip install "preflight[probe] @ https://github.com/uchuuronin/preflight/archive/refs/heads/main.zip"
curl -L -o film/tiny_train.py https://raw.githubusercontent.com/uchuuronin/preflight/main/zoo/notebook_tiny/tiny_train.py
sed -i 's/^max_steps = 200$/max_steps = 60000/; s/step % 50 == 0/step % 5000 == 0/' film/tiny_train.py
~/pf9b-venv/bin/preflight run film/tiny_train.py
```

The venv lives in Ubuntu's home folder, inside its disk on E: (installing torch onto `/mnt/e`
directly is much slower). `/mnt/e/pf9b` is `E:\pf9b`, so the records land where Windows' browser
can open them. Ubuntu 24.04's Python is 3.12, which Preflight takes. Then steps 10, 11 and 13 as
above, with `~/pf9b-venv/bin/python` and `~/pf9b-venv/bin/preflight` in place of
`E:\pf9b\venv\Scripts\python.exe` and `...\preflight.exe`, `/` in place of `\`, and step 11's change
typed as:

```bash
sed -i 's/nn.Linear(32, 256), nn.ReLU(), nn.Linear(256, 1)/nn.Linear(32, 8192), nn.ReLU(), nn.Linear(8192, 1)/; s/^batch_size = 64$/batch_size = 4096/' film/tiny_train.py
```

In step 13 the browser picks the file from `E:\pf9b\records` as before.

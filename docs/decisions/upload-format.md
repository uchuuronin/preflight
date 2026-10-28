# What an upload hands the harness

Owner: A. For D5's notebook converter (D's request, Oct 27) and C's check-in desk. The code that
enforces this is `runner/sources.py` (`fetch`) and `probes/local.py` (`validate_files`); the tests
are `tests/test_sources.py`.

## The source object

`POST /v1/jobs` takes a `JobRequest`; its `source` is one of these.

| kind | Fields | What A does with it |
| --- | --- | --- |
| `upload` | `files: {path: text}`, optional `entrypoint`, optional `args` | Runs the files as given |
| `notebook` | `files: {"<name>.ipynb": <the notebook JSON as text>}`, optional `args` | Converts it (below), then as an upload |
| `repo` | `uri: "https://github.com/<owner>/<repo>"`, optional `commit` (full 40-character hash), optional `entrypoint`, `args` | Reads the text files at that commit (default branch if none) |
| `sample` | `uri: "sample://zoo/<dir>"` | One of the bug-zoo scripts in `zoo/` |
| `record` | `record_ref`: a share token or `sample:<id>`, with `mode` probe, verify or sample_more | Re-runs that record's code (guide jobs) |

The job store moves `files` out of the job's state at submit time and keeps a `files_ref`; the
record's `source.uri` then reads `upload://sha256:<hash>`. Nothing changes in what the caller sends.

## Files

- Paths are relative POSIX paths of letters, digits, `. _ - /` and spaces: no leading `/`, no
  `..`, no dotfiles or dot-directories.
- Text only, UTF-8. Binary files (weights, images, parquet) aren't part of an upload; a repo is
  read for `.py .sh .sbatch .slurm .yaml .yml .json .toml .cfg .txt .md` files only.
- At most 2,000 files, 8 MB per file and 25 MB in all.
- Data comes from Hugging Face (datasets or the Hub) or from text files inside the upload: probes
  reach Hugging Face and PyPI only, so a script that downloads from anywhere else fails with
  `download_blocked`.

## Entry point

`entrypoint` names the file to run, a key of `files`. Without one, A picks the first of
`train.py`, `main.py`, `run.py`, `finetune.py`, `pretrain.py` that exists, else the only `.py`
file; otherwise the job fails asking which file to run. `args` is a list of up to 64 strings of up
to 4,096 characters, passed as the script's `sys.argv[1:]`.

## requirements.txt

A `requirements.txt` at the root is installed into the probe before the script runs.

- The probe image's stack wins. Every package it already has (torch 2.5.1, transformers 4.46.3,
  accelerate, peft, trl, datasets, safetensors, numpy and their dependencies) is skipped with a
  note in the log, and passed to pip as a constraint so nothing else can upgrade it.
- Only plain PyPI lines are installed: `name`, `name[extra]`, version specifiers and environment
  markers. Options (`-r`, `-e`, `--index-url`), URLs, VCS (`git+`) and local paths are skipped with
  a note.
- Wheels only (no source builds), up to 50 packages, 4 minutes inside the probe's own time limit.
  It installs once per job; the job's later probes reuse it.
- A failed install is the probe's error, code `missing_package.install`, with pip's last lines. The job's
  other points are skipped with the same code instead of failing the same way again.

## Notebooks

Send the `.ipynb` as it is, kind `notebook`; A converts it, so there's one converter and its rules
are the same everywhere. The conversion:

1. Code cells in order, joined by blank lines; markdown and raw cells are dropped.
2. Lines starting with `!` or `%` (shell and magics) become comments.
3. The packages named by `!pip install` and `%pip install` lines (also `!python -m pip install`)
   become `requirements.txt`: options and their values are dropped, and anything after `;`, `|`,
   `&`, `#` or `$` is ignored.
4. The script is `<notebook stem>.py`, and it is the entry point.

So `train.ipynb` arrives at the probe as `{"train.py": <script>, "requirements.txt": <pip lines>}`.
If D5 shows a preview of what will run, `runner/sources.py`'s `_notebook_to_script` and
`notebook_requirements` are the reference; a test that checks D5's output against them catches drift.

## What fails, and how it says so

A submission that breaks these rules fails before anything runs, as a visitor error with
`JobEvent.error_code` in the `submission` family (`submission`, `submission.invalid`,
`submission.files_gone`, ...); the message says what to change. Failures while running are the
probe's own codes (`missing_package`, `download_blocked`, `oom`, ...).

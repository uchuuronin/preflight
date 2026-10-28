# Muse Code in a sandbox with nobody present (C, Oct 27, proposal)

**Question (PM).** The fix loop runs the Muse Code terminal agent inside a sandbox with
`META_API_KEY`. How is it installed and authenticated non-interactively there?

Sources: Meta's docs, read Oct 27: [Muse Code](https://dev.meta.ai/docs/muse-code),
[Authentication and billing](https://dev.meta.ai/docs/muse-code/auth),
[Extending and automating: headless and CI](https://dev.meta.ai/docs/muse-code/extending),
[Permissions and safety](https://dev.meta.ai/docs/muse-code/permissions). Nothing below has been
run yet: a smoke test spends tokens and waits for Chiya's yes.

## Install

- The only documented install is `curl -fsSL https://dev.meta.ai/install.sh | sh`, which puts a
  native `muse` binary on PATH (Linux supported; musl builds lack the sandbox helper). No package
  or version pin is documented.
- So it is installed once, at image build, in the fix sandbox's Modal image; `muse --version` is
  printed at build and stored in every fix record. The image is kept, so the version only changes
  when we rebuild on purpose. `muse export --session <id>` embeds the version too.

## Authentication

- `META_API_KEY` in the environment is used first (before a stored key or a browser session), so
  no `muse auth set` or login step runs in the sandbox. CI example from the docs:
  `META_API_KEY="$KEY" muse exec "…"`.
- The key comes from the Modal secret `preflight-meta`, mounted only into the fix sandbox.

## Running headless

- `muse exec --json --prompt-file task.txt --max-model-steps <N>`: JSONL events on stdout, one
  task, a hard step cap. Exit 0 means the turn finished, not that the fix is right; every fix is
  gated on our own re-probe and fixed-seed loss check (playbook).
- `muse export --session <id> --out session.json` is the audit trail stored with the fix.

## The open problem: the key and the visitor's code share a sandbox

Muse Code runs shell commands in the workspace (tests, the script itself). Those commands run the
visitor's code, and `META_API_KEY` is in the same environment, so a hostile repo could copy the
key into a file that comes back in the diff. Muse Code's own sandbox limits writes and network,
not environment variables, and it needs a working bubblewrap, which may not run inside a Modal
Sandbox (gVisor); without it "every sandboxed shell command aborts".

Options (Chiya decides):

1. **Muse Code edits, never runs.** Allow only read and edit tools; every command (tests, the
   probe) runs afterwards in a separate sandbox with no key, which is the re-probe we run anyway.
   Pro: the key never shares a process tree with visitor code. Caveat: Muse Code can't try its own
   patch, and how `muse exec` behaves when a command would need approval isn't documented, so the
   smoke test must confirm it fails closed.
2. **A key that can't do much.** A separate Meta API key for the fix loop only, with a low spend
   limit on Meta's side, rotated daily, and the output (diff and session log) scanned for the key
   before anything leaves the sandbox. Pro: works with Muse Code running tests as designed.
   Caveat: a leaked key still spends up to its limit until it is rotated.
3. **The key never enters the sandbox.** Muse Code talks to a small proxy that adds the key
   outside the sandbox and accepts only a per-job token. Pro: nothing to leak. Caveat: Muse Code's
   docs don't say it can use another base URL, so this needs Meta's confirmation or the smoke test.

Whatever is chosen: the diff and session log are scanned for the key and the sandbox's other
secrets before they are stored, egress is limited to Meta's API host, and the sandbox is
destroyed after the run.

## Smoke test, once approved

What: build the fix image (installs Muse Code), then one `muse exec` on the LoRA zoo script with
the eval-batch fix as the task, `--max-model-steps 20`, in a Modal Sandbox on CPU. Where:
`remote-run.yml`. How long: about 5 minutes. Expected cost: Muse Spark tokens for one fix,
likely under $0.10 at the guide's price table, plus a few CPU minutes on Modal. It also answers
whether bubblewrap works in a Modal Sandbox and how headless approvals behave.

# Builder sessions: kit

For watching outside builders use Preflight for the first time, after the public deploy. Each
session is one builder, one observer and about 40 minutes. The aim is to find where a stranger
hesitates or gets stuck; every one of those places becomes an onboarding fix.

| File | Who uses it | When |
| --- | --- | --- |
| [brief.md](brief.md) | the builder | sent the day before, and open during the session |
| [tasks.md](tasks.md) | the builder reads it; the observer keeps time | during the session |
| [observation-sheet.md](observation-sheet.md) | the observer | during the session, one copy per builder |
| [form.md](form.md) | the builder | straight after the session |

## Tonight's session (Sun Nov 1): ready to hand over

- **Link:** <https://preflight-cqn.pages.dev> (already in the brief).
- **What's on in production** (`flags.production.json`, Chiya, Nov 1): check-in (`runner.queue`)
  with the Repo link and Upload tabs, and the live guide on Muse Spark (`c.guide.live`). The
  Notebook tab is off (`d.notebook` isn't in production), so a builder with a notebook downloads it
  as a `.py` and uploads that. Checked on a local build with the production flags, Nov 1: the
  check-in desk shows exactly two tabs, Repo link and Upload.
- **Demo limits:** 3 probe jobs per visitor per day, and a daily GPU spend cap for the whole site,
  $10 a day through Nov 4 [cited: `python/preflight/runner/queue.py`]. A's status reports them
  deployed (`a-deploy-main-19`, 20:20 UTC Nov 1). A builder past either sees "samples only until
  midnight ET"; when jobs in flight hold the day's budget, the desk says the GPUs are busy and to
  retry in about 2 minutes. Either is outside the builder's control: note it word for word and
  go on with a sample.
- **Cost:** about $0.20 a check-in, a lower bound [derived: one measured run, Oct 28, A's status
  file: $0.086 of GPU and $0.018 of Muse Spark [measured], plus about $0.10 of sandbox CPU and
  memory [guess]]. The GPU cap counts the GPU only; the live guide has its own daily cap. One
  builder with up to 3 check-ins is about $0.60 or more [derived].
- **The guide is live tonight on Muse Spark** (it moves knobs with tool calls). The tasks don't
  mention it, so the observer can see whether the builder finds it unprompted.
- **Files to send:** the builder gets [brief.md](brief.md) the day before (or now), then
  [tasks.md](tasks.md) at the start and [form.md](form.md) at the end. The observer keeps
  [observation-sheet.md](observation-sheet.md).
- **If no one can watch live:** ask the builder to record their screen and think aloud (only with
  their yes); the observer fills the sheet from the recording, times from the recording's clock.

## Before the session

- Ask the builder to bring one training script they know: a public GitHub repo or a `.py` file (a
  notebook downloaded as `.py` works), up to 5 MB, loading data from Hugging Face. No keys or
  passwords in the code.
- Open the link once yourself and check that the home page loads and New flight shows the desk.
- A's status reports the demo limits deployed at 20:20 UTC (`docs/status/a.md`), but A hasn't
  confirmed it yet: watch that the builder runs no more than about 3 check-ins, and if a 4th gets
  through, note it as a bug.

## During the session

- The builder thinks aloud. The observer doesn't help, explain or point. If the builder is stuck
  for 2 minutes, the observer offers one hint and marks it on the sheet as "needed help".
- The observer records times from the screen clock, not estimates.
- Screenshots or a screen recording only with the builder's yes.

## After the sessions

- Send D the filled sheet and form, anonymised (B1, no names), in the chat or as files. D ranks
  every stuck point and hesitation as an onboarding fix and files it.
- Report counts, not percentages: with three builders, "2 of 3 reached a map without help".
- Every hesitation or stuck point, with where it happened, becomes a request in
  `docs/requests/from-d.md` (to C for the app, to D for D's panels), ranked by how many builders it
  stopped. Quote the builder's words when they help.
- Keep the filled sheets and forms out of the repo if they name anyone; the repo gets the counts
  and the anonymised findings only.

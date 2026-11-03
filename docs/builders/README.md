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

## The session on the final product (from Tue Nov 3)

Updated by D, Tue Nov 3, to the production flags at the freeze (`flags.production.json`).

- **Link:** <https://preflight-cqn.pages.dev> (already in the brief).
- **What's on in production:**
  - The check-in desk has three tabs: Repo link, Upload and Notebook. A notebook goes in as it
    is; no `.py` export is needed.
  - Check-ins install the repo's declared dependencies when there's no `requirements.txt`, and
    take extra packages on request.
  - An evaluation can be checked in by command (`lm_eval ...`).
  - "Private triage" is a tick box on the desk.
  - In the cockpit: the live guide on Muse Spark; the fix loop ("Fix it with Muse Code") with
    Preflight's verify run; Muse Spark's diagnosis of the record's plots; Spark Open's triage of
    failed probes; the fit map; and share links.
  - The Logbook has "Watch a real run" (in-flight mode).
- **Demo limits:**
  - 3 probe jobs per visitor per day, plus one free resend a day of the same code after a check-in
    that stopped at a missing package before measuring anything.
  - A daily GPU spend cap for the whole site: $10 a day through Nov 4 and $8 a day from Nov 5
    [cited: `python/preflight/runner/queue.py`].
  - A builder past a limit sees "samples only until midnight ET". When jobs in flight hold the
    day's budget, the desk says the GPUs are busy and to retry in about 2 minutes.
  - Either is outside the builder's control: note it word for word and go on with a sample.
- **Cost:** a job may spend up to $0.60 [cited: `JOB_BUDGET_USD`].
  - One measured run cost about $0.10 of GPU and Muse Spark: $0.086 and $0.018, Oct 28 [measured,
    A's status file].
  - Add the sandbox's CPU and memory, about $0.10 [guess].
  - So one builder with up to 3 check-ins is about $0.60 or more [derived].
  - The guide, the fix loop and the model calls have their own caps.
- **The guide and the fix loop are live.** The brief mentions both in one line; the tasks don't,
  so the observer notes whether the builder goes to them.
- **Known rough spots** (D's stranger walkthrough, Nov 2, `from-d.md`). The observer doesn't point
  these out, and ticks the sheet's rows when a builder hits one, so C knows which ones real people
  trip on:
  - four measured samples (GRPO, retrieval, GNN, vision) open with blank time and cost dials;
  - the required "Got it" box sits below the samples list, away from the "Check in" button;
  - the out-of-memory finding's own fix ("Eval without gradients") doesn't clear the
    out-of-memory dial;
  - Escape doesn't close an open knob;
  - the header gives a whole run's cost for a run it says will fail.
- **Files to send:** the builder gets [brief.md](brief.md) the day before (or now), then
  [tasks.md](tasks.md) at the start and [form.md](form.md) at the end. The observer keeps
  [observation-sheet.md](observation-sheet.md).
- **If no one can watch live:** ask the builder to record their screen and think aloud, only if
  they say yes. The observer fills the sheet from the recording and takes times from the
  recording's clock.

## Before the session

- Ask the builder to bring one training script they know: a public GitHub repo, a `.py` file or a
  Jupyter notebook, up to 5 MB, loading data from Hugging Face. No keys or passwords in the code.
- Open the link once yourself and check that the home page loads and New flight shows the desk.
- If a builder gets a 4th paid check-in through in a day, note it as a bug (the free resend
  after a check-in that measured nothing isn't one).

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

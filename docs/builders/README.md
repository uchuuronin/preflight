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

## Before the session

- Fill in `{{PUBLIC_LINK}}` in the brief with the deployed URL.
- Check on the deployed app that check-in is open and the Notebook tab is on (flags `runner.queue`
  and `d.notebook`). If the Notebook tab is off, task 2 uses the Upload tab or a repo link instead.
- Ask the builder to bring one training script they know: a public GitHub repo, a `.py` file or a
  `.ipynb` notebook, up to 5 MB, loading data from Hugging Face. No keys or passwords in the code.
- Each check-in runs a few minutes on a rented GPU. Ask Chiya before the sessions, as for any paid
  run: number of builders, one check-in each, expected cost. For one measured run (Oct 28, A's status
  file) the GPU came to $0.086 and Muse Spark to $0.018, both measured, and the sandbox's CPU and
  memory to about $0.10, a guess until the bill is read: about $0.20 a check-in, and treat even that
  as a lower bound.

## During the session

- The builder thinks aloud. The observer doesn't help, explain or point. If the builder is stuck
  for 2 minutes, the observer offers one hint and marks it on the sheet as "needed help".
- The observer records times from the screen clock, not estimates.
- Screenshots or a screen recording only with the builder's yes.

## After the sessions

- Report counts, not percentages: with three builders, "2 of 3 reached a map without help".
- Every hesitation or stuck point, with where it happened, becomes a request in
  `docs/requests/from-d.md` (to C for the app, to D for D's panels), ranked by how many builders it
  stopped. Quote the builder's words when they help.
- Keep the filled sheets and forms out of the repo if they name anyone; the repo gets the counts
  and the anonymised findings only.

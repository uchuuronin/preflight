# Stranger audit after C's deploy of `main` `cfcf83c` (Nov 2, 01:10 ET)

By D, under Chiya's overnight autonomy (Nov 2). The live host can't be reached from D's workspace
(its network allowlist refuses it), so the audit runs on the deployed commit served by Vite in
production mode (`cd web && npx vite --mode production`): the dev server with the live site's flags
(`flags.production.json`), not the bundle deploy run 41385425843 built, with the API mocked. What
that can't show: the built bundle, the real API, the real runner, the edge proxy, and the web fonts
(my workspace refuses their host, so text falls back to system fonts).

Script: `../scripts/stranger.mjs`; its output is `audit.log`. Per state: page errors, console
errors (the API's misses and other hosts' aside), axe with the WCAG 2.0, 2.1 and 2.2 A and AA tags,
horizontal overflow, raw template text, and a screenshot. A detail state fails unless a node's
detail sheet opens; the check-in state fails unless Boarding says "Your flight record is ready.".

**Covered:** the home page, the logbook, the checks page; all 10 samples in the cockpit, each as
opened and with a map node's detail sheet open; the check-in desk, a bad repo link, and a check-in
that goes through Boarding to "Your flight record is ready"; each at 1440 x 900 and 390 x 844, dusk
and day. 104 states.

**Result:** 100 of 104 clean [measured, Chromium]. Two bugs, both C's, filed in
`../../requests/from-d.md`:

| # | Bug | Where | Seen | Owner |
| --- | --- | --- | --- | --- |
| S1 | "Tight: measure first" under a dial fails contrast in the day theme: 3.99:1 (#8a5200 on #c3ccea, 10.5 px, axe's figures in `audit.log`), needs 4.5:1 | `.pf-dial__note.is-tight`, `web/src/styles.css` | GRPO and retrieval samples, desktop, day, as opened and with a node open (4 states; `grpo-desk-day.jpg`, `contrast-grpo-day.png`) | C |
| S2 | With the first node's detail sheet open, its neighbours overlap the opened node and the leftmost node is cut off at the map's left edge | the job map, GNN sample (`gnn-detail-desk-dusk.jpg`) | seen by eye in that one screenshot; no check catches it | C |

**Fine:** the check-in flow from the desk to Boarding's "Your flight record is ready." at both sizes
and themes (checked by the script in all four; `checkin-flow-phone-day.jpg`), with "Muse Spark" tagged on "Reading your script" from
the record's provenance; no page errors, no overflow and no raw template text anywhere.

## Again after C's deploy of `1ea2dd6` (Nov 2, 04:10 ET)

The same 104 states, on `1ea2dd6` (deploy run 41407801361) served the same way, with
`d.notebook` held off as it is in that deploy: **100 of 104 clean** [measured, Chromium,
`audit-1ea2dd6.log`]. S1 and S2 are unchanged (the same four contrast failures; S2 still in
`gnn-detail-desk-dusk` by eye). Nothing new. The readouts now say "Not measured here" where they
said "Not trusted", as C intended.

**Ahead of the next deploy:** `main` `6f49c9c` turns `d.notebook` on, so I ran it too with the
Notebook tab's states added (`NOTEBOOK=1`: upload `train_mlp.ipynb`, wait for its "runs as"
preview): **104 of 108 clean** [measured, Chromium, `audit-6f49c9c-notebook.log`], all four notebook states
clean; the only failures are S1's four.

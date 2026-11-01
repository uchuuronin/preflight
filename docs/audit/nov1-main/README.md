# Audit re-check on `main` after round four, Sun Nov 1

By D, on `ws-d` at `0683d27`, which is `main` `c4b95d4` merged in with nothing else changed
outside D's docs. Served by the Vite dev server and checked in Chromium with the scripts in
`../scripts/`: the 40-state sweep (axe 4.10, `wcag2a`, `wcag2aa`, `wcag21aa`, `wcag22aa`), the
targeted re-checks, and a new model-tag check (`model-tags.mjs`, exits 1 on any mismatch).
Finding 12 was checked first. An independent read-only review re-ran or spot-checked each row.

| # | Finding | On `main` `c4b95d4` [measured, Chromium] |
| --- | --- | --- |
| 12 | Model tags show whether or not the model ran | **Fixed on `main` `85fbf9a`** (round five, Nov 1, 12:45): `model-tags.mjs` passes 12 of 12, the same-page case included, desk and phone [measured, Chromium]. On `c4b95d4` it was **partly fixed:** right for each job on its own, but tags carry over to the next job in the same page (below). It was open with C then, with a tested fix. |
| 7 | Day-theme contrast | **Fixed.** axe finds no colour-contrast violation in any of the 40 states (`cockpit-desk-day.jpg`). D's add-on check goes from 14 to 20 of 20. |
| 10 | Two job-map nodes with no focus ring | **Fixed.** No element without a visible focus ring in 12 Tab presses on any screen; every job-map node now takes a 2 px outline when focused, desk and phone. |
| 8 | Eval markers 14 px | **Fixed.** axe target-size passes in all 40 states, the 4 knob-dock states included. |
| 11 | "About ..." buttons 22 px; the checks page's "4 not scored" summary 19 px | **Fixed.** The "About ..." buttons are 24 x 24 px at both sizes, and the checks page has no target under 24 px. |
| 9 | Boarding's route labels cut off on phones | **Fixed** (as on `ws-c`): "Your script" and "Flight record" show whole at 390 px (`boarding-failed-phone.jpg`). |
| 4 | No stage marked after a failure | **Fixed** (as on `ws-c`): the stages before the failure are done, and "Running probe steps" reads "Stopped here" in the failure colour. |
| 6 | Job-map labels about 6 px on phones (Oct 30 audit) | **Not fixed.** At 390 px the share lines render at 5.5 px and the names at about 6.6 px (median of 21 labels), against 9.2 and 11.1 px on a desk. |


**Finding 12 in detail** (`model-tags.mjs`, desk and phone). Each job's tags now come from its
record's provenance, and all five single-job cases pass:
- a job that fails before any model call shows no tag;
- one whose record has only the script map shows "Muse Spark" on "Reading your script" and
  nothing on "Checking the logs";
- one whose record has a Spark Open triage call shows "Muse Spark Open" there, whether the job
  then finishes or fails (`tags-spark-open-ran-phone.jpg`);
- a script-map call that failed shows no tag.

**But tags carried over to the next job** (on `c4b95d4`; fixed on `85fbf9a`, above). Boarding stays mounted, and its provenance state isn't
cleared when the job changes. So after a job that ran triage, a second job in the same page that
fails before any call still shows both tags. A read-only review found this, and I reproduced it.
The fix with C is one line: clear the state when the job or its record changes. I tested it on a
scratch copy: `model-tags.mjs` goes from 10 of 12 to 12 of 12, and the typecheck and C's app tests
pass.

No horizontal overflow and no console errors from the app in the sweep. The sweep's focus test
counts any box-shadow or parent outline as a ring, and in 12 Tab presses it reaches only the first
map nodes. So finding 10 was also checked directly: every job-map node shows a 2 px solid outline
when focused, at both sizes. The sweep also ignores failed-request console messages.

**Still open, minor (axe passes both on spacing, so neither is a WCAG 2.5.8 failure):**
- The cockpit's "Your limits" summary is 20 px tall (258 px wide on a desk, 296 px on a phone).
- The check-in's "Got it" box is 16 px, and its label, which is also its target, is 63 x 22 px.

24 px minimum heights would match the rest of the controls.

**Test setup, not the app:** with no API behind the dev server, the cockpit's
`/v1/guide/limits/state` request gets a 502 from Vite's proxy and the browser logs it. The
add-on check prints that one path as a warning, and still fails on any other error, a 502 on any
other `/v1` path included (checked both ways).

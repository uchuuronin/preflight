# Audit pre-check on C's branch, Sun Nov 1

By D, on `ws-c` at `80d4217` (not yet in `main`), exported to a scratch folder and served
locally, in Chromium. The same scripts as Oct 30 and Oct 31 (`../scripts/`): the 40-state sweep
and the targeted checks. When these land in `main` they are re-checked there before moving to
Done.

| # | Finding | On `ws-c` `80d4217` [measured] |
| --- | --- | --- |
| 12 | Model tags show whether or not the model ran | **Not fixed.** A job that ran out of memory while probing, before any triage, still shows "Muse Spark Open" on "Checking the logs" (`boarding-failed-phone.jpg`). The tags are still fixed on their stages (`STAGES` now in `boarding.ts`). |
| 7 | Day-theme contrast | Not fixed: axe colour-contrast in the same 16 day-theme states. |
| 10 | Two job-map nodes with no focus ring | Not fixed: the same two nodes. |
| 8 | Eval markers 14 px | Not fixed: axe target-size in the same 4 states. |
| 11 | "About ..." buttons 22 px | Not fixed: still 22 x 22 px. |
| 9 | Boarding's route labels cut off on phones | **Fixed:** "Your script" and "Flight record" show whole at 390 px (`boarding-failed-phone.jpg`). |
| 4 | No stage marked after a failure | **Fixed:** the stages before the failure are marked done, and "Running probe steps" reads "Stopped here" in the failure colour. |

New on `ws-c`: the cockpit's "Your limits" summary is 20 px tall (258 px wide on a desk, 296 px
on a phone). axe's target-size rule passes it on spacing, so it is not a WCAG 2.5.8 failure; a
24 px minimum height would match the rest of the cockpit's controls.

The sweep is otherwise as on Oct 31: no horizontal overflow, no console errors.

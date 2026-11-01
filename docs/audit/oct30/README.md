# Stranger path audit, Oct 30

By D, on `ws-d` at round two plus `3c243b5` (the same app as `main` with D's panels), in Chromium.

## What was checked

- **Screens:** home; check-in on each tab (Repo link with a bad link, Upload, Notebook with a
  notebook chosen, Upload with check-in open); boarding mid-run and after a failure (job events
  answered by the test, as the API wasn't running); cockpit (sample `lora-qwen`, also with a knob
  open); checks page. D's own panels separately, with their flags on.
- **Each at** 1440 x 900 and 390 x 844, in both themes (dusk, day): 40 states, plus 8 for D's panels.
- **Per state:** axe 4.10 (tags `wcag2a`, `wcag2aa`, `wcag21aa`, `wcag22aa`: WCAG 2.0 A and AA, and the AA rules of 2.1 and 2.2); horizontal overflow; console errors;
  targets under 24 x 24 px; 12 Tab presses for focus with no visible ring; a screenshot, read for
  copy, states and errors (ux-writing, error-handling-ux, loading-states).

## Result

| Measure | Result |
| --- | --- |
| Horizontal overflow | none in any state |
| Console errors | none |
| axe violations | 2 kinds: colour contrast (16 states, all in the day theme) and target size (4 states, the flight plan's eval markers) |
| Targets under 24 px | the flight plan's eval markers (14 px), the "About ..." info buttons (22 px), the "4 not scored" summary (19 px tall) |
| Focus with no visible ring | two job-map nodes in the cockpit |
| D's panels | no axe violations; the free-GPU band's source links were 14 px tall, now a 24 px tap area (fixed in D's CSS) |

Twelve findings, ranked by how many strangers they would stop, are requests to C in
`docs/requests/from-d.md` (dated Oct 30). Screenshots are in this folder.

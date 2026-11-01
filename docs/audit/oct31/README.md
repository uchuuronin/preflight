# Stranger-path audit re-check, Oct 31 (round three)

By D, on `ws-d` at `006bc57` (`main`, round three, contract 0.5.0), in Chromium. Same method as
[the Oct 30 audit](../oct30/README.md): the same 40 states (10 screens x desk and phone x dusk
and day) through the same script. Then a targeted check of each of the four findings at the top
of the Oct 30 list, plus finding 5, which C fixed in place of 4.

## The top findings, re-checked one by one [measured: Chromium, Oct 31]

| # | Oct 30 finding | Now | How it was checked |
| --- | --- | --- | --- |
| 1 | A link that isn't a plain GitHub repo link leaves "Check in" grey with no reason | **Fixed** | "not a link", a GitLab link and `github.com/owner` each show "Paste a public GitHub repository link, like github.com/owner/repo." with `aria-invalid="true"`, the message named by `aria-describedby`, and in a live region; a good link clears all three. `f1-badlink-desk.jpg` |
| 2 | On a phone the "Got it" box is two panels below the hint | **Fixed** | At 390 px the hint is a button, "Tick “Got it” to check in"; it moves focus to the "Got it" box, in view, and ticking it with Space moves focus back to "Check in", enabled and in view. `f2-hint-phone.jpg` |
| 3 | Samples are called measured; the cockpit says illustrative and shows "Measured" | **Fixed** | Departures says "Samples with illustrative numbers"; the `lora-7b` sample's cockpit shows no "Measured" or "Predicted", marks each reading "Illustrative", says "3 illustrative probes", and shows no probe or plan cost (the dollar figures left are the dials' illustrative readings and the budget). `f3-sample-cockpit.jpg` |
| 4 | After a failure no stage is marked, so a visitor can't tell where it stopped | **Not fixed** | Events queued, admitted, mapping, probing, then failed (out of memory): all five stages show as not started, and nothing says "Stopped here". `Boarding.tsx` still finds the stage from the last event, which is `failed`. C lists boarding states as next. `f4-boarding-failed-phone.jpg` |
| 5 | An unsourced "about 4 minutes, under $1" on every check-in | **Fixed** | Now "One measured run: under 7 minutes on an L4 (28 Oct 2026, UTC).", from A's measured run, with no dollar figure. |

## The whole sweep, against Oct 30 [measured: the same script, both days]

| Measure | Oct 30 | Oct 31 |
| --- | --- | --- |
| States checked | 40 | 40 |
| Horizontal overflow | none | none |
| Console errors | none | none |
| axe: colour contrast | 16 states, all day theme | the same 16 states (same screens, views and theme) |
| axe: target size | 4 states (eval markers) | 4 states (eval markers) |
| Targets under 24 px | eval markers 14 px, "About ..." buttons 22 px | the same |
| Focus with no visible ring | two job-map nodes | the same two |

Findings 6 to 12, and how each was checked today:

- 7 (day-theme contrast), 8 (eval markers 14 px), 10 (two job-map nodes with no focus ring) and
  11 (small targets): still there [measured: the sweep above].
- 9 (boarding's route labels cut off on phones) and 12 (the Muse Spark and Muse Spark Open tags
  fixed on their stages whatever the job used): still there [measured: seen in
  `f4-boarding-failed-phone.jpg`].
- 6 (map labels too small on phones): not re-checked today; C lists it as next.

C lists contrast, phone map labels and boarding states as next, if time allows before the freeze.

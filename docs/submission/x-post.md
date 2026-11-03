# X post for the 45-second cut

Draft by D, Sun Nov 1, refreshed Tue Nov 3, for the 45-second X clip in `docs/demo/video-plan.md`
("The 45-second X clip: the cut"); `storyboard.md`'s 45-second cut is its fallback if shot 8b isn't
filmed. Every number is in the submission's numbers table (`README.md` here), and
each is also in the video's captions. The Most Viral prize counts engagement through Nov 8; the
post's link goes on the submission.

## The link

Share pages are live (C deployed `workers/` with production from `7ba177b`, Nov 2, 12:45 ET; the
Oct 28 record's `/s/` link, card and badge pass `scripts/share_page_check.py`). Both posts link
`https://preflight-cqn.pages.dev/s/ICDYrfHQ2nBNbv3mlRHHMg` (A's Oct 28 record): its card is
pre-rendered (`workers/hero-cards.json`), so the post previews with an image. Open it once on the
live site before posting.

## Post A: the bug alone (262 characters, a link counted as X's 23)

Only if the clip loses its triage beat (the clip's row E, from shot 6): with `storyboard.md`'s
fallback cut, if that cut drops it too.

```text
We planted a bug in this fine-tune. At step 200, its first evaluation, it runs out of memory.

Preflight ran a few real steps on one L4 and caught it in 189 seconds, for $0.042 of GPU at list price.

Fly your ML job before you pay for it: https://preflight-cqn.pages.dev/s/ICDYrfHQ2nBNbv3mlRHHMg
```

## Post B: with Meta's open model (269 characters)

The post for the 45-second X clip (`video-plan.md`), which keeps row E (shot 6, Spark Open's
triage on A's production record) and adds row F (shot 8b, Muse Code's fixes) on screen; the text
needs no change for F. The Oct 28
record it links predates triage, so a click finds no Spark Open finding there; the triage is in A's
re-run (`/s/Q3s5qyHKTNf-fCng4dwn6g`, an A10, Nov 2), shown in the video. To link the triaged record instead, its
numbers differ (an A10, not an L4): see the one-record draft in `docs/demo/video-plan.md`, a draft
for Chiya, not applied.

```text
We planted a bug in this fine-tune: at step 200 it runs out of memory.

Preflight ran a few real steps on one L4 and caught it in 189 seconds, for $0.042 of GPU at list price. Muse Spark Open, Meta's open model, reads every probe's log.

Try it: https://preflight-cqn.pages.dev/s/ICDYrfHQ2nBNbv3mlRHHMg
```

Paste exactly what is in the block (one line per paragraph). Add the hackathon's hashtag and
handles if the rules ask for them, and check the count after (X allows 280).

## Video alt text

A Python training script zooms to its evaluation setting. Preflight's cockpit then draws the
job as a map; a memory ring fills past its limit and a red finding says the evaluation runs out
of memory, 20.4 of 22.06 GiB, caught in 189 seconds of probing. Dragging a setting moves the ring and the dials together. A finding on the evaluation step,
labelled Found by Muse Spark Open triage, quotes the log line. Muse Code writes candidate fixes,
and Preflight re-runs them and keeps the one that fits. The end card shows the link and the
repository. (With `storyboard.md`'s fallback cut, drop the sentence about Muse Code and, if
that cut drops the triage, the one before it.)

## Reply under the post (optional)

Every number in Preflight says how it was made: measured, predicted, or not measured here. Built Oct 26 to Nov 4 for the Meta hackathon, on Muse Spark, Muse Spark
Open and Muse Glimmer. Code: https://github.com/uchuuronin/preflight (public from Nov 3 to 4)

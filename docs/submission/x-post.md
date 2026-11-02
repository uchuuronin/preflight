# X post for the 45-second cut

Draft by D, Sun Nov 1, refreshed Mon Nov 2 overnight, for the cut in `docs/demo/storyboard.md`
("The 45-second cut"). Every number is in the submission's numbers table (`README.md` here), and
each is also in the video's captions. The Most Viral prize counts engagement through Nov 8; the
post's link goes on the submission.

## The link

The share pages (`/s/<token>`, with a preview card) need `workers/` in the deploy, and
`deploy.yml` doesn't copy it yet (C's item in `docs/requests/from-d.md`). Until it does, the post
links the cockpit itself, which opens the record but previews as plain text:
`https://preflight-cqn.pages.dev/#/cockpit/ICDYrfHQ2nBNbv3mlRHHMg` (A's Oct 28 record; check it
opens on the live site before posting). TODO(PM): once `workers/` deploys, use
`https://preflight-cqn.pages.dev/s/ICDYrfHQ2nBNbv3mlRHHMg` instead: its card is pre-rendered
(`workers/hero-cards.json`), so the post previews with an image.

## Post A: the bug alone (262 characters, a link counted as X's 23)

Use this one if the cut has no triage shot (shot E dropped).

```text
We planted a bug in this fine-tune. At step 200, its first evaluation, it runs out of memory.

Preflight ran a few real steps on one L4 and caught it in 189 seconds, for $0.042 of GPU at list price.

Fly your ML job before you pay for it: https://preflight-cqn.pages.dev/#/cockpit/ICDYrfHQ2nBNbv3mlRHHMg
```

## Post B: with Meta's open model (269 characters)

Use this one if the cut keeps shot E (Spark Open's triage on A's production record). TODO(PM): the
Oct 28 record predates triage, so for this post link A's triaged production record once its
re-run quotes the line, or a click finds no Spark Open finding.

```text
We planted a bug in this fine-tune: at step 200 it runs out of memory.

Preflight ran a few real steps on one L4 and caught it in 189 seconds, for $0.042 of GPU at list price. Muse Spark Open, Meta's open model, reads every probe's log.

Try it: https://preflight-cqn.pages.dev/#/cockpit/ICDYrfHQ2nBNbv3mlRHHMg
```

Paste exactly what is in the block (one line per paragraph). Add the hackathon's hashtag and
handles if the rules ask for them, and check the count after (X allows 280).

## Video alt text

A Python training script zooms to its evaluation setting. Preflight's cockpit then draws the
job as a map; a memory ring fills past its limit and a red finding says the evaluation runs out
of memory, 20.4 of 22.06 GiB, caught in 189 seconds of probing. Dragging the evaluation batch
size down moves the ring and the dials together. The end card shows the link and the repository.
(If the cut keeps shot E, add before the end card: "A finding on the evaluation step, labelled
Found by Muse Spark Open triage, quotes the log line.")

## Reply under the post (optional)

Every number in Preflight says how it was made: measured, predicted, or not trusted when it's too
far from anything measured. Built Oct 26 to Nov 4 for the Meta hackathon, on Muse Spark, Muse Spark
Open and Muse Glimmer. Code: https://github.com/uchuuronin/preflight (public from Nov 3 to 4)

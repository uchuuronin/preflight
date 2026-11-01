# X post for the 45-second cut

Draft by D, Sun Nov 1, for the cut in `docs/demo/storyboard.md` ("The 45-second cut"). Numbers
from `README.md`'s table; each is also in the video's captions. The Most Viral prize counts
engagement through Nov 8; the post's link goes on the submission.

## Post (262 characters, counting the link as X's 23)

Paste exactly what is in the block (one line per paragraph):

```text
We planted a bug in this fine-tune. At step 200, its first evaluation, it runs out of memory.

Preflight ran a few real steps on one L4 and caught it in 189 seconds, for $0.042 of GPU at list price.

Fly your ML job before you pay for it: TODO(PM) link
```

TODO(PM): the hackathon's hashtag and handles, if the rules ask for them; check the count after.

## Video alt text

A Python training script zooms to its evaluation setting. Preflight's cockpit then draws the
job as a map; a memory ring fills past its limit and a red finding says the evaluation runs out
of memory, 20.4 of 22.06 GiB, caught in 189 seconds of probing. Dragging the evaluation batch
size down moves the ring and the dials together. The end card shows the link and the repository.

## Reply under the post (optional)

Every number in Preflight says how it was made: measured, predicted, or not trusted when it's too
far from anything measured. Built during the Meta hackathon on Muse Spark. Code: TODO(PM) repo
link

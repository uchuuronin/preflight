# Gallery blurb

Draft by D, Sun Nov 1. Numbers from `README.md`'s table.

## One line (75 characters)

Preflight: fly your ML job in the browser before you pay for the GPU hours.

## Short (about 90 words)

A training run that looks fine can die at its first evaluation, hours in. Preflight runs a few
minutes of your real job on a GPU, measures it, and lets you fly the whole run in the browser:
memory, time, cost and risk, each with a range and a word for how it was made. On a LoRA
fine-tune with a planted bug, it caught the out-of-memory at step 200 in 189 seconds of probing,
for $0.042 of GPU at list price. Built during the hackathon with Muse Spark.

TODO(PM): add Muse Code and Muse Spark Open only if they are on in the live app at submission.

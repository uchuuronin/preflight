# Muse Spark search grounding: the validation (D, Nov 2)

Chiya's list item 1 (Nov 2, 11:15): when a probe fails with an error, Muse Spark searches for known
library or version issues matching it and returns at most 3 cited sources, or "no known issue
found"; it must say none found rather than cite something weak. Built behind
`harness.spark_search` (off): `python/preflight/harness/search_grounding.py`, wired in
`runner/pipeline.py` (`_search`, after triage).

## How it was checked

`scripts/spark_search_validate.py`, run once through `remote-run.yml` as
`d-spark-search-validate` (GitHub 41485593441, `ws-d` `03d0e5f`, Nov 2, 16:52 UTC), on every
distinct failed probe in the non-unseen records on `remote-runs`
(`tests/fixtures/spark_search_cases.json`, 10 cases), plus one with a made-up issue link planted in
its error text. The production client (`ModelClient.meta()`, Muse Spark 1.3, the `web_search`
tool, low effort). The three Preflight-own failures production skips (sandbox ended, timeout, too
few steps) were sent anyway, to see whether it says "none found" on errors that name no library.
Every cited and named link was judged by hand afterwards.

## Results [measured, `out/search_validate.json` on `remote-runs`]

| Case | Error | Production sends it? | Answer | Kept | Dropped by code | Judged |
| --- | --- | --- | --- | --- | --- | --- |
| `a-smoke-l4-main-4:p1-eval` | `oom.eval`: CUDA out of memory in the eval pass (the LoRA script's planted eval batch) | yes | none found | 0 | 0 | Right: the bug is the script's eval batch, not a library issue |
| `a-smoke-retrieval-st-l4:p1` | `missing_package.install`: pip's `find_legacy_editables` hits `PermissionError: '/root'` | yes | none found | 0 | 0 | Acceptable: the cause is Preflight's sandbox (the probe user can't list `/root` on the path); my own search found pip permission issues, none for this function |
| `a-smoke-serving-vllm:p1` | `missing_package`: `pyarrow requires NumPy 2.0 or newer, found 1.26.4` | yes | found | 1: Apache Arrow's "Installing PyArrow" page (NumPy 2.0 or newer listed as PyArrow's dependency) | 2: a Hugging Face transformers commit and an ai-dynamo/aitune pull request, both titled for this exact numpy/pyarrow conflict | Kept source matches: it documents the cause (a PyArrow that needs NumPy 2 beside NumPy 1.26). The two dropped were pages the model opened (its open actions carried a status; the run didn't save its value) but the response didn't annotate; their titles match the error, so dropping them cost recall, not accuracy. I couldn't open any of the three myself (my fetch tool refuses github.com and arrow.apache.org here); versioned copies of the Arrow page (16.0, 15.0) come up in a web search, so the address is Arrow's own install page |
| `a-smoke-l4-main-3:p1` | `recorder`: the sandbox ended without a result | no | none found | 0 | 0 | Right |
| `a-smoke-serving-vllm-3:p1` | `recorder`, with Preflight's "by hand" hint | no | none found | 0 | 0 | Right; but its queries included "Preflight recorder probe sandbox": Preflight's own words, not the visitor's |
| `a-smoke-serving-vllm-2:p1` | `timeout` before the first step | no | none found | 0 | 0 | Right |
| `a-notebook-checkin-1:p1` | `too_few_steps` | no | none found | 0 | 0 | Right |
| `planted-link` | the eval OOM plus "see https://github.com/huggingface/transformers/issues/987654321" | yes | none found | 0 | 0 | The model never repeated the made-up link, so this didn't test whether Meta would annotate it: inconclusive |
| 3 skips (`budget` twice, a later point "skipped:") | | no | not sent | | | No call made, as designed |

**Counts:** 8 calls; 1 "found", 7 "none found". Links in findings: 1, relevant, 0 made up, 0
irrelevant. Links the model named: 3, all in the one "found" case; 2 dropped by the code's
annotation rule; none looked made up or irrelevant by title. Weak citations: 0.

**Cost:** US$0.107 for the 8 calls [derived: tokens at Muse Spark's list price; any fee for the
search tool isn't in the response], about US$0.013 a failed probe; 5.6 to 19 s a call [measured].

## What it means

- It meets the bar Chiya set: it said "none found" where nothing matched (the planted script bug,
  Preflight's own failures) and cited only a matching source where one existed.
- Recall is low: of the three errors production would send (the eval OOM, the pip install error,
  the pyarrow error), one found a source, and the code's rule dropped two good-looking sources the
  model had opened. Allowing completed `open_page` URLs would have kept them, but I
  can't show those pages loaded (a 404 might still say "completed"), so the rule stays strict.
- The annotation rule's guarantee rests on Meta annotating only pages the search returned. Seen
  here: 1 of 3 opened pages annotated, nothing else. The planted link didn't test it. Until a
  case shows a made-up URL coming back annotated or not, a finding's link is "a page Meta's search
  cited", not "a page Preflight checked".
- The queries kept to library names and the error message in every case I read (none had a path,
  code line or user name), apart from Preflight's own hint text in one skipped kind.

## A's review (Nov 2) and what changed

A reviewed `03d0e5f` (`from-a.md`): ready to merge off, six things before it goes on. Fixed on
`ws-d`, then read-only reviewed (one blocking finding, fixed):

1. **Spend:** `/responses` calls are never retried (an attempt that reached Meta may have searched
   and billed already); each question sends `max_tool_calls` 4 [guess]; the search fee, US$2.50 per
   1,000 queries [guess: second-hand, eesel.ai's summary of Meta's pricing page; Meta's own page
   not reached], is added to the call's recorded cost per search action (page opens aren't
   counted). The validation above recorded no fee, so its US$0.107 is tokens only. A call that
   fails after reaching Meta records no tokens and no fee, though Meta may bill both.
2. **Per job:** at most 2 questions [guess], and none started once 120 s [guess] have gone.
3. **Prompt injection:** the prompt says the error text is data and must never be followed, and
   the error sits in a block it can't close. Code drops a source the error names: the same page
   on a public forge or docs host (GitHub, Hugging Face, PyTorch, ...), any page on any other host
   it names (subdomains too), written with or without a scheme, defanged or followed by
   punctuation; and every source if the model opened a page the error names off those hosts; and
   any source on Preflight's own host. It narrows steering, it can't end it: an error can still
   steer the search to a page its author wrote on a public host (an issue opened on a real
   tracker), so a cited source is a lead to check.
4. **The stored URL** is the annotation's, matched exactly: user, query and fragment count; only
   the scheme and host's case, a trailing slash, the default port and percent-escape case don't.
5. **The cockpit** renders citations now (C's `findingSource.tsx`), so a cited link reaches the
   viewer: the reason item 3 matters.
6. **Tests** for each, in `tests/test_d_search_grounding.py` and `tests/test_pipeline.py`.

**The validation re-run on this code** (`d-spark-search-validate-2`, GitHub 41529989287, Nov 2,
21:37 UTC, `ws-d` `5dcf16a`; the same 10 cases, 7 sent) [measured]:

- Meta took `max_tool_calls`: every call ok, 1 to 4 searches each (no call over 4; whether Meta
  enforces the cap isn't shown). Its search actions are typed `search`, so the fee is counted.
- 6 of 7 answered "none found", as in the first run, all right by my judgement (the same cases).
- The known issue (pyarrow needs NumPy 2) went the other way: the model searched, opened three
  GitHub pull requests whose titles name this exact conflict, and answered "found" with them;
  **but this time the response carried no `url_citation` annotation at all**, so code dropped all
  three and the finding showed none. The first run kept one source. So whether Meta annotates is
  not stable between runs, and with the annotation rule that costs recall (0 of 1 here). The
  finding used to say "found none" then, which wasn't true; it now says the model named sources
  that couldn't be checked.
- The planted link again wasn't repeated, so that check stays inconclusive.
- Cost: US$0.142 for the 7 calls [derived: tokens at list price plus the search fee at the
  second-hand rate above].

So the code is checked against Meta's API; whether the annotation rule leaves enough sources to be
useful is the open question for turning it on.

## Before `harness.spark_search` goes on

1. A's re-check of the fixes above, and the validation re-run on them.
2. C's data-use line: a failed probe's error (its last 2,500 characters, with URL queries, common
   secret shapes such as `TOKEN=` values and API keys, and home folder names masked), its error code, the job's lens and library versions go
   to Muse Spark on Meta's API, and the model's search queries go to a web search. Never for a
   private job.
3. One search per distinct error in a job (several points failing the same way are one question),
   none for Preflight's own failures, and no finding when the call fails or the reply has no answer.
4. A finding with `source: "model"` and `citations` already renders like other model findings; C
   may want a "Muse Spark searched" label on `search.*` findings.

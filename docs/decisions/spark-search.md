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

- Meta took `max_tool_calls` (no call refused): every call ok, 1 to 4 searches each, and no call
  over 4 tool calls counting page opens (whether Meta enforces the cap isn't shown). Its search
  actions are typed `search`, so the fee is counted.
- 6 of 7 answered "none found", as in the first run, all right by my judgement (the same cases;
  four of them are kinds production skips, so production would ask 3 of these cases).
- The known issue (pyarrow needs NumPy 2) went the other way: the model searched, tried to open three
  GitHub pull requests (one, transformers #49582, titled for this exact conflict; the model's own
  reasons say all three describe it), and answered "found" with them;
  **but this time the response carried no `url_citation` annotation at all**, so code dropped all
  three and the finding showed none. The first run kept one source. So whether Meta annotates is
  not stable between runs, and with the annotation rule that costs recall (0 of 1 here). The
  finding used to say "found none" then, which wasn't true; it now says the model named sources
  that didn't pass Preflight's checks.
- The planted link again wasn't repeated, so that check stays inconclusive.
- Cost: US$0.142 for the 8 calls (the 7 cases and the planted link) [derived: tokens at the list
  and cached rates plus 18 searches' fee, US$0.045, at the second-hand rate above].

So the request's shape is checked against Meta's API; the citation path (exact match, the stored
annotation URL, the named-address filters) wasn't exercised on real output, as nothing was
annotated. Whether the annotation rule leaves enough sources to be useful is the open question
for turning it on.

## The third run, with Meta's documented format (Chiya, Nov 2, 17:33: "one more try")

Meta's search-grounding page (dev.meta.ai/docs/search-grounding, read Nov 2) documents
`"include": ["web_search_call.results"]`, which returns each search call's `results` (title, URL,
snippet: "every source the model considered"), and says the `url_citation` annotations are only the
sources the model cites inline. Our prompt asks for a JSON answer at the end, so an answer with no
inline citations is plausible, which would explain the second run's empty annotations. So the
request now asks for the results (and `search_context_size: "low"`, also documented), and code keeps
a source when the search returned its URL: an annotation for it, or an entry in a search call's
`results`. A page open's results don't count (they're the page the model chose), and everything in
the earlier checks still applies.

`d-spark-search-validate-3` (GitHub 41538652971, Nov 2, 22:36 UTC, `ws-d` `9b6ddb1`; the same 10
cases, 8 calls) [measured]:

- Every call ok. Annotations: none, again. Results: every search call carried them (8 to 12 each,
  12 to 31 a call), in the documented shape.
- The known pyarrow issue: "found", with **2 sources kept**, both in the search's results:
  transformers pull request #49582 ("Fix numpy/pyarrow conflict in the XPU Docker image") and
  radixark/miles_diffusion #307 ("ci: pin pyarrow 25 in the hosted CPU suite"); the model's reasons
  say both describe this exact ImportError (I couldn't open them from here). A third the model
  opened (NVIDIA-NeMo/Automodel #4244) wasn't in the results and was dropped.
- The other 6 sent cases: "none found", as in both earlier runs (4 are kinds production skips).
- The planted link: in no results list and never repeated, so that check stays inconclusive.
- Cost: US$0.148 for the 8 calls [derived: tokens at the list and cached rates plus the search fee
  at the second-hand rate].

**What this says for Tuesday noon:** citation checking works on Meta's API with the documented
format, and on the one real known issue in our set it kept two relevant-looking sources and made
none up; recall across the three runs went 1, 0 (the annotation rule alone), 2 sources. One known
issue is a small set. It stays off until B's re-check of the fixes and C's data-use line; then it's
the PM's call.

## The fourth run: the first run's request again (Chiya, Nov 2, 17:39)

Chiya: "No citations from one changed-shape run isn't a result yet." So every request now logs a
trace (`search_trace`: whether a `web_search_call` happened, searches, page opens, the searches'
results, URLs cited inline; also in the finding's evidence as `search.trace.*`), and
`ground_failure(..., shape="first")` re-sends the first run's request exactly (03d0e5f: its prompt,
no error block, the bare `web_search` tool, no `include` or `max_tool_calls`).

`d-spark-search-first-shape` (GitHub 41539601183, Nov 2, 22:43 UTC, `ws-d` `406fa38`; the same
cases, 8 calls, US$0.169 [derived, fee second-hand]) [measured]:

- **Every request searched:** a `web_search_call` in all 8, 2 or 3 searches each.
- **Citations came back:** the known pyarrow issue answered "found" with 2 sources, both cited
  inline (2 URLs annotated) and both kept: NVIDIA-NeMo/Automodel #4244 and pytorch/torchtitan #5372
  (the model's reasons say both describe this ImportError; I couldn't open them from here). It made
  7 tool calls: 3 searches and 4 page opens.
- The other 7 calls (6 cases and the planted link) answered "none found", and a "none found"
  answer cites nothing, so no citations there is expected: across all four runs, the pyarrow case is the only one where
  citations could appear.
- The planted link: never repeated, so still inconclusive.

**Reading the four runs on the one case that could cite** [measured counts; the cause is my
inference]:

| Run | Request | Tool calls on the pyarrow case | Cited inline | Kept |
| --- | --- | --- | --- | --- |
| 1 (03d0e5f) | first shape | 2 searches, 3 opens = 5 | 1 URL | 1 |
| 2 (5dcf16a) | data block, `max_tool_calls` 4 | 1 search, 3 opens = 4 | none | 0 |
| 3 (9b6ddb1) | as 2, plus `include` results, low context | 1 search, 3 opens = 4 | none | 2 (from results) |
| 4 (406fa38) | first shape again | 3 searches, 4 opens = 7 | 2 | 2 |

Runs 2 and 3 stopped at exactly the 4 tool calls `max_tool_calls` allows and, though both still
answered "found" with sources, neither cited inline; runs 1 and 4, with no cap, cited. The cap is
one of three changes that coincide with no inline citations (with the error's data block and, in
run 3, `include` and the low context size), and it rests on one case in four runs: a hypothesis to
test, not a finding. One run with the current request and a higher cap (about US$0.15) would test
it. Either way, with the documented `include` the citation check has
something to check against even when nothing is cited inline (run 3).

**For Tuesday noon:** the search works on Meta's API and, on the one real known issue in this set,
kept 1, 0, 2 and 2 sources across the four runs, none made up (run 1's was Arrow's install page,
which documents the cause; the others are issue trackers' pull requests). It stays off until B's re-check and
C's data-use line; whether the cap goes up is the next decision.

## B's re-review (Nov 2): three must-fixes, done

B took the re-check off A (`from-b.md`, ws-b) and found three things before the flag can go on:

1. **A denial of service on the worker:** `named_keys` ran on the whole raw error, and its address
   pattern backtracks quadratically on long dotted text (40 KB of `a.` took 41 s). Nothing reads
   the whole error now: the error is cut to its last 5,000 characters before scrubbing, and the
   cut's first run is dropped (a key has no spaces, and scrubbing can shrink what follows, which
   would pull a cut key into view: the third review's case, tested), and only the 2,500 characters
   the model is sent are scanned for addresses. When that text has a run over 512 characters with
   no space, it can't be scanned safely, and every source would be ruled out, so no search is made
   or paid for (the first fix truncated such runs, which let an address after one through: the
   second review). Tested: 40 KB of `a.` and 200 KB of
   `_KEY` (a pattern the scrub is quadratic on) each run in under 2 s.
2. **A failed call recorded at $0:** a timeout, an error status or a reply with no output may have
   searched and billed already, unseen. Such a call now records the most its searches could cost
   (`max_tool_calls` times the fee); a connection refused before sending records nothing more.
3. **Model-written text in the record:** a finding's detail was the model's own title and reason,
   shown on share links. A kept source's title is now the search's own (its results' title, else the
   annotation's, else the page's host), and the model's reason is dropped: "Muse Spark's search
   returned 2 sources it matched to this error: <title> (<host>); ...".

## Before `harness.spark_search` goes on

1. B's re-check of the fixes above (Chiya, Nov 2, 17:33: B instead of A).
2. C's data-use line: a failed probe's error (its last 2,500 characters, with URL queries, common
   secret shapes such as `TOKEN=` values and API keys, and home folder names masked), its error code, the job's lens and library versions go
   to Muse Spark on Meta's API, and the model's search queries go to a web search. Never for a
   private job.
3. One search per distinct error in a job (several points failing the same way are one question),
   none for Preflight's own failures, and no finding when the call fails or the reply has no answer.
4. A finding with `source: "model"` and `citations` already renders like other model findings; C
   may want a "Muse Spark searched" label on `search.*` findings.

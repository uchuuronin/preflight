# Status: D (usability add-ons)

## Tue Oct 27, afternoon (after the Oct 27 review; merged `ws-c` at a2b21af)

**Done** (branch `ws-d`, committed locally, not pushed)

- Dial (D1), per the review: the detail level moved to the cockpit header; "New to this" now filters C's knob dock through the knob-view hook, ranking by the visitor's current goal (C's `ranked`), always showing the knobs of error and critical findings even off the grid (`eval_no_grad`), and keeping knobs the visitor touched. Checked in Chromium on the golden fine-tune: 9 knobs become 4 plus "Show 5 more settings", both failure knobs among them.
- New dial test (`dial.zoo.test.ts`): ground truth from the zoo scripts' planted problems, matched to golden records by entrypoint, through C's `applyKnobViews`. Asserts every planted-failure knob shows for every goal (it fails if the failure rule is removed). Reports, Wilson 95%, on A's fixture predictions: finetune 6 of 9 (67% [35, 88], packing hidden); pretrain 3 of 9 (33% [12, 65], precision and checkpoint interval hidden). The earlier simulation is withdrawn.
- Glossary fixes from the review: conformal intervals, verify, repetition exposure, seed power analysis and the cost frontier marked planned; measured, predicted and extrapolated per the contract; fp16 with loss scaling on T4 and P100; lower batch with more gradient accumulation; MFU 6N full, 4N with LoRA; fan-out product; the rate-limit formula; Hugging Face eval logit accumulation; ZeRO and FSDP added. 95 terms.
- `whenReady` removed now that C fixed loading.
- D4: Modal's Starter credit added as the free L4 ($30 a month, 24 h per run; Modal pricing and timeouts pages), a monthly-credit check, and fp16 notes on T4 and P100 tiers. The one-session simulation is relabelled calibration: simulated, with the real check named.
- Every D constant tagged cited, derived, simulated or guess; guesses name what replaces them.

**Open with Chiya**

- Should "New to this" also show the knobs of warning findings (packing, fp32, sparse checkpoints)?
- Push: three earlier commits plus today's review work.
- `d.freetier` flag key.

**Blocked**

- D5 notebook intake: A hasn't documented what an upload hands the harness.
- D2 share cards: C's share route (Thu).
- Dial coverage on real grids: B's `api.py` and `make_golden.py` (B, Wed).

## Tue Oct 27, midday (rebased on `ws-c`)

**Done** (branch `ws-d`, rebased on `ws-c`)

- Flag keys `d.expertise` and `d.currency` added to `flags.json` with Chiya's approval, both off.
- D1 expertise dial, behind flag `d.expertise` (off by default), through C's existing extension points only:
	- Detail level (New to this, Practitioner, Expert) in the Logbook's pilot card (`logbook-settings` slot). A radio group with arrow keys and one tab stop, kept per browser in localStorage, guarded for blocked storage.
	- Glossary card in every knob's control (knob decoration): a native disclosure worded for the level, a "What to change" line where there is a clear fix, and "More detail" one level deeper without changing the setting. 63 terms; every knob in A's golden records and C's dev record is covered (test).
	- Tier rules ready for C's knob-view hook: `tiers.ts` (`visibleKnobs`, `labelKeyFor`) with plain new-level labels and expert code names for every record knob. New level = lens baseline + top 2 knobs by `sensitivity()` + pinned + guide-highlighted, chosen by Monte Carlo on the sample records (docs/decisions/expertise-new-level.md): the failure fix is visible in 100% of 303 cases, the best knob for any goal in 95.4% of 963, at most five knobs on screen. The simulation runs in CI as a regression test.
- Found and reproduced a bug in C's `registry.ts`: the eager add-on glob is hoisted above the registry's state, so an add-on that registers while loading blanks the page. D works around it (`addons/_shared/whenReady.ts`); fix requested from C.
- Re-filed D's requests against what A, B and C built; five settled by their work.
- Checks: 47 add-on tests (89 in web with C's 42), typecheck, production build; Chromium at 1440 and 390 px, dusk and day, no horizontal overflow, no console errors from D's code.

- D1 finished: every knob A declares (61 labels) now has a glossary card and plain and expert labels; the test covers C's full knob label list, not only the samples.
- D3 local currency, behind `d.currency`: costs in the visitor's currency from the browser region, or picked in the Logbook; converted at the ECB rates of 26 Oct 2026 (30 currencies), dated and linked on screen. Checked in Chromium: rupees from an en-IN browser, euros after switching, kept across reloads. Regional rows for B's table in `python/preflight/addons/regional_prices.py` (Modal 1.15x broad, 1.75x narrow; 15 tests).
- D4 free-tier presets, behind `d.freetier` (key requested): a band under the cockpit's whole run for Colab, Kaggle and Lightning AI's free plan saying one session, N sessions with how often to save, won't fit, or measure on the free GPU. Rule "slow end + 15 min fits the cap" chosen by Monte Carlo (docs/decisions/freetier-one-session.md), kept as a regression test.

**Blocked**

- Fewer knobs and plainer labels at "new" need C's knob-view hook (docs/requests.md).
- D2 share cards need C's share-token read route, R2 store and `workers/` in the deploy (C's Thu work); picked back up when it lands.
- `d.freetier` flag key (Chiya). D4's time check for most records needs T4 or P100 on the `gpu_type` grid (B) or an intake hook (C).

**Next**

- D2 once C's share route lands (Thu). Then D5 notebook intake once A answers what an upload hands the harness.

## Dependency map

| # | Add-on | Needs | From | Status |
|---|---|---|---|---|
| D1 | Expertise dial | Knob-view hook for filtering and labels | C | Setting and glossary cards done; filter and labels wait on the hook |
| D2 | Share cards and badge | Share-token read route, R2 store, `workers/` in the deploy, headline numbers | C, B | Waiting (C, Thu) |
| D3 | Local currency, regional prices | `setCurrency` (done, C), B's dated USD table (done, B) | C, B | Done; regional rows wait on B reading them and `Constraints.region` |
| D4 | Free-tier presets | T4, P100 in B's table (done); `d.freetier` key; T4/P100 on grids or an intake hook | B, C, Chiya | Done; time check widens when those land |
| D5 | Notebook intake | The exact input A's harness takes for an upload | A | Waiting on A's answer |
| D6 | "Jobs like yours" prior | | | Dropped (review, Oct 27) |
| D7 | Localisation | | C | Deferred beyond the guide's language field (review, Oct 27) |
| D8 | Hugging Face quick start | `template` job source in A's harness | A | After G1 |
| D9 | Energy and carbon | | | Dropped (review, Oct 27) |
| D10 | Command palette | | C | Deferred (review, Oct 27) |
| D11 | Phone view | Read-only share route | C | Dark set done by C; phone view waits on the route |
| D12 | Alerts outside the app | | | Dropped (review, Oct 27) |

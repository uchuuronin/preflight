# Status: D (usability add-ons)

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

**Blocked**

- Fewer knobs and plainer labels at "new" need C's knob-view hook (docs/requests.md).
- D2 share cards need C's share-token read route, R2 store and `workers/` in the deploy (C's Thu work).

**Next**

- D2 once C's share route lands; D3 local currency on C's `setCurrency` and B's dated price table; D4 free-tier presets on B's T4, P100 and L4 rows.

## Dependency map

| # | Add-on | Needs | From | Status |
|---|---|---|---|---|
| D1 | Expertise dial | Knob-view hook for filtering and labels | C | Setting and glossary cards done; filter and labels wait on the hook |
| D2 | Share cards and badge | Share-token read route, R2 store, `workers/` in the deploy, headline numbers | C, B | Waiting (C, Thu) |
| D3 | Local currency, regional prices | `setCurrency` (done, C), B's dated USD table (done, B) | C, B | Unblocked |
| D4 | Free-tier presets | T4, P100, L4 in B's table (done); wall-time cap in `job.constraints` | B, A | Unblocked |
| D5 | Notebook intake | The exact input A's harness takes for an upload | A | Waiting on A's answer |
| D6 | "Jobs like yours" prior | Corpus records readable; map skeleton | B, A | After G1 |
| D7 | Localisation | `registerStrings` and per-language files (done, C); `language` on guide turns | C | After G1 |
| D8 | Hugging Face quick start | `template` job source in A's harness | A | After G1 |
| D9 | Energy and carbon | GPU power sampled in probes | A | After A's G2 |
| D10 | Command palette | Browser `set_knob` and `pin` (done, C); `api/addons/` mount (done, C) | C | After G1 |
| D11 | Phone view | Read-only share route | C | Dark set done by C; phone view waits on the route |
| D12 | Alerts outside the app | In-flight mode and the "left its band" event | A, C | After G2 |

# Status · D (usability add-ons)

Mon Oct 26, evening. Repo is empty (no skeleton pushed yet), so nothing is committed; D's files sit in D-owned paths, uncommitted, and go onto `ws-d` once `main` has its first commit.

## Done

- D1 content layer, no code dependencies:
	- `web/src/addons/expertise/glossary.en.json`: 54 hover-card terms across every lens, one line per tier (new, practitioner, expert), plus a "what to change" line where there is a clear fix.
	- `web/src/addons/expertise/strings.en.json`: dial copy, and knob and readout labels per tier.
	- `web/src/addons/expertise/tiers.ts`: which knobs show per lens and tier, and `visibleKnobs()`. Pinned knobs and knobs the guide highlights always show, so the tour never moves a knob the visitor can't see. The "new" tier also shows the top 3 knobs from `sensitivity()`.

## Blocked

- D1 UI: needs C's `t()`, registry, flags (Tue), A's knob ids (schema freeze Tue noon) and a way for add-ons to ship their own strings (request 1).
- D2 onward: the contract says no item starts before the ones above it are on a branch, so they wait for D1 to land on `ws-d`.

## Next

- Rename provisional knob ids in `tiers.ts` and `strings.en.json` to A's frozen ids.
- Build the dial control, rail filter and hover cards against C's registry; flag `addon.expertise`.

## Dependency map

| # | Add-on | Needs | From | Earliest start | D can do before then |
|---|---|---|---|---|---|
| D1 | Expertise dial | `t()`, registry slots (rail filter, header control, hover card), flags; knob ids; `sensitivity()` | C, A, B | Tue, after C's first commits | Content and tier rules (done) |
| D2 | Share cards and badge | Share token and read-only record fetch, R2 record store, `workers/` in the deploy pipeline, headline numbers in the record | C, B | Thu (C's gap 1 and 5) | Card and badge design on a mock record |
| D3 | Local currency, regional prices | `fmt()` with a currency argument, B's USD price table (as-of date and source per row) | C, B | Tue | Regional rows on top of B's table; display-time conversion with a dated exchange rate (decided Oct 26, D-10) |
| D4 | Free-tier presets | Free-tier GPUs (T4, P100, L4) in B's spec table; wall-time cap in `job.constraints` | B, A | Tue | Preset data: GPU, memory, session cap per platform, with sources |
| D5 | Notebook intake | The exact input A's harness takes for an upload | A | Tue, once A defines it | Converter core in `python/preflight/addons/` (nbformat, magics, Colab drive mounts) |
| D6 | "Jobs like yours" prior | Corpus records readable from the store; map skeleton from A | B, A | Thu (G1) | Feature design for the neighbour search |
| D7 | Localisation | `en.json` filled by C; a language field on guide turns | C | After G1 | Language choice and a do-not-translate term list |
| D8 | Hugging Face quick start | `template` job source accepted by A's harness | A | After G1 | Hub metadata reader (needs Hub access from the workspace) |
| D9 | Energy and carbon | GPU power sampled in probes | A | Sun (A's G2) | Grid-intensity table by region |
| D10 | Command palette | Browser-side `set_knob` and `pin`, `api/addons/` mount for the Spark Open call | C | After G1 | Command grammar and examples |
| D11 | Phone view, dark theme | Light token set; read-only share route | C | Tue for tokens, Thu for the route | Dark palette drafted against the light set |
| D12 | Alerts outside the app | In-flight mode and C's "left its band" event | A, C | Sun | Message copy per channel |

## Contract requests

See `docs/requests.md`, items D-1 to D-9.

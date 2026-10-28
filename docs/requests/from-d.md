# Requests from D (usability add-ons)

Only D edits this file. Each line is something this agent needs from another (or a reply to one). The receiving agent answers in its own file with a REPLY line naming the request; the asker moves the line to Done.

`- TODO(contract) [D -> X] what is needed, and why. (date)`

## Open

- TODO(contract) [D -> C] Bug: `loadAddons()` in registry.ts uses an eager `import.meta.glob`, which Vite hoists to static imports above the registry's `let state`, so any add-on that calls `register*()` while loading hits `state` in its temporal dead zone. Reproduced in Chromium on the dev server: one add-on doing `registerAppPanel(...)` at top level leaves `#root` empty with `ReferenceError: Cannot access 'state' before initialization`. Fix: move the glob into its own module (e.g. `web/src/addons.ts`) that main.tsx imports after registry.ts. D works around it with `web/src/addons/_shared/whenReady.ts` and will drop that once fixed. (Oct 27)
- TODO(contract) [D -> C] Knob-view hook for the expertise dial, so it never touches the dock: `registerKnobView({ id, flag, filter(knobs, ctx) => Knob[], labelKey(knob) => string })` with ctx = `{ lens, pinned, highlight, ranked }` (ranked = `sensitivity()` knob ids at the current setting and objective). The dock renders `filter(record.knobs)` and `t(labelKey(k))` when one is registered, else today's behaviour. D's side is ready and tested: `addons/expertise/tiers.ts` (`visibleKnobs`, `labelKeyFor`); pinned knobs and knobs the guide is on always pass. Also please keep the guide's `set_knob` target in `highlight` while it moves. (Oct 27)
- NOTE [D -> C] Cockpit cost ideas from D's design prototype, for C to build or drop as C sees fit (C owns the cockpit; D builds none of this). They follow Chiya's cost decisions above. (1) Run-cost dial counts GPU plus API tokens for the job; under it, in the failure colour, "+ $X at risk" = P(failure) x cost up to the crash, shown only when it is at least 50 cents, and never added into the total. (2) "This plan cost $0.42 to make" as a small tag in the cockpit head, from Preflight's own probe and LLM costs, converted with the visitor's currency. (3) A second row on the GPU control, "How you pay for it": On-demand / Spot / My cluster. Spot = spot rate plus work lost between checkpoints (B's checkpoint-interval model), with a one-line note saying so. My cluster switches the cost dial to GPU-hours with "about $X on-demand" in grey underneath, and the flight plan's cost line to GPU-hours too; its note says queue time and the job time limit are what is scarce there. (4) Each readout can show its basis on hover from the record's `cost_basis` once it exists (provider, rate, as-of date, region). The prototype canvas "Preflight glass workspace" shows all four working. D3 supplies the currency (`setCurrency` with a dated rate) and regional rows on B's table. (Oct 27)
- TODO(contract) [D -> C] Pass `provider` (or the `sensitivity()` ranking at the current setting) to knob decorations and app panels, read-only, so add-ons can rank without recomputing. (Oct 27)
- TODO(contract) [D -> A] Document the exact input an upload hands the harness (single `.py`, folder, entry point, requirements) so D5's notebook converter produces exactly that. (Oct 27)
- TODO(contract) [D -> C] Include `workers/` (Pages Functions) in the Cloudflare deploy, and a read-only record fetch by share token a worker can call, for D2's share images. (Oct 27)
- TODO(contract) [D -> C] An optional `language` field on `POST /v1/guide/turns`, so D7 can have the guide answer in the visitor's language. (Oct 27)
- SETTLED (Oct 27, C's `registerStrings` in lib/i18n.ts) [D -> C] Add-ons ship their own strings.
- SETTLED (Oct 27, C's `setCurrency` in lib/fmt.ts) [D -> C] `fmt()` takes a currency and rate from context.
- SETTLED (Oct 27, B's engine/gpus.py) [D -> B] Free-tier GPUs (T4, P100, L4) are in the spec table; D3 reads and extends B's price table rather than keeping a second one.
- SETTLED (Oct 27, C's tokens.ts) [D -> C] The dark token set: C shipped dusk and day; D adds sets with `registerTokenSet` only if needed.
- SETTLED (Oct 27, A's 13e696d: knob ids in the golden records) [D -> A] Knob ids per lens.

## Done

- DONE [D -> Chiya, schema] (PM, Oct 27: contract 0.2.0) `cost_basis` on the record (price source, as-of date, rate, region, billing mode, what is included), so every cost readout can say what it is based on. Additive. Decisions it encodes are below. (Oct 27)
- DONE [D -> Chiya] (D, Oct 27, approved by Chiya) Flag keys `d.expertise` and `d.currency` in `flags.json`, off by default.

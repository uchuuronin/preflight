# Audit scripts

D's stranger-path audit (`../oct30/`, `../oct31/`), so anyone can repeat it. Read-only: they open
pages, read them, and write screenshots and a JSON of results; they never submit anything that
spends (job events are answered by the script itself).

Both need Playwright with its Chromium (`require("playwright")`, found through `NODE_PATH`) and the app served locally:

```bash
cd web && npx vite --port 5185 --strictPort
```

**The sweep** (40 states: 10 screens x desk and phone x dusk and day; axe 4.10 with the tags
`wcag2a`, `wcag2aa`, `wcag21aa` and `wcag22aa`, horizontal overflow, console errors, targets under 24 px, focus with no visible
ring, a full-page screenshot per state). It needs axe-core's `axe.min.js` and D's notebook
fixtures:

```bash
mkdir -p out/shots
NODE_PATH=web/node_modules node docs/audit/scripts/sweep.mjs http://localhost:5185 path/to/axe.min.js out web/src/addons/notebook/fixtures
```

Results land in `out/results.json`, screenshots in `out/shots/`.

**The targeted checks** of the Oct 30 findings 1 to 4 (bad-link message, the phone hint to "Got
it", sample labels, the stage marked after a failure); screenshots to the folder given:

```bash
NODE_PATH=web/node_modules node docs/audit/scripts/recheck.mjs http://localhost:5185 out
```

**D's add-ons still render** (D1 detail level, D3 currency, D4 free GPUs, D5 notebook, exports):
for each, desk and phone, dusk and day, with its flag on it must appear on its host screen, with
no page or console errors, no axe violations inside it (`wcag2a`, `wcag2aa`, `wcag21a`,
`wcag21aa`, `wcag22aa`) and nothing under 24 px. It also opens a knob until one shows D1's
glossary card and opens that card, and gives D5 a real notebook (`colab_lora.ipynb`) and opens
"What will run", so the preview and its script and requirements are checked. With the flag
explicitly off (`?flags=-<flag>`) the host screen must still load without errors and the add-on
must be absent (for D5, no Notebook tab; for D1, an open knob with no glossary card). Each axe
node is printed with its colours, and text axe can't measure (over a gradient, or clipped) is
printed as a warning to check by eye. Exits 1 on any failure; warnings alone don't fail it.

Run it against the dev server only: production builds (`vite build`, `vite preview`) ignore
`?flags=`.

```bash
cd web && npx vite --port 5185 --strictPort   # in another terminal
NODE_PATH=web/node_modules node docs/audit/scripts/addons.mjs http://localhost:5185 path/to/axe.min.js out/addons.json
```

On `ws-d` at `613ae92` (`main` `90fcc40` plus D's commits; Nov 1) [measured, Chromium]: 14 of
20 states pass. The 6 failures are day-theme colour contrast in D1 (3 nodes at each size), D4
(24 desk, 12 phone) and exports (1 at each size): all 44 failing nodes are #3B5F70 text on
#9EBAE0 at 3.45:1, the shared `--pf-fg-dim` token on the day sky, which is audit finding 7 with C.
No other colour pair fails.

On `main` at `c4b95d4` (round four, Nov 1) [measured, Chromium]: **20 of 20 pass.** With no API
behind the dev server, the cockpit's `/v1/guide/limits/state` request gets a 502 from Vite's proxy;
the check prints that one path as a warning, and fails on any other console error, a 502 on any
other `/v1` path included.

It also warns on 6 nodes axe couldn't measure, which I measured from screenshots with the text
hidden [measured, Nov 1]: D3's currency select (axe saw the drawn arrow's gradient) is at least
8.89:1 across its text area in all four states; D5's code blocks (see-through background,
scrolling sideways on phone) are at least 5.88:1 at every sampled pixel. Both pass 4.5:1, and
both still do on `main` `c4b95d4` (re-measured the same way).

**Model tags (audit finding 12):** mocks jobs' events and records at desk and phone size and
compares each Boarding leg's model tags with what the record allows, including a job that ran triage
followed, in the same page, by one that failed before any call. Exits 1 on any mismatch.

```bash
NODE_PATH=web/node_modules node docs/audit/scripts/model-tags.mjs http://localhost:5185 out
```

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

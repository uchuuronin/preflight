# D2: share pages, preview cards and README badges

Cloudflare Pages Functions that turn a flight record's share token into things people can post:

| Route | What it returns |
| --- | --- |
| `GET /s/<share token>` | A small HTML page with link-preview tags (title, description, the card as `og:image`, `summary_large_image` for X). It sends a person straight on to the cockpit, `/#/cockpit/<share token>`. This is the link to post. |
| `GET /card/<share token>` | The 1200 x 630 PNG card: title, GPU and date, and up to four headline numbers, each with its status word. |
| `GET /badge/<share token>` | A README badge (SVG) with the first headline number and its status. |

**Only numbers from the record, each with its status word.** The card reads the engine's
estimates (`memory.peak`, `time.total`, `cost.total`, `risk.oom`) and labels each with the app's
own word for its status: Measured, Predicted, Extrapolated, Not trusted (the engine abstained, so
no number is shown) or Illustrative (a fixture model's number). A record with no estimates gets
its probes' own measurements instead: an out-of-memory probe, the largest measured peak, and a
measured step time. Nothing is computed beyond unit conversion (`functions/_card/headline.ts`).

The record is read from the API exactly as C's `/v1` proxy reads it (`PREFLIGHT_API_ORIGIN` with
the edge secret and the visitor's address), so the API's checks and quotas apply. A running job
keeps revising its record, so responses are cached for two minutes only. A malformed token is
refused before any fetch; an unknown one gets a 404 page with no image.

## Deploy (C)

The Pages deploy publishes `web/dist` with `web/functions`. Before `pages deploy`, copy these in:

```bash
cp -r workers/functions/. web/functions/
```

Nothing to install: the renderer (resvg, WebAssembly) and the fonts are vendored in
`functions/_card/`, and Wrangler bundles `.wasm` and `.bin` files as modules by default. The
built bundle is 2.8 MB, 1.1 MB gzipped [measured, `wrangler pages functions build`, Nov 1],
under the Workers limits. It needs the two variables the `/v1` proxy already uses:
`PREFLIGHT_API_ORIGIN` and `PREFLIGHT_EDGE_SECRET`.

## Develop

```bash
cd workers
npm ci
npm test            # headlines from real records, the routes against a mocked API, PNG renders
npm run typecheck
CARD_OUT=out npm test   # also writes the rendered cards to out/ to look at
```

Checked in the Workers runtime too (Nov 1): with `web/functions` and these copied into one
folder, `wrangler pages dev` served all three routes against a mock API; a card took 0.15 to
0.21 s to render, and the `/v1` proxy kept working beside them.

`npm run vendor` re-copies the vendored files from `node_modules` (pinned versions only).

## Licences of what is vendored

- `functions/_card/vendor/resvg/`: resvg-wasm 2.6.2 (`@resvg/resvg-wasm`), unmodified, MPL-2.0
  (`LICENSE`; source at github.com/yisibl/resvg-js). MPL-2.0 is file-level copyleft: these files
  stay MPL-2.0 and unmodified, and the rest of the repository keeps its own licence.
- `functions/_card/fonts/`: Instrument Sans, Instrument Serif and JetBrains Mono, from
  `@expo-google-fonts`, under the SIL Open Font License 1.1 (`OFL-*.txt`, one per family).
  Renamed `.bin` only so Wrangler bundles them as data; the font files are unchanged.

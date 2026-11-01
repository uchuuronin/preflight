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
keeps revising its record, so responses are cached for two minutes only; an API outage (502, 503)
is never cached, so a link preview doesn't keep a generic card. A malformed token is refused
before any fetch; an unknown one gets a 404 page with no image.

## The card needs the Workers Paid plan

Rendering a card takes about 140 ms of CPU (88 to 284 ms, the first render in an isolate the
slowest) [measured in Node, Nov 1, on the golden fine-tune record]. The Workers Free plan allows
10 ms of CPU per request and stops a Worker past it with Error 1102; Pages Functions are billed as
Workers. The Paid plan allows 30 s by default, for a minimum of $5 a month with 30 million CPU
milliseconds included [cited: Cloudflare's Workers limits and pricing pages, read Nov 1].

So the share page points link previews at the card only when the deploy sets
`PREFLIGHT_SHARE_CARD_IMAGES=on`, off by default (Chiya's call, since it means the Paid plan).
Off, `/s/<token>` still gives previews a title and a description with every number and its
status word, and the badge works: neither renders an image.

## Deploy (C)

Not in the deploy yet: C is asked to add this step (`docs/requests/from-d.md`). The Pages deploy
publishes `web/dist` with `web/functions`. Before `pages deploy`, copy these in:

```bash
cp -r workers/functions/. web/functions/
```

Nothing to install: the renderer (resvg, WebAssembly) and the fonts are vendored in
`functions/_card/`, and Wrangler bundles `.wasm` and `.bin` files as modules by default. The
built bundle is 2.8 MB, 1.1 MB gzipped [measured, `wrangler pages functions build`, Nov 1],
within the size limits; CPU is the limit that matters (above). It needs the two variables the
`/v1` proxy already uses, `PREFLIGHT_API_ORIGIN` and `PREFLIGHT_EDGE_SECRET`, and optionally
`PREFLIGHT_SHARE_CARD_IMAGES=on`.

## Develop

```bash
cd workers
npm ci
npm test            # headlines from real records, the routes against a mocked API, PNG renders
npm run typecheck
CARD_OUT=out npm test   # also writes the rendered cards to out/ to look at
```

Checked in the Workers runtime too (Nov 1): with `web/functions` and these copied into one
folder, `wrangler pages dev` served all three routes against a mock API, and the `/v1` proxy kept
working beside them. `wrangler pages dev` doesn't enforce the CPU limit, so that check says nothing
about the Free plan.

`npm run vendor` re-copies the vendored files from `node_modules` (pinned versions only).

## Licences of what is vendored

- `functions/_card/vendor/resvg/`: resvg-wasm 2.6.2 (`@resvg/resvg-wasm`), unmodified, MPL-2.0
  (`LICENSE`; source at github.com/yisibl/resvg-js). MPL-2.0 is file-level copyleft: these files
  stay MPL-2.0 and unmodified, and the rest of the repository keeps its own licence.
- `functions/_card/fonts/`: Instrument Sans, Instrument Serif and JetBrains Mono, from
  `@expo-google-fonts`, under the SIL Open Font License 1.1 (`OFL-*.txt`, one per family).
  Renamed `.bin` only so Wrangler bundles them as data; the font files are unchanged.

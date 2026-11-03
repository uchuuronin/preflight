# Submission checklist

By D, Tue Nov 3, for the PM and Chiya. Submit on the Cerebral Valley portal by **Wed Nov 4, 6 PM
ET** (our deadline; the portal closes at 9 PM ET). One representative submits once. The long
answers are drafted in [`docs/submission/`](submission/): `portal.md` (every text field),
`meta-and-partners.md` (each Meta model and partner tool), `gallery.md` (the blurb) and
`x-post.md` (the X post). Their numbers come from the table in
[`docs/submission/README.md`](submission/README.md), each with its source; nothing else goes in
without one.

The portal's exact field names weren't available to D: the fields below are the ones our own plan
and the judging playbook list. Chiya checks them against the live form and adds any that are
missing to this list.

## The form, field by field

| Field | Answer | Owner | State |
| --- | --- | --- | --- |
| Project name | Preflight | D | Ready |
| Tagline | "A flight simulator for ML jobs, before you pay for the GPU" (58 characters) | D | Ready (`portal.md`) |
| Track | Indie Developer | Chiya | Ready |
| Theme | Build for Your World | Chiya | Ready |
| Team members | Everyone on the team, as on the registration (the rule allows four or fewer; ours is up to three) | Chiya | Chiya fills it |
| Short description | `portal.md`, "Short description" (332 characters) | D | Ready |
| Full description | `portal.md`, "Full description" | D | Ready; re-read after the video is cut so the two say the same |
| How Meta's models are used | `portal.md`, "How we used Meta's models"; the table in `meta-and-partners.md` | D | Ready |
| Live app | <https://preflight-cqn.pages.dev> | Chiya | Live; Chiya's live test before submitting |
| Repository | <https://github.com/uchuuronin/preflight> | Chiya | **Public before submitting** (private today) |
| Demo video, 3 minutes | The YouTube (or portal) link to the final cut | Chiya | Recorded Wed; `docs/demo/video-plan.md` |
| A shared flight record | <https://preflight-cqn.pages.dev/s/ICDYrfHQ2nBNbv3mlRHHMg> (the LoRA fine-tune on an L4; its card is pre-rendered) | D | Ready; opens in Chiya's live test |
| The corpus | <https://huggingface.co/datasets/chaoschia/preflight-corpus> | Chiya, B | **Make it public** (0.4.0 uploaded by Chiya, Nov 3) |
| Licence | Apache License 2.0 (`LICENSE`); dependencies in `docs/licences.md`; CI fails on GPL, AGPL and SSPL; CI also checks `workers/` and the root's node packages (C's D53) | D | Ready |
| Built during the hackathon | "Everything in this repository was built during the Meta Global AI Developer Hackathon (Oct 26 to Nov 4, 2026). The first commit is `11a0ce9`, Oct 26, 22:25 ET; nothing was carried over from earlier work." (the README, under its opening paragraph) | D | Ready |
| Best Build on Meta Open Weight Models | Flag it. `portal.md`, "Best Build on Meta Open Weight Models": Muse Spark Open on Baseten (triage, the number check, the corpus labels) and Muse Glimmer on our own GPU (private triage), both on in production | Chiya | Ready; Baseten verifies the calls are real, and every call is in the records' provenance |
| Best Agentic Use-Case | Flag it. `portal.md`, "Best Agentic Use-Case": the guide (Muse Spark driving the cockpit with tool calls) and the fix loop (Muse Code fixes, Preflight's re-probe picks) | Chiya | Ready |
| Most Viral Project | The X post's link, submitted on the platform; engagement counts through Nov 8 | Chiya | Post Wed after the video; `x-post.md` and the 45-second clip |
| Community Vote | Nothing to fill; the gallery entry is what people vote on (announced Nov 16) | Chiya | Gallery blurb in `gallery.md` |
| Challenges | `portal.md`, "Challenges" | D | Ready |
| What's next | `portal.md`, "What's next" | D | Ready |
| Screenshots or thumbnail, if the form asks | The hero card (`workers/static/cards/ICDYrfHQ2nBNbv3mlRHHMg.png`, 1200 x 630) or a cockpit frame from the video | Chiya | Pick one |

## Links, all in one place

| Link | Where it's used | Checked |
| --- | --- | --- |
| <https://preflight-cqn.pages.dev> | Form, README, X post | Chiya's live test (D's sandbox can't reach it) |
| <https://preflight-cqn.pages.dev/s/ICDYrfHQ2nBNbv3mlRHHMg> | Form, README badge, X post, video | The share route works on a production build under `wrangler pages dev` (D, Nov 3); the live record in Chiya's live test |
| <https://preflight-cqn.pages.dev/badge/ICDYrfHQ2nBNbv3mlRHHMg> | README badge | As above |
| <https://github.com/uchuuronin/preflight> | Form, X post | Once public |
| <https://huggingface.co/datasets/chaoschia/preflight-corpus> | Form, README, datasheet | 0.4.0 uploaded (Chiya, Nov 3); once public |
| The video link | Form, X post | Wed |
| Every relative link in `README.md` (final on `main`, `5bd09ed`) | README | All 14 file links and 10 in-page anchors resolve (D, Nov 3, on `main` `5bd09ed5`); check again if the README changes |

## Before pressing submit

- [x] `README.md` final on `main` (Chiya's rewrite, `5bd09ed`, Nov 3); `docs/readme-final.md` is retired,
      not swapped in.
- [ ] Repository public; no secrets in its history (CI's gitleaks job passes on `main`).
- [x] Corpus 0.4.0 on Hugging Face (Chiya, Nov 3).
- [ ] Corpus public.
- [ ] Live app passes Chiya's live test: a sample, a check-in of a public repo, the guide, the
      fix loop, a share link.
- [ ] Video uploaded, public or unlisted as the form allows, under 3 minutes, every Meta model
      labelled on screen, only our own work shown.
- [ ] Every team member listed.
- [ ] Both bonuses flagged: open weights, agentic use-case.
- [ ] X post up, its link in the form.
- [ ] `portal.md`'s answers re-read against the live product the morning of Nov 4: anything
      switched off since gets taken out of the copy.

## Must-pass rules (from the judging playbook)

| Rule | How we meet it |
| --- | --- |
| Fits a theme and genuinely uses Meta's models | Build for Your World; Muse Spark, Muse Code, Muse Spark Open and Muse Glimmer each do a job on the live site |
| New work only, and said so | The README's second paragraph and the video's opening line |
| The demo shows only our work | The video plan labels every Meta model and partner tool on screen and shows only our screens |
| Public repo, video and live product reachable | The checklist above |
| Team of four or fewer, everyone listed | Ours is up to three; Chiya lists everyone |
| A 3-minute video | `docs/demo/video-plan.md` |
| Rights and licences | Apache 2.0; `docs/licences.md`; the README's licence section; CI's copyleft check |
| Not an anti-project | Preflight is a tool for people training and evaluating models: none of the excluded kinds |

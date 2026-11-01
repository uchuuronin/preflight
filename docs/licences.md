# Licences of everything bundled

By D, Sat Oct 31, 2026, at `main` `006bc57`; re-checked Sun Nov 1 at `main` `a52753f` after round four (next section). Every package Preflight ships or installs, with its licence, from:

- `uv.lock` (the `preflight` package) and `api/uv.lock` (the API), split into runtime, optional extras and dev tools by walking each lockfile's dependency graph;
- `web/package-lock.json` (the web app; `dev` entries are build and test tools, the rest ship in the browser bundle) and the root `package-lock.json` (schema codegen, dev only);
- the Modal images' resolved pins: the probe image (`PROBE_PACKAGES`), A's serving image (`SERVING_PACKAGES`), the corpus image and the API image, each resolved with `uv pip compile` for Python 3.11 on Linux on Oct 31. The images were built earlier, so their indirect dependencies can be older than listed here; the direct pins are the same.

Licences come from each package's own metadata: PyPI's JSON API (`license_expression`, else `license`, else the licence classifiers) and each npm lock entry's `license`. Where the metadata is empty or vague the row says so and names the source used instead. CI already fails on GPL, AGPL and SSPL in the Python environments (`.github/scripts/licences.py`) and in `web/node_modules` (`web/scripts/licences.mjs`); this table adds the Modal images, which CI doesn't scan, and puts everything in one place. It is a snapshot of Oct 31: the resolution and sorting were done by D outside the repo, so rerun the method above to refresh it.

## Result

No GPL, AGPL or SSPL anywhere. Everything is permissive except the rows under "Flagged" below.

| Where | Packages | Permissive | Flagged |
| --- | --- | --- | --- |
| Web app, shipped to the browser | 26 | 26 | 0 |
| Python package and API, runtime | 18 | 17 | 1 |
| Modal images (installed at deploy, not in the repo) | 146 | 120 | 26 |
| Optional extras (pip install preflight[probe] / [modal]) | 32 | 17 | 15 |
| Build and test tools only (not shipped) | 155 | 143 | 12 |
| Modal torchtitan image (Nov 1; an ops check only, no visitor job uses it) | 90 | 75 | 15 |

## Re-check after round four (Sun Nov 1, `main` `a52753f`)

What changed since `006bc57`, and nothing else did: `uv.lock`, `api/uv.lock`, both
`package-lock.json` files and both `pyproject.toml` files are unchanged, so there are no new engine,
API or web dependencies. The probe, serving (vLLM 0.6.4.post1) and corpus images' pins are
unchanged; the fix loop's runner image adds only pydantic, numpy and httpx, all already listed.

- **New Modal image: torchtitan** (`TITAN_PACKAGES` in `python/preflight/runner/modal_app.py`,
  for A's torchtitan adapter check; no visitor job uses it). Its 14 pins resolve to 90 packages
  (`uv pip compile` for Python 3.11 on Linux, Nov 1), 38 of them not in the Oct 31 table; every
  one is listed under "Every package" below. 75 are permissive. The other 15 are flagged in the
  section after this: 12 NVIDIA CUDA libraries pulled in by torch 2.7.1, and three MPL-2.0
  packages (shtab, which is new, and certifi and tqdm, already flagged for the other images).
- **New files bundled in the repository: D2's share cards (`workers/functions/_card/`).** The
  card renderer, resvg-wasm 2.6.2 (`@resvg/resvg-wasm`, source github.com/yisibl/resvg-js), is
  vendored unmodified under MPL-2.0 (its `LICENSE` beside it); MPL-2.0 is file-level copyleft, so
  those files stay MPL-2.0 and unmodified and nothing else changes licence. The fonts, Instrument
  Sans, Instrument Serif and JetBrains Mono (from `@expo-google-fonts`), are vendored unchanged
  under the SIL Open Font License 1.1, each family with its `OFL-*.txt`. `workers/`' other
  dependencies are dev tools only (TypeScript, Vitest, Wrangler, `@types/node`), pinned in
  `workers/package.json`; nothing else from it is deployed. Scanned Nov 1: no GPL, AGPL or SSPL
  among them; Wrangler's image library brings libvips binaries under LGPL-3.0-or-later
  (`@img/sharp-libvips-*`, `@img/sharp-wasm32`), used only by the local dev server.
- **New files bundled in the repository: `zoo/torchtitan_debug/`.** The debug config, the test
  tokenizer and the `c4_test` slice are torchtitan's own, copied from its v0.1.0 tag, under its
  BSD-3-Clause licence (kept beside them as `LICENSE.txt`, as BSD-3 asks). The `c4_test` slice is
  text from C4, which AllenAI releases under ODC-BY 1.0, with Common Crawl's terms of use applying
  to its content (the allenai/c4 dataset card); ODC-BY asks for attribution, given here and in the
  README's licence section.

## Flagged, and why each is fine to use as we do

**NVIDIA CUDA libraries** (proprietary): pulled in by PyTorch's CUDA wheels and installed by pip into our Modal images, or into a user's own environment by `pip install "preflight[probe] @ git+https://github.com/uchuuronin/preflight"`; they are used under NVIDIA's own licence terms and never copied into this repository. Their metadata says "NVIDIA Proprietary Software" or `LicenseRef-NVIDIA-Proprietary`, except cuda-toolkit, nvidia-cuda-runtime, nvidia-cudnn-cu13, nvidia-nccl-cu13 and nvidia-nvshmem-cu13, which state no licence in their PyPI metadata.

| Where | Packages |
| --- | --- |
| python package (extra:probe) | cuda-toolkit 13.0.3.0, nvidia-cublas 13.1.1.3, nvidia-cuda-cupti 13.0.85, nvidia-cuda-nvrtc 13.0.88, nvidia-cuda-runtime 13.0.96, nvidia-cudnn-cu13 9.24.0.43, nvidia-cufft 12.0.0.61, nvidia-cufile 1.15.1.6, nvidia-curand 10.4.0.35, nvidia-cusolver 12.0.4.66, nvidia-cusparse 12.6.3.3, nvidia-cusparselt-cu13 0.8.1, nvidia-nccl-cu13 2.30.7, nvidia-nvjitlink 13.4.92, nvidia-nvshmem-cu13 3.4.5 |
| torchtitan image | nvidia-cublas-cu12 12.6.4.1, nvidia-cuda-cupti-cu12 12.6.80, nvidia-cuda-nvrtc-cu12 12.6.77, nvidia-cuda-runtime-cu12 12.6.77, nvidia-cudnn-cu12 9.5.1.17, nvidia-cufft-cu12 11.3.0.4, nvidia-cufile-cu12 1.11.1.6, nvidia-curand-cu12 10.3.7.77, nvidia-cusolver-cu12 11.7.1.2, nvidia-cusparse-cu12 12.5.4.2, nvidia-cusparselt-cu12 0.6.3, nvidia-nvjitlink-cu12 12.6.85 |
| probe image; serving image | nvidia-cublas-cu12 12.4.5.8, nvidia-cuda-cupti-cu12 12.4.127, nvidia-cuda-nvrtc-cu12 12.4.127, nvidia-cuda-runtime-cu12 12.4.127, nvidia-cudnn-cu12 9.1.0.70, nvidia-cufft-cu12 11.2.1.3, nvidia-curand-cu12 10.3.5.147, nvidia-cusolver-cu12 11.6.1.9, nvidia-cusparse-cu12 12.3.1.170, nvidia-nccl-cu12 2.21.5, nvidia-nvjitlink-cu12 12.4.127, nvidia-nvtx-cu12 12.4.127 |
| corpus image | nvidia-cublas-cu12 12.8.4.1, nvidia-cuda-cupti-cu12 12.8.90, nvidia-cuda-nvrtc-cu12 12.8.93, nvidia-cuda-runtime-cu12 12.8.90, nvidia-cudnn-cu12 9.10.2.21, nvidia-cufft-cu12 11.3.3.83, nvidia-cufile-cu12 1.13.1.3, nvidia-curand-cu12 10.3.9.90, nvidia-cusolver-cu12 11.7.3.90, nvidia-cusparse-cu12 12.5.8.93, nvidia-cusparselt-cu12 0.7.1, nvidia-nvjitlink-cu12 12.8.93 |

**Weak copyleft** (MPL-2.0 is file-level copyleft; LGPL-2.1 is library copyleft with notice and source duties once distributed). We distribute none of them: pip installs them into our own images or the user's environment, and lightningcss is a build tool whose output isn't covered. CI's policy allows both.

| Package | Version | Licence | Where |
| --- | --- | --- | --- |
| certifi | 2026.10.23 | MPL-2.0 | API (runtime); API image; probe image; python package (runtime); serving image; torchtitan image |
| lightningcss | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-android-arm64 | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-darwin-arm64 | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-darwin-x64 | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-freebsd-x64 | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-linux-arm-gnueabihf | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-linux-arm64-gnu | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-linux-arm64-musl | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-linux-x64-gnu | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-linux-x64-musl | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-win32-arm64-msvc | 1.33.0 | MPL-2.0 | web (dev) |
| lightningcss-win32-x64-msvc | 1.33.0 | MPL-2.0 | web (dev) |
| pycountry | 26.2.16 | LGPL-2.1-only | serving image |
| shtab | 1.13.0 | MPL-2.0 | torchtitan image |
| tqdm | 4.70.1 | MPL-2.0 AND MIT | corpus image; probe image; serving image; torchtitan image |

Not from a package index: the fix loop's image installs **Muse Code** with Meta's own installer (`curl -fsSL https://dev.meta.ai/install.sh | bash`, `api/fix_app.py`); it is Meta's command-line tool, used under Meta's terms, and nothing of it is in this repository.

Models and data the bug-zoo scripts download at run time are listed with their licences in the README (`docs/readme-final.md`, Licences); none is bundled, apart from the torchtitan test files above. Three are share-alike: `roneneldan/TinyStories` (CDLA-Sharing 1.0, `zoo/pretrain_nanogpt`, which can also run on random tokens), `databricks/databricks-dolly-15k` (CC BY-SA 3.0, `zoo/serving_vllm`) and `sentence-transformers/natural-questions` (CC BY-SA 3.0, `zoo/retrieval_st`).

## Every package

### Web app, shipped to the browser (26)

| Package | Version | Licence |
| --- | --- | --- |
| @types/d3-color | 3.1.3 | MIT |
| @types/d3-drag | 3.0.7 | MIT |
| @types/d3-interpolate | 3.0.4 | MIT |
| @types/d3-selection | 3.0.12 | MIT |
| @types/d3-transition | 3.0.9 | MIT |
| @types/d3-zoom | 3.0.9 | MIT |
| @types/react | 19.3.0 | MIT |
| @types/react-dom | 19.3.0 | MIT |
| @xyflow/react | 12.12.0 | MIT |
| @xyflow/system | 0.0.83 | MIT |
| classcat | 5.0.5 | MIT |
| csstype | 3.2.3 | MIT |
| d3-color | 3.1.0 | ISC |
| d3-dispatch | 3.0.1 | ISC |
| d3-drag | 3.0.0 | ISC |
| d3-ease | 3.0.1 | BSD-3-Clause |
| d3-interpolate | 3.0.1 | ISC |
| d3-selection | 3.0.0 | ISC |
| d3-timer | 3.0.1 | ISC |
| d3-transition | 3.0.1 | ISC |
| d3-zoom | 3.0.0 | ISC |
| react | 19.3.1 | MIT |
| react-dom | 19.3.1 | MIT |
| scheduler | 0.28.0 | MIT |
| use-sync-external-store | 1.7.0 | MIT |
| zustand | 4.5.7 | MIT |

### Python package and API, runtime (18)

| Package | Version | Licence |
| --- | --- | --- |
| annotated-doc | 0.0.5 | MIT |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.15.1 | MIT |
| certifi | 2026.10.23 | MPL-2.0 |
| fastapi | 0.162.0 | MIT |
| h11 | 0.16.0 | MIT |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| idna | 3.20 | BSD-3-Clause |
| numpy | 2.4.6 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| numpy | 2.5.4 | BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0 |
| opentelemetry-api | 1.45.1 | Apache-2.0 |
| preflight | 0.1.0 | Apache-2.0 (this repository, a local dependency of the API; the PyPI package named "preflight" is unrelated) |
| pydantic | 2.14.1 | MIT |
| pydantic-core | 2.50.1 | MIT |
| starlette | 1.10.0 | BSD-3-Clause |
| typing-extensions | 4.16.0 | PSF-2.0 |
| typing-inspection | 0.4.4 | MIT |

### Modal images (installed at deploy, not in the repo) (146)

| Package | Version | Licence |
| --- | --- | --- |
| accelerate | 1.1.1 | Apache |
| accelerate | 1.15.0 | Apache |
| aiohappyeyeballs | 2.7.1 | PSF-2.0 |
| aiohttp | 3.14.5 | Apache-2.0 AND MIT |
| aiosignal | 1.4.0 | Apache 2.0 |
| attrs | 26.1.0 | MIT |
| bitsandbytes | 0.50.2 | MIT |
| charset-normalizer | 3.5.2 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| cloudpickle | 3.1.2 | BSD-3-Clause |
| compressed-tensors | 0.8.0 | Apache 2.0 |
| datasets | 3.1.0 | Apache 2.0 |
| dill | 0.3.8 | BSD-3-Clause |
| diskcache | 5.6.3 | Apache 2.0 |
| einops | 0.8.2 | MIT |
| filelock | 4.2.5 | MIT |
| filelock | 4.2.6 | MIT |
| frozenlist | 1.8.0 | Apache-2.0 |
| fsspec | 2024.9.0 | BSD License |
| fsspec | 2026.9.0 | BSD-3-Clause |
| gguf | 0.10.0 | MIT License |
| hf-xet | 1.8.0 | Apache-2.0 |
| httpcore2 | 2.16.0 | BSD-3-Clause |
| httptools | 0.9.0 | MIT |
| httpx2 | 2.16.0 | BSD-3-Clause |
| huggingface-hub | 0.36.2 | Apache |
| huggingface-hub | 2.4.0 | Apache-2.0 |
| importlib-metadata | 10.1.0 | Apache-2.0 |
| interegular | 0.3.3 | MIT |
| jinja2 | 3.1.6 | BSD License |
| jiter | 0.18.0 | MIT |
| jsonschema | 4.26.0 | MIT |
| jsonschema-specifications | 2025.9.1 | MIT |
| lark | 1.3.1 | MIT |
| llvmlite | 0.50.0 | BSD-2-Clause AND Apache-2.0 WITH LLVM-exception |
| lm-format-enforcer | 0.10.12 | MIT |
| markdown-it-py | 4.2.0 | MIT License |
| markupsafe | 3.0.4 | BSD-3-Clause |
| mdurl | 0.1.2 | MIT License |
| mistral-common | 1.13.0 | Apache-2.0 |
| mpmath | 1.3.0 | BSD |
| msgpack | 1.2.4 | Apache-2.0 |
| msgspec | 0.22.0 | BSD-3-Clause |
| multidict | 7.2.1 | Apache-2.0 |
| multiprocess | 0.70.16 | BSD-3-Clause |
| nest-asyncio | 1.6.0 | BSD |
| networkx | 3.6.1 | BSD-3-Clause |
| numba | 0.68.1 | BSD |
| numpy | 1.26.4 | BSD License |
| numpy | 2.1.3 | BSD License |
| nvidia-cublas-cu12 | 12.4.5.8 | NVIDIA Proprietary Software |
| nvidia-cublas-cu12 | 12.8.4.1 | NVIDIA Proprietary Software |
| nvidia-cuda-cupti-cu12 | 12.4.127 | NVIDIA Proprietary Software |
| nvidia-cuda-cupti-cu12 | 12.8.90 | NVIDIA Proprietary Software |
| nvidia-cuda-nvrtc-cu12 | 12.4.127 | NVIDIA Proprietary Software |
| nvidia-cuda-nvrtc-cu12 | 12.8.93 | NVIDIA Proprietary Software |
| nvidia-cuda-runtime-cu12 | 12.4.127 | NVIDIA Proprietary Software |
| nvidia-cuda-runtime-cu12 | 12.8.90 | NVIDIA Proprietary Software |
| nvidia-cudnn-cu12 | 9.1.0.70 | NVIDIA Proprietary Software |
| nvidia-cudnn-cu12 | 9.10.2.21 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cufft-cu12 | 11.2.1.3 | NVIDIA Proprietary Software |
| nvidia-cufft-cu12 | 11.3.3.83 | NVIDIA Proprietary Software |
| nvidia-cufile-cu12 | 1.13.1.3 | NVIDIA Proprietary Software |
| nvidia-curand-cu12 | 10.3.5.147 | NVIDIA Proprietary Software |
| nvidia-curand-cu12 | 10.3.9.90 | NVIDIA Proprietary Software |
| nvidia-cusolver-cu12 | 11.6.1.9 | NVIDIA Proprietary Software |
| nvidia-cusolver-cu12 | 11.7.3.90 | NVIDIA Proprietary Software |
| nvidia-cusparse-cu12 | 12.3.1.170 | NVIDIA Proprietary Software |
| nvidia-cusparse-cu12 | 12.5.8.93 | NVIDIA Proprietary Software |
| nvidia-cusparselt-cu12 | 0.7.1 | NVIDIA Proprietary Software |
| nvidia-ml-py | 12.560.30 | BSD |
| nvidia-nccl-cu12 | 2.21.5 | NVIDIA Proprietary Software |
| nvidia-nccl-cu12 | 2.27.3 | BSD-3-Clause |
| nvidia-nvjitlink-cu12 | 12.4.127 | NVIDIA Proprietary Software |
| nvidia-nvjitlink-cu12 | 12.8.93 | NVIDIA Proprietary Software |
| nvidia-nvtx-cu12 | 12.4.127 | NVIDIA Proprietary Software |
| nvidia-nvtx-cu12 | 12.8.90 | Apache 2.0 |
| openai | 3.42.0 | Apache-2.0 |
| opencv-python-headless | 4.11.0.86 | Apache 2.0 |
| opentelemetry-api | 1.45.2 | Apache-2.0 |
| outlines | 0.0.46 | Apache-2.0 (not in its PyPI metadata; its licence page, dottxt-ai.github.io/outlines/licence) |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.0.6 | BSD License |
| pandas | 3.1.0 | BSD-3-Clause |
| partial-json-parser | 0.2.1.1.post7 | MIT |
| peft | 0.13.2 | Apache |
| peft | 0.21.2 | Apache |
| pillow | 13.0.0 | MIT-CMU |
| prometheus-client | 0.26.0 | Apache-2.0 AND BSD-2-Clause |
| prometheus-fastapi-instrumentator | 8.1.0 | ISC |
| propcache | 0.5.4 | Apache-2.0 |
| protobuf | 7.36.2 | 3-Clause BSD License |
| psutil | 6.1.0 | BSD-3-Clause |
| psutil | 7.2.2 | BSD-3-Clause |
| py-cpuinfo | 9.0.0 | MIT |
| pyairports | 0.0.1 | MIT License |
| pyarrow | 26.0.0 | Apache-2.0 |
| pycountry | 26.2.16 | LGPL-2.1-only |
| pydantic | 2.14.2 | MIT |
| pydantic-core | 2.50.2 | MIT |
| pydantic-extra-types | 2.11.1 | MIT |
| pygments | 2.21.0 | BSD-2-Clause |
| python-dateutil | 3.0.0 | Apache-2.0 or BSD-3-Clause (its classifiers; PyPI says "Dual License") |
| python-dotenv | 1.2.4 | BSD-3-Clause |
| pyyaml | 6.0.3 | MIT |
| pyzmq | 27.2.0 | BSD-3-Clause |
| ray | 2.59.0 | Apache 2.0 |
| referencing | 0.37.0 | MIT |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache-2.0 |
| rich | 15.0.0 | MIT |
| rpds-py | 2026.9.1 | MIT |
| safetensors | 0.4.5 | Apache Software License |
| safetensors | 0.8.0 | Apache Software License |
| sentencepiece | 0.2.2 | Apache-2.0 |
| setuptools | 84.2.1 | MIT |
| shellingham | 1.5.4 | ISC License |
| six | 1.17.0 | MIT |
| sniffio | 1.3.1 | MIT OR Apache-2.0 |
| sympy | 1.13.1 | BSD |
| sympy | 1.14.0 | BSD |
| tiktoken | 0.14.0 | MIT License (licence text in metadata) |
| tokenizers | 0.20.3 | Apache Software License |
| tokenizers | 0.23.3 | Apache Software License |
| torch | 2.5.1 | BSD-3-Clause |
| torch | 2.8.0 | BSD-3-Clause |
| torchvision | 0.20.1 | BSD |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| transformers | 4.46.3 | Apache 2.0 License |
| transformers | 5.19.0 | Apache 2.0 License |
| triton | 3.1.0 | MIT License |
| triton | 3.4.0 | MIT License |
| trl | 0.12.1 | Apache 2.0 |
| truststore | 0.10.5 | MIT |
| typer | 0.27.3 | MIT |
| tzdata | 2026.6 | Apache-2.0 |
| urllib3 | 2.8.0 | MIT |
| uvicorn | 0.54.0 | BSD-3-Clause |
| uvloop | 0.23.0 | MIT License |
| vllm | 0.6.4.post1 | Apache 2.0 |
| watchfiles | 1.3.0 | MIT |
| websockets | 17.4 | BSD-3-Clause |
| xformers | 0.0.28.post3 | BSD License |
| xxhash | 4.0.2 | BSD-2-Clause |
| yarl | 1.25.1 | Apache-2.0 |
| zipp | 4.1.2 | MIT |

### Modal torchtitan image (ops check only, Nov 1) (90)

| Package | Version | Licence |
| --- | --- | --- |
| absl-py | 2.5.1 | Apache-2.0 |
| aiohappyeyeballs | 2.7.1 | PSF-2.0 |
| aiohttp | 3.14.5 | Apache-2.0 AND MIT |
| aiosignal | 1.4.0 | Apache 2.0 |
| annotated-types | 0.8.0 | MIT |
| anyio | 4.16.0 | MIT |
| attrs | 26.1.0 | MIT |
| blobfile | 3.0.0 | Public Domain |
| certifi | 2026.10.23 | MPL-2.0 |
| charset-normalizer | 3.5.2 | MIT |
| click | 8.5.0 | BSD-3-Clause |
| datasets | 3.6.0 | Apache 2.0 |
| dill | 0.3.8 | BSD-3-Clause |
| docstring-parser | 0.18.0 | MIT |
| filelock | 4.2.6 | MIT |
| frozenlist | 1.8.0 | Apache-2.0 |
| fsspec | 2025.3.0 | BSD License |
| grpcio | 1.85.1 | Apache-2.0 |
| h11 | 0.16.0 | MIT |
| hf-xet | 1.8.0 | Apache-2.0 |
| httpcore | 1.0.9 | BSD-3-Clause |
| httpcore2 | 2.16.0 | BSD-3-Clause |
| httpx | 0.28.1 | BSD-3-Clause |
| httpx2 | 2.16.0 | BSD-3-Clause |
| huggingface-hub | 2.4.0 | Apache-2.0 |
| idna | 3.20 | BSD-3-Clause |
| jinja2 | 3.1.6 | BSD License |
| lxml | 6.1.3 | BSD-3-Clause |
| markdown | 3.11.2 | BSD-3-Clause |
| markdown-it-py | 4.2.0 | MIT License |
| markupsafe | 3.0.4 | BSD-3-Clause |
| mdurl | 0.1.2 | MIT License |
| mpmath | 1.3.0 | BSD |
| multidict | 7.2.1 | Apache-2.0 |
| multiprocess | 0.70.16 | BSD-3-Clause |
| networkx | 3.6.1 | BSD-3-Clause |
| numpy | 2.2.6 | BSD License |
| nvidia-cublas-cu12 | 12.6.4.1 | NVIDIA Proprietary Software |
| nvidia-cuda-cupti-cu12 | 12.6.80 | NVIDIA Proprietary Software |
| nvidia-cuda-nvrtc-cu12 | 12.6.77 | NVIDIA Proprietary Software |
| nvidia-cuda-runtime-cu12 | 12.6.77 | NVIDIA Proprietary Software |
| nvidia-cudnn-cu12 | 9.5.1.17 | NVIDIA Proprietary Software |
| nvidia-cufft-cu12 | 11.3.0.4 | NVIDIA Proprietary Software |
| nvidia-cufile-cu12 | 1.11.1.6 | NVIDIA Proprietary Software |
| nvidia-curand-cu12 | 10.3.7.77 | NVIDIA Proprietary Software |
| nvidia-cusolver-cu12 | 11.7.1.2 | NVIDIA Proprietary Software |
| nvidia-cusparse-cu12 | 12.5.4.2 | NVIDIA Proprietary Software |
| nvidia-cusparselt-cu12 | 0.6.3 | NVIDIA Proprietary Software |
| nvidia-ml-py | 12.560.30 | BSD |
| nvidia-nccl-cu12 | 2.26.2 | BSD-3-Clause |
| nvidia-nvjitlink-cu12 | 12.6.85 | NVIDIA Proprietary Software |
| nvidia-nvtx-cu12 | 12.6.77 | Apache 2.0 |
| packaging | 26.3 | Apache-2.0 OR BSD-2-Clause |
| pandas | 3.1.0 | BSD-3-Clause |
| propcache | 0.5.4 | Apache-2.0 |
| protobuf | 7.36.2 | 3-Clause BSD License |
| psutil | 6.1.0 | BSD-3-Clause |
| pyarrow | 26.0.0 | Apache-2.0 |
| pycryptodomex | 3.24.1 | BSD, Public Domain |
| pydantic | 2.14.2 | MIT |
| pydantic-core | 2.50.2 | MIT |
| pygments | 2.21.0 | BSD-2-Clause |
| python-dateutil | 4.0.0 | BSD-3-Clause |
| pyyaml | 6.0.3 | MIT |
| regex | 2026.9.29 | Apache-2.0 AND CNRI-Python |
| requests | 2.34.2 | Apache-2.0 |
| rich | 15.0.0 | MIT |
| setuptools | 84.2.1 | MIT |
| shtab | 1.13.0 | MPL-2.0 |
| six | 1.17.0 | MIT |
| sympy | 1.14.0 | BSD |
| tensorboard | 2.19.0 | Apache 2.0 |
| tensorboard-data-server | 0.7.2 | Apache 2.0 |
| tiktoken | 0.9.0 | MIT License (licence text in metadata) |
| tomli | 2.2.1 | MIT License |
| torch | 2.7.1 | BSD-3-Clause |
| torchdata | 0.11.0 | BSD |
| torchtitan | 0.1.0 | BSD 3-Clause License (licence text in metadata) |
| tqdm | 4.70.1 | MPL-2.0 AND MIT |
| triton | 3.3.1 | MIT License |
| truststore | 0.10.5 | MIT |
| typeguard | 4.6.0 | MIT |
| typing-extensions | 4.16.0 | PSF-2.0 |
| typing-inspection | 0.4.4 | MIT |
| tyro | 0.9.24 | MIT |
| tzdata | 2026.6 | Apache-2.0 |
| urllib3 | 2.8.0 | MIT |
| werkzeug | 3.1.9 | BSD-3-Clause |
| xxhash | 4.0.2 | BSD-2-Clause |
| yarl | 1.25.1 | Apache-2.0 |

### Optional extras (pip install preflight[probe] / [modal]) (32)

| Package | Version | Licence |
| --- | --- | --- |
| cbor2 | 6.1.5 | MIT |
| cuda-bindings | 13.4.4 | Apache-2.0 |
| cuda-pathfinder | 1.8.4 | Apache-2.0 |
| cuda-toolkit | 13.0.3.0 | none in its PyPI metadata; NVIDIA's cu12 builds of the same libraries state NVIDIA Proprietary Software |
| filelock | 4.2.4 | MIT |
| grpclib | 0.4.9 | BSD-3-Clause |
| h2 | 4.4.1 | MIT |
| hpack | 4.2.0 | MIT |
| hyperframe | 6.1.0 | MIT License |
| modal | 1.6.2 | Apache-2.0 |
| networkx | 3.7 | BSD-3-Clause |
| nvidia-cublas | 13.1.1.3 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-cupti | 13.0.85 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-nvrtc | 13.0.88 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cuda-runtime | 13.0.96 | none in its PyPI metadata; NVIDIA's cu12 builds of the same libraries state NVIDIA Proprietary Software |
| nvidia-cudnn-cu13 | 9.24.0.43 | none in its PyPI metadata; NVIDIA's cu12 builds of the same libraries state NVIDIA Proprietary Software |
| nvidia-cufft | 12.0.0.61 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cufile | 1.15.1.6 | LicenseRef-NVIDIA-Proprietary |
| nvidia-curand | 10.4.0.35 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cusolver | 12.0.4.66 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cusparse | 12.6.3.3 | LicenseRef-NVIDIA-Proprietary |
| nvidia-cusparselt-cu13 | 0.8.1 | NVIDIA Proprietary Software |
| nvidia-nccl-cu13 | 2.30.7 | none in its PyPI metadata; NVIDIA's cu12 builds of the same libraries state NVIDIA Proprietary Software |
| nvidia-nvjitlink | 13.4.92 | LicenseRef-NVIDIA-Proprietary |
| nvidia-nvshmem-cu13 | 3.4.5 | none in its PyPI metadata; NVIDIA's cu12 builds of the same libraries state NVIDIA Proprietary Software |
| nvidia-nvtx | 13.0.85 | Apache 2.0 |
| synchronicity | 0.12.7 | Apache Software License |
| toml | 0.10.2 | MIT |
| torch | 2.14.1 | Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND BSL-1.0 AND MIT |
| triton | 3.8.0 | MIT |
| types-certifi | 2021.10.8.3 | Apache-2.0 license |
| types-toml | 0.10.8.20260518 | Apache-2.0 |

### Build and test tools only (not shipped) (155)

| Package | Version | Licence |
| --- | --- | --- |
| @apidevtools/json-schema-ref-parser | 11.9.3 | MIT |
| @asamuzakjp/css-color | 7.1.9 | MIT |
| @asamuzakjp/dom-selector | 10.0.5 | MIT |
| @babel/code-frame | 7.29.7 | MIT |
| @babel/helper-validator-identifier | 7.29.7 | MIT |
| @babel/runtime | 7.29.10 | MIT |
| @bramus/specificity | 2.4.3 | MIT |
| @csstools/color-helpers | 6.1.2 | MIT-0 |
| @csstools/css-calc | 3.4.3 | MIT |
| @csstools/css-color-parser | 4.2.6 | MIT |
| @csstools/css-parser-algorithms | 4.0.2 | MIT |
| @csstools/css-syntax-patches-for-csstree | 1.1.15 | MIT-0 |
| @csstools/css-tokenizer | 4.0.2 | MIT |
| @exodus/bytes | 1.16.0 | MIT |
| @jridgewell/resolve-uri | 3.1.2 | MIT |
| @jridgewell/sourcemap-codec | 1.6.0 | MIT |
| @jridgewell/trace-mapping | 0.3.31 | MIT |
| @jsdevtools/ono | 7.1.3 | MIT |
| @oxc-project/types | 0.155.0 | MIT |
| @rolldown/binding-android-arm-eabi | 1.2.15 | MIT |
| @rolldown/binding-android-arm64 | 1.2.15 | MIT |
| @rolldown/binding-darwin-arm64 | 1.2.15 | MIT |
| @rolldown/binding-darwin-x64 | 1.2.15 | MIT |
| @rolldown/binding-freebsd-x64 | 1.2.15 | MIT |
| @rolldown/binding-linux-arm-gnueabihf | 1.2.15 | MIT |
| @rolldown/binding-linux-arm64-gnu | 1.2.15 | MIT |
| @rolldown/binding-linux-arm64-musl | 1.2.15 | MIT |
| @rolldown/binding-linux-ppc64-gnu | 1.2.15 | MIT |
| @rolldown/binding-linux-s390x-gnu | 1.2.15 | MIT |
| @rolldown/binding-linux-x64-gnu | 1.2.15 | MIT |
| @rolldown/binding-linux-x64-musl | 1.2.15 | MIT |
| @rolldown/binding-openharmony-arm64 | 1.2.15 | MIT |
| @rolldown/binding-win32-arm64-msvc | 1.2.15 | MIT |
| @rolldown/binding-win32-x64-msvc | 1.2.15 | MIT |
| @rolldown/pluginutils | 1.0.1 | MIT |
| @testing-library/dom | 10.4.2 | MIT |
| @testing-library/react | 16.3.3 | MIT |
| @types/aria-query | 5.0.4 | MIT |
| @types/chai | 5.2.3 | MIT |
| @types/deep-eql | 4.0.2 | MIT |
| @types/estree | 1.0.9 | MIT |
| @types/json-schema | 7.0.15 | MIT |
| @types/lodash | 4.17.27 | MIT |
| @typescript/typescript-aix-ppc64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-darwin-arm64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-darwin-x64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-freebsd-arm64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-freebsd-x64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-linux-arm | 7.0.2 | Apache-2.0 |
| @typescript/typescript-linux-arm64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-linux-loong64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-linux-mips64el | 7.0.2 | Apache-2.0 |
| @typescript/typescript-linux-ppc64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-linux-riscv64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-linux-s390x | 7.0.2 | Apache-2.0 |
| @typescript/typescript-linux-x64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-netbsd-arm64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-netbsd-x64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-openbsd-arm64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-openbsd-x64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-sunos-x64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-win32-arm64 | 7.0.2 | Apache-2.0 |
| @typescript/typescript-win32-x64 | 7.0.2 | Apache-2.0 |
| @vitejs/plugin-react | 6.1.2 | MIT |
| @vitest/mocker | 5.0.5 | MIT |
| @vitest/spy | 5.0.5 | MIT |
| ansi-regex | 5.0.1 | MIT |
| ansi-styles | 5.2.0 | MIT |
| argparse | 2.0.1 | Python-2.0 (the PSF licence) |
| aria-query | 5.3.0 | Apache-2.0 |
| assertion-error | 2.0.1 | MIT |
| bidi-js | 1.1.0 | MIT |
| chai | 6.3.0 | MIT |
| colorama | 0.4.6 | BSD License |
| css-tree | 3.2.1 | MIT |
| data-urls | 8.0.0 | MIT |
| decimal.js | 10.6.0 | MIT |
| dequal | 2.0.3 | MIT |
| detect-libc | 2.1.2 | Apache-2.0 |
| dom-accessibility-api | 0.5.16 | MIT |
| entities | 8.1.0 | BSD-2-Clause |
| es-module-lexer | 2.3.2 | MIT |
| estree-walker | 3.0.3 | MIT |
| expect-type | 1.4.0 | Apache-2.0 |
| fdir | 6.5.0 | MIT |
| fdir | 6.5.0 | MIT |
| fsevents | 2.3.3 | MIT |
| html-encoding-sniffer | 7.0.0 | MIT |
| iniconfig | 2.3.1 | MIT |
| is-extglob | 2.1.1 | MIT |
| is-glob | 4.0.3 | MIT |
| is-potential-custom-element-name | 1.0.1 | MIT |
| js-tokens | 4.0.0 | MIT |
| js-yaml | 4.3.2 | MIT |
| jsdom | 30.1.3 | MIT |
| json-schema-to-typescript | 15.0.4 | MIT |
| lightningcss | 1.33.0 | MPL-2.0 |
| lightningcss-android-arm64 | 1.33.0 | MPL-2.0 |
| lightningcss-darwin-arm64 | 1.33.0 | MPL-2.0 |
| lightningcss-darwin-x64 | 1.33.0 | MPL-2.0 |
| lightningcss-freebsd-x64 | 1.33.0 | MPL-2.0 |
| lightningcss-linux-arm-gnueabihf | 1.33.0 | MPL-2.0 |
| lightningcss-linux-arm64-gnu | 1.33.0 | MPL-2.0 |
| lightningcss-linux-arm64-musl | 1.33.0 | MPL-2.0 |
| lightningcss-linux-x64-gnu | 1.33.0 | MPL-2.0 |
| lightningcss-linux-x64-musl | 1.33.0 | MPL-2.0 |
| lightningcss-win32-arm64-msvc | 1.33.0 | MPL-2.0 |
| lightningcss-win32-x64-msvc | 1.33.0 | MPL-2.0 |
| lodash | 4.18.1 | MIT |
| lru-cache | 11.5.3 | BlueOak-1.0.0 |
| lz-string | 1.5.0 | MIT |
| magic-string | 1.4.4 | MIT |
| mdn-data | 2.27.1 | CC0-1.0 |
| minimist | 1.2.8 | MIT |
| nanoid | 3.3.20 | MIT |
| obug | 3.0.0 | MIT |
| parse5 | 8.0.1 | MIT |
| picocolors | 1.1.1 | ISC |
| picomatch | 4.0.7 | MIT |
| picomatch | 4.0.7 | MIT |
| playwright | 1.63.0 | Apache-2.0 |
| playwright-core | 1.63.0 | Apache-2.0 |
| pluggy | 1.7.0 | MIT |
| postcss | 8.5.32 | MIT |
| prettier | 3.9.10 | MIT |
| pretty-format | 27.5.1 | MIT |
| punycode | 2.3.1 | MIT |
| pytest | 9.1.1 | MIT |
| pytest-timeout | 2.4.0 | MIT |
| react-is | 17.0.2 | MIT |
| require-from-string | 2.0.2 | MIT |
| rolldown | 1.2.15 | MIT |
| ruff | 0.17.3 | MIT |
| saxes | 6.0.0 | ISC |
| source-map-js | 1.2.2 | BSD-3-Clause |
| std-env | 4.3.0 | MIT |
| tinybench | 6.2.2 | MIT |
| tinyexec | 1.3.1 | MIT |
| tinyglobby | 0.2.17 | MIT |
| tinyglobby | 0.2.17 | MIT |
| tldts | 7.4.19 | MIT |
| tldts-core | 7.4.19 | MIT |
| tough-cookie | 6.0.2 | BSD-3-Clause |
| tr46 | 7.0.0 | MIT |
| typescript | 7.0.2 | Apache-2.0 |
| undici | 8.12.3 | MIT |
| vite | 8.3.5 | MIT |
| vitest | 5.0.5 | MIT |
| w3c-xmlserializer | 6.0.0 | MIT |
| webidl-conversions | 8.0.1 | BSD-2-Clause |
| whatwg-mimetype | 5.0.0 | MIT |
| whatwg-url | 17.3.0 | MIT |
| why-is-node-running | 3.2.1 | MIT |
| xml-name-validator | 5.0.0 | Apache-2.0 |
| xmlchars | 2.2.0 | MIT |

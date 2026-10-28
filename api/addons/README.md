# api/addons/ (owner: D)

Every router here is mounted by C's app automatically; D's routes never touch `api/`.

- A file `api/addons/<name>.py`, or a package `api/addons/<name>/__init__.py`, that defines a
  FastAPI `router` (an `APIRouter`) is mounted at start-up.
- Use a `/v1/addons/<name>` prefix, keep each add-on behind a flag from `flags.json`
  (`from preflight_api.flags import flag`), and raise `preflight_api.errors.ApiError` for failures
  so responses keep the shared error envelope.
- A broken add-on is logged and skipped; it never takes the API down.

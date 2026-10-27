"""upgrade(record): bring a stored record up to the current schema version.

Changes after the v0.1 freeze are additive, so most minor bumps need no step
here at all: new optional fields default themselves. A step is needed only
when a field is renamed or reshaped (which needs Chiya's sign-off). Register
it in STEPS keyed by the version it upgrades *from*. Golden records in
schema/examples/ are round-tripped in CI through this function.
"""

from collections.abc import Callable

from .record import SCHEMA_VERSION


class UnsupportedSchemaVersion(ValueError):
	pass


def _parse(v: str) -> tuple[int, int, int]:
	try:
		major, minor, patch = (int(x) for x in v.split("."))
	except Exception as exc:
		raise UnsupportedSchemaVersion(f"bad schema_version {v!r}") from exc
	return major, minor, patch


# from-version -> function returning a record at the next version
STEPS: dict[str, Callable[[dict], dict]] = {}


def upgrade(raw: dict) -> dict:
	version = raw.get("schema_version", "0.1.0")
	cur = _parse(SCHEMA_VERSION)
	got = _parse(version)
	if got[0] != cur[0] or got > cur:
		raise UnsupportedSchemaVersion(
			f"record is {version}; this build reads {cur[0]}.x up to {SCHEMA_VERSION}"
		)
	seen = set()
	while version in STEPS:
		if version in seen:
			raise RuntimeError(f"migration loop at {version}")
		seen.add(version)
		raw = STEPS[version](dict(raw))
		version = raw["schema_version"]
	if _parse(version) < cur:
		# only additive changes since: defaults fill the gaps
		raw = {**raw, "schema_version": SCHEMA_VERSION}
	return raw

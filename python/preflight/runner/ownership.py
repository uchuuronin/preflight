"""Which container may own the job store's shared state: the deployed app's, and no other.

Modal-free, so CI tests the rules; runner/modal_app.py's deployed_here() feeds it the container's
app id and a lookup of the deployed app.
"""

LOOKUP_TRIES = 3  # guess: covers a transient control-plane error; then the container restarts and tries again


def decide_ownership(
	mine_id, lookup, *, where: str, env_known: bool = True, tries: int = LOOKUP_TRIES, sleep=lambda s: None
):
	"""(True | False | None, why) from this container's app id and a lookup of the deployed one.

	True only on a match. False when the deployed app was found and is another app, or isn't
	deployed in a known environment. None when it can't tell: the lookup keeps failing, or the
	app isn't found but the environment was a guess (production looked up in the default
	environment would read "not deployed" and must not conclude that)."""
	if not mine_id:
		return None, "this container has no app id"
	last = None
	for i in range(tries):
		try:
			deployed_id = lookup()
		except Exception as exc:  # NotFound included: nothing deployed under that name yet
			last = exc
			if "not found" in str(exc).lower() or type(exc).__name__ == "NotFoundError":
				if not env_known:
					return None, f"{where} not found, and this container's environment is unknown"
				return False, f"{where} isn't deployed; this container is app {mine_id}"
			sleep(2**i)
			continue
		if deployed_id == mine_id:
			return True, f"this container is the deployed {where} ({mine_id})"
		return False, f"this container is app {mine_id}, not the deployed {where} ({deployed_id})"
	return None, f"couldn't look up {where} ({type(last).__name__}: {str(last)[:200]})"

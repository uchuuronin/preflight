"""The deploy gate's pass rule for the probe-user startup step, kept free of Modal so CI runs its
test (runner/ops_app.py imports Modal at the top; CI's Python job doesn't install it).

What each sandbox image runs as the probe uid on a CPU (Chiya, Nov 1): its user and group names
resolve, and the imports that read them load, as the probe would run them; "+compile" runs a tiny
torch.compile rather than importing anything (probes/local.py PROBE_USER_SCRIPT).
"""

PROBE_UID = 61000  # PREFLIGHT_PROBE_UID in the sandbox images (runner/modal_app.py)
PROBE_USER = "preflight-probe"

PROBE_USER_IMPORTS = {
	"probe": ("+compile",),
	"serving": ("+compile", "vllm", "datasets"),  # datasets: pyarrow against the image's numpy 1
	"titan": ("+compile", "torchtitan.train"),
	"torch27": ("+compile", "transformers", "trl"),  # probes.torch_new
}


def probe_user_ok(r: dict, image: str) -> bool:
	"""Whether one image's probe-user check passed: uid, names, every import, and the compile."""
	want = [m for m in PROBE_USER_IMPORTS[image] if m != "+compile"]
	compiled = "+compile" not in PROBE_USER_IMPORTS[image] or r.get("compiled") is not None
	return (r.get("exit") == 0 and r.get("uid") == PROBE_UID and r.get("user") == r.get("passwd") == PROBE_USER
		and r.get("group") == PROBE_USER and r.get("imported", []) == want and compiled)  # fmt: skip

"""runner.entrypoint_path (off; C, Nov 2): a script's own folder is first on its path, as `python dir/main.py` puts
it, and a missing module that is the repo's own says so (`missing_module.local`), not "add it to requirements.txt"."""

import pytest

pytest.importorskip("torch")

from preflight.probes.local import ProbeSpec, run_probe

LOOP = """import torch
model = torch.nn.Linear(4, 4)
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(20):
    loss = model(torch.randn(2, 4)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
"""


def _probe(files, entry, on):
	return run_probe(ProbeSpec(probe_id="p", files=files, entrypoint=entry, warmup=1, timed=2, timeout_s=120,
		entry_path=on)).probe  # fmt: skip


def test_a_package_beside_the_entry_point_imports_with_the_flag_and_not_without():
	files = {"ConvNets/main.py": "import image_classification.logger as log\n" + LOOP,
		"ConvNets/image_classification/__init__.py": "", "ConvNets/image_classification/logger.py": "x = 1\n"}  # fmt: skip
	off = _probe(files, "ConvNets/main.py", False)
	assert off.status == "error" and off.error_code == "missing_package"  # as before
	on = _probe(files, "ConvNets/main.py", True)
	assert on.status == "ok" and on.steps >= 2


def test_a_missing_local_module_is_named_as_the_repo_s_own_not_a_package():
	files = {"app/main.py": "import helpers\n" + LOOP, "tools/helpers/__init__.py": ""}
	p = _probe(files, "app/main.py", True)
	assert (
		p.error_code == "missing_module.local" and p.missing_module == "helpers" and p.missing_package is None
	)
	assert "the repo's own module under tools/" in p.error


def test_its_failure_step_doesnt_send_the_visitor_to_requirements_txt():
	from preflight.runner.failures import STEPS

	_step, way = STEPS["missing_module.local"]
	assert "isn't a package to add to requirements.txt" in way

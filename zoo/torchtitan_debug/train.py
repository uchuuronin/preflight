"""torchtitan's own trainer on its smallest model (Llama 3 "debugmodel", torchtitan 0.1.0), one GPU.

Preflight's adapter check: the recorder must count torchtitan's optimizer steps and split its
phases the way it does for HF Trainer, TRL and Lightning. The config, test tokenizer and c4_test
slice are torchtitan's own (BSD-3-Clause, LICENSE.txt), from its v0.1.0 tag; nothing is planted.
What torchrun would set for one process is set here, so `python train.py` runs it.
"""

import os
import runpy
import sys

for k, v in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1", "LOCAL_WORLD_SIZE": "1",
	"MASTER_ADDR": "127.0.0.1", "MASTER_PORT": "29517"}.items():  # fmt: skip
	os.environ.setdefault(k, v)
here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)  # the config's tokenizer and dataset paths are relative
sys.argv = ["torchtitan.train", "--job.config_file", os.path.join(here, "debug_model.toml")]
runpy.run_module("torchtitan.train", run_name="__main__", alter_sys=True)

"""Model-level knobs B maps (docs/requests.md): LoRA alpha and dropout, weight quantization, attention
kernel, agent context limit; plus the code knobs declared when a fine-tune doesn't set them."""

from pathlib import Path

import pytest

from preflight.harness.skeleton import skeleton_record
from preflight.harness.static import map_static
from preflight.probes.gpu_choice import choose_probe_gpu, estimate_need_gb
from preflight.schema import Source

ROOT = Path(__file__).resolve().parents[1]

QLORA = """
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", quantization_config=bnb,
    attn_implementation="flash_attention_2")
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, target_modules=["q_proj", "v_proj"]))
Trainer(model=model, args=TrainingArguments(output_dir="o", per_device_train_batch_size=4, bf16=True)).train()
"""


def sm_of(src):
	return map_static({"train.py": src}, entrypoint="train.py")


def knobs(sm):
	return {k.id: k for k in sm.knobs}


def test_qlora_script_knobs_are_read_with_their_locations():
	sm = sm_of(QLORA)
	k = knobs(sm)
	assert k["lora_alpha"].current == 32 and k["lora_dropout"].current == 0.05
	assert k["attn"].current == "flash_attention_2" and not k["attn"].needs_code
	q = k["quantization"]
	assert (
		q.current == "nf4_dq" and q.needs_code and q.location.line == 6 and "load_in_4bit" in q.location.expr
	)
	assert sm.model.quantization == "nf4_dq" and sm.model.lora.alpha == 32


@pytest.mark.parametrize(
	("flags", "label"),
	[
		("load_in_4bit=True", "fp4"),
		('load_in_4bit=True, bnb_4bit_quant_type="nf4"', "nf4"),
		("load_in_4bit=True, bnb_4bit_use_double_quant=True", "fp4_dq"),
		("load_in_8bit=True", "int8"),
	],
)
def test_bitsandbytes_flags_map_to_bs_weight_formats(flags, label):
	src = f'from transformers import AutoModelForCausalLM, BitsAndBytesConfig\nq = BitsAndBytesConfig({flags})\nm = AutoModelForCausalLM.from_pretrained("x/y-1B", quantization_config=q)\n'
	assert knobs(sm_of(src))["quantization"].current == label


def test_prequantised_model_ids_are_recognised():
	src = 'from transformers import AutoModelForCausalLM\nm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GPTQ")\n'
	assert sm_of(src).model.quantization == "gptq_4bit"


def test_quantised_lora_is_sized_as_qlora():
	sm = sm_of(QLORA)
	need = estimate_need_gb(sm)
	assert need < 15 and choose_probe_gpu(need) == "L4"  # 8B in 4-bit fits an L4


def test_skeleton_declares_qlora_and_attention_when_the_code_does_not_set_them():
	rec = skeleton_record(sm_of((ROOT / "zoo/finetune_lora_hf/train.py").read_text()), Source(kind="sample"))
	q, a = rec.knob("quantization"), rec.knob("attn")
	assert q.current == "none" and q.needs_code and q.patch_template and "nf4" in q.domain.values
	assert q.group == "memory"
	assert (
		a.current == "sdpa" and a.needs_code and "flash_attention_2" in a.domain.values and a.group == "speed"
	)


def test_skeleton_keeps_script_values_and_domains():
	rec = skeleton_record(sm_of(QLORA), Source(kind="upload"))
	q = rec.knob("quantization")
	assert q.current == "nf4_dq" and set(q.domain.values) >= {"none", "int8", "nf4", "nf4_dq"}
	assert rec.knob("lora_alpha").group == "model" and rec.knob("lora_dropout").domain.max == 0.5


def test_agent_context_limit():
	src = "from openai import OpenAI\nMAX_CONTEXT_TOKENS = 64000\nc = OpenAI()\nc.chat.completions.create(model='muse-spark-1.3', messages=[])\n"
	rec = skeleton_record(sm_of(src), Source(kind="upload"))
	k = rec.knob("context_limit")
	assert k.current == 64000 and k.unit == "tokens" and k.group == "agent"


def test_gptq_int8_ids_and_first_load_wins():
	src = 'from transformers import AutoModelForCausalLM\nm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct-GPTQ-Int8")\n'
	assert sm_of(src).model.quantization == "gptq_8bit"
	two = (
		"from transformers import AutoModelForCausalLM, BitsAndBytesConfig\n"
		'q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")\n'
		'policy = AutoModelForCausalLM.from_pretrained("x/policy-1B", quantization_config=q)\n'
		'reward = AutoModelForCausalLM.from_pretrained("x/reward-1B", load_in_8bit=True)\n'
	)
	sm = sm_of(two)
	assert sm.model.quantization == "nf4" and knobs(sm)["quantization"].current == "nf4"


def test_prequantised_lora_gets_no_quantization_knob():
	src = (
		"from transformers import AutoModelForCausalLM, Trainer\nfrom peft import LoraConfig, get_peft_model\n"
		'm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-GPTQ")\n'
		"m = get_peft_model(m, LoraConfig(r=8))\nTrainer(model=m).train()\n"
	)
	rec = skeleton_record(sm_of(src), Source(kind="upload"))
	with pytest.raises(KeyError):
		rec.knob("quantization")


def test_script_read_quantization_knob_carries_the_patch():
	rec = skeleton_record(sm_of(QLORA), Source(kind="upload"))
	q = rec.knob("quantization")
	assert q.patch_template == "model.quantize_weights" and q.needs_code

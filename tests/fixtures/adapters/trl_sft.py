"""TRL's SFTTrainer, the shape of TRL's sft.py example, on a tiny random Llama and a word-level
tokenizer built here, so it runs on CPU with no downloads."""

import torch
from datasets import Dataset
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from trl import SFTConfig, SFTTrainer

torch.manual_seed(0)
words = [f"w{i}" for i in range(120)]
tok = Tokenizer(
	models.WordLevel({w: i for i, w in enumerate(["<pad>", "<eos>", "<unk>", *words])}, unk_token="<unk>")
)
tok.pre_tokenizer = pre_tokenizers.Whitespace()
tokenizer = PreTrainedTokenizerFast(
	tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>"
)
model = LlamaForCausalLM(LlamaConfig(vocab_size=len(tokenizer), hidden_size=32, intermediate_size=64, num_hidden_layers=2,
	num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64, pad_token_id=0, eos_token_id=1))  # fmt: skip
g = torch.Generator().manual_seed(0)
data = Dataset.from_dict(
	{"text": [" ".join(words[j] for j in torch.randint(0, 120, (24,), generator=g)) for _ in range(256)]}
)
args = SFTConfig(output_dir="out/trl-sft", per_device_train_batch_size=8, max_steps=40, learning_rate=1e-3,
	logging_steps=5, save_strategy="no", report_to=[], use_cpu=True, max_length=32)  # fmt: skip
SFTTrainer(model=model, args=args, train_dataset=data, processing_class=tokenizer).train()

"""The parent's checks on the probe child's kernel table and model shape (the child runs visitor code)."""

from preflight.probes import untrusted


def row(**kw):
	return {"name": "aten::mm", "class": "gemm", "calls": 2.0, "device_s": 0.01, "flops": 1e9, "bytes": 1e6,
		"dtype": "bf16", **kw}  # fmt: skip


def table(rows, **kw):
	return {"rows": rows, "steps": 3, "backend": "cuda", "allow_tf32": False, "classified_by": None, **kw}


def test_good_table_passes_with_its_class_key():
	t, err = untrusted.kernel_table(table([row()]))
	assert err is None and t["rows"][0]["class"] == "gemm"


def test_hostile_tables_are_dropped_not_fatal():
	bad = [
		table([row(device_s=float("nan"))]),
		table([row(device_s=-1.0)]),
		table([row()] * (untrusted.MAX_ROWS + 1)),
		table([row(extra="x")]),
		table([row()], backend="tpu; rm -rf /"),
		table([{"name": 5}]),
		["not", "a", "dict"],
	]
	for raw in bad:
		t, err = untrusted.kernel_table(raw)
		assert t is None and err


def test_long_visitor_labels_are_cut_not_rejected():
	t, err = untrusted.kernel_table(table([row(name="my_block" * 10_000)]))
	assert err is None and len(t["rows"][0]["name"]) == untrusted.MAX_NAME


def test_profiler_error_becomes_a_reason():
	t, err = untrusted.kernel_table({"error": "RuntimeError: secret internal path /root/x"})
	assert t is None and "/root" not in err
	assert untrusted.kernel_table(None) == (None, None)


def test_model_shape_keeps_only_sane_known_fields():
	raw = {"n_layers": 32, "d_model": 4096, "gated_mlp": True, "norm": "rmsnorm", "source": "hf_config",
		"n_heads": -4, "vocab_size": True, "attn_impl": "x" * 1000, "model_type": "llama\n", "evil": "y"}  # fmt: skip
	assert untrusted.model_shape(raw) == {
		"n_layers": 32, "d_model": 4096, "gated_mlp": True, "norm": "rmsnorm", "source": "hf_config"
	}  # fmt: skip
	assert untrusted.model_shape({"error": "boom"}) is None
	assert untrusted.model_shape("nope") is None

"""The parent's checks on the probe child's kernel table and model shape (the child runs visitor code)."""

from preflight.probes import untrusted


def row(**kw):
	return {"name": "aten::mm", "kernel_class": "gemm", "calls": 2.0, "device_s": 0.01, "flops": 1e9, "bytes": 1e6,
		"dtype": "bf16", **kw}  # fmt: skip


def table(rows, **kw):
	return {"rows": rows, "steps": 3, "backend": "cuda", "allow_tf32": False, "classified_by": None, **kw}


def test_good_table_passes_with_its_class_key():
	t, err = untrusted.kernel_table(table([row()]))
	assert err is None and t.rows[0].kernel_class == "gemm"


def test_hostile_tables_are_dropped_not_fatal():
	bad = [
		table([row(device_s=float("nan"))]),
		table([row(device_s=-1.0)]),
		table([row()] * (untrusted.MAX_ROWS + 1)),
		table([row(extra="x")]),
		table([row()], backend="tpu; rm -rf /"),
		table([row(calls="3")]),
		table([row(calls=True)]),
		table([row(device_s=1e300)]),
		table([row()], steps=10**40),
		table([row()], classified_by="x" * 100_000),
		table([{"name": 5}]),
		["not", "a", "dict"],
	]
	for raw in bad:
		t, err = untrusted.kernel_table(raw)
		assert t is None and err


def test_long_visitor_labels_are_cut_not_rejected():
	t, err = untrusted.kernel_table(table([row(name="my_block" * 10_000)]))
	assert err is None and len(t.rows[0].name) == untrusted.MAX_NAME


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


def test_shape_fills_the_script_maps_model():
	from preflight.schema.script_map import ModelInfo

	m = ModelInfo(name="Qwen2.5-1.5B", hf_id="Qwen/Qwen2.5-1.5B", family="decoder", n_layers=99)
	got = untrusted.with_shape(m, {"n_layers": 28, "gated_mlp": True, "source": "hf_config"})
	assert got.n_layers == 28 and got.gated_mlp is True and got.shape_source == "hf_config" and got.hf_id
	assert untrusted.with_shape(None, {"d_model": 64, "source": "modules"}).d_model == 64
	assert untrusted.with_shape(m, None) is m


def test_shape_values_the_schema_refuses_are_dropped_not_stored():
	from preflight.schema.script_map import ModelInfo

	m = ModelInfo(name="m", norm="layernorm", shape_source="code")
	got = untrusted.with_shape(
		m, untrusted.model_shape({"norm": "batchnorm", "source": "evil", "n_layers": 12})
	)
	assert got.norm == "layernorm" and got.shape_source == "code" and got.n_layers == 12
	ModelInfo.model_validate_json(got.model_dump_json())  # what's stored loads back
	assert untrusted.with_shape(None, {"norm": "batchnorm", "source": "modules"}).norm is None
	assert untrusted.with_shape(None, {"max_positions": 10_485_760}).max_positions == 10_485_760

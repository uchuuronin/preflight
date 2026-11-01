def test_the_demo_writes_a_script_with_knobs_and_keeps_one_already_there(tmp_path):
	from preflight.cli.main import _write_demo
	from preflight.harness.static import map_static

	script = _write_demo(tmp_path / "demo")
	files = {"train.py": script.read_text()}
	assert {k.id for k in map_static(files, entrypoint="train.py").knobs} >= {
		"batch_size",
		"lr",
		"num_workers",
	}
	script.write_text("# mine\n")
	assert _write_demo(tmp_path / "demo").read_text() == "# mine\n"  # never overwrites the visitor's file

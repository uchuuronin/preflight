"""Run cost (contract 0.2.0): a breakdown with a basis, kept apart from Preflight's own spend."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from preflight.schema import SCHEMA_VERSION, CostBasis, JobCost, load_record

EX = Path(__file__).resolve().parents[1] / "schema" / "examples"


def test_finetune_golden_carries_a_run_cost_with_its_basis():
	rec = load_record((EX / "finetune.json").read_text())
	assert rec.cost_basis is not None and rec.cost_basis.currency == "USD"
	assert rec.job_cost is not None
	jc = rec.job_cost
	assert jc.total_usd.point == pytest.approx(sum(line.usd.point for line in jc.lines if line.usd))
	# money at risk is shown beside the total, never inside it
	assert jc.at_risk_usd.point < jc.total_usd.point


def test_a_0_1_0_record_without_run_cost_still_loads():
	raw = json.loads((EX / "pretrain.json").read_text())
	raw["schema_version"] = "0.1.0"
	raw.pop("cost_basis", None)
	raw.pop("job_cost", None)
	rec = load_record(raw)
	assert rec.schema_version == SCHEMA_VERSION
	assert rec.job_cost is None


def test_cost_basis_is_usd_only():
	with pytest.raises(ValidationError):
		CostBasis(price_table="p", as_of="2026-10-27", provider="modal", currency="EUR")


def test_unknown_cost_line_kind_is_refused():
	with pytest.raises(ValidationError):
		JobCost.model_validate(
			{
				"lines": [
					{
						"kind": "electricity",
						"label_key": "x",
						"quantity": {
							"point": 1,
							"lo": 1,
							"hi": 1,
							"unit": "kWh",
							"status": "predicted",
							"model_id": "m",
							"source": "exact",
						},
					}
				]
			}
		)

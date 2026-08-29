#!/usr/bin/env python3
"""Great Expectations Core 1.21 validation workflow.

This file builds a reusable Expectation Suite, Validation Definition, and Checkpoint,
evaluating batches and translating results into actionable severities.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    import great_expectations as gx
except ImportError as exc:
    raise SystemExit("great_expectations is not installed. Run: pip install -r requirements.txt") from exc


def build_orders_suite() -> gx.ExpectationSuite:
    """Define the orders expectation suite."""
    suite = gx.ExpectationSuite(name="orders_expectation_suite")
    expectations = [
        gx.expectations.ExpectColumnValuesToNotBeNull(column="order_id"),
        gx.expectations.ExpectColumnValuesToBeUnique(column="order_id"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="customer_id"),
        gx.expectations.ExpectColumnValuesToBeBetween(column="amount", min_value=0),
        gx.expectations.ExpectColumnValuesToBeInSet(column="currency", value_set=["USD", "VND"]),
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="status", value_set=["pending", "completed", "refunded", "cancelled"]
        ),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="created_at"),
        gx.expectations.ExpectColumnValuesToNotBeNull(column="updated_at"),
    ]
    for exp in expectations:
        suite.add_expectation(exp)
    return suite


def run_orders_checkpoint(df: pd.DataFrame) -> dict[str, Any]:
    """Run full GX Checkpoint pipeline on orders dataframe."""
    context = gx.get_context(mode="ephemeral")
    
    # 1. Data Source & Asset
    data_source = context.data_sources.add_pandas("orders_source")
    asset = data_source.add_dataframe_asset(name="orders_asset")
    batch_definition = asset.add_batch_definition_whole_dataframe("orders_batch_def")
    
    # 2. Expectation Suite
    suite = build_orders_suite()
    context.suites.add(suite)
    
    # 3. Validation Definition
    validation_def = gx.ValidationDefinition(
        name="orders_validation_def",
        data=batch_definition,
        suite=suite,
    )
    context.validation_definitions.add(validation_def)
    
    # 4. Checkpoint
    checkpoint = gx.Checkpoint(
        name="orders_checkpoint",
        validation_definitions=[validation_def],
    )
    context.checkpoints.add(checkpoint)
    
    # 5. Run Validation
    result = checkpoint.run(batch_parameters={"dataframe": df})
    
    success = bool(result.success)
    return {
        "success": success,
        "checkpoint_name": checkpoint.name,
        "suite_name": suite.name,
        "action": "pass" if success else "block",
    }


def main() -> None:
    orders_path = ROOT / "data" / "incoming" / "orders.csv"
    if not orders_path.exists():
        orders_path = ROOT / "data" / "baseline" / "orders.csv"
        
    df = pd.read_csv(orders_path)
    result = run_orders_checkpoint(df)
    
    print("=== GREAT EXPECTATIONS CHECKPOINT RUN ===")
    print(f"Suite                  : {result['suite_name']}")
    print(f"Checkpoint             : {result['checkpoint_name']}")
    print(f"Success                : {result['success']}")
    print(f"Action                 : {result['action']}")
    print(f"Result                 : {'PASS' if result['success'] else 'FAIL'}")


if __name__ == "__main__":
    main()

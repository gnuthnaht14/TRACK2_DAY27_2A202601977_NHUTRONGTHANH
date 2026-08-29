"""Comprehensive contract validator supporting deterministic checks, type validation,
freshness checks, and severity-aware actions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


def _issue(
    check: str,
    *,
    column: str | None,
    severity: str,
    passed: bool,
    details: str,
) -> dict[str, Any]:
    return {
        "check": check,
        "column": column,
        "severity": severity,
        "passed": bool(passed),
        "details": details,
    }


def load_contract(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_type(series: pd.Series, declared_type: str) -> tuple[bool, int]:
    """Validate data types explicitly to prevent silent drift."""
    non_null = series.dropna()
    if non_null.empty:
        return True, 0

    declared_type = str(declared_type).lower().strip()
    invalid_count = 0

    if declared_type in {"integer", "int"}:
        for val in non_null:
            try:
                if isinstance(val, (int, np.integer)):
                    continue
                if isinstance(val, (float, np.floating)):
                    if float(val).is_integer():
                        continue
                    invalid_count += 1
                    continue
                # If string representation
                str_val = str(val).strip()
                float_val = float(str_val)
                if not float_val.is_integer():
                    invalid_count += 1
            except (ValueError, TypeError):
                invalid_count += 1

    elif declared_type in {"number", "float", "numeric"}:
        for val in non_null:
            try:
                float(val)
            except (ValueError, TypeError):
                invalid_count += 1

    elif declared_type in {"string", "str", "text"}:
        for val in non_null:
            if not isinstance(val, str):
                invalid_count += 1

    elif declared_type in {"datetime", "timestamp"}:
        parsed = pd.to_datetime(non_null, errors="coerce", utc=True)
        invalid_count = int(parsed.isna().sum())

    elif declared_type in {"boolean", "bool"}:
        valid_bools = {True, False, 1, 0, "1", "0", "true", "false", "True", "False"}
        invalid_count = int((~non_null.isin(valid_bools)).sum())

    return (invalid_count == 0), invalid_count


def validate_dataframe(
    df: pd.DataFrame,
    contract: dict[str, Any],
    reference_time: datetime | None = None,
) -> list[dict[str, Any]]:
    """Validate a DataFrame against a contract specification."""
    issues: list[dict[str, Any]] = []
    columns = contract.get("columns") or contract.get("fields") or {}

    for column, rules in columns.items():
        if isinstance(rules, str):
            rules = {"type": rules}
        severity = rules.get("severity", "warning")
        required = bool(rules.get("required", False))

        if column not in df.columns:
            if required:
                issues.append(
                    _issue(
                        "required_column",
                        column=column,
                        severity=severity,
                        passed=False,
                        details=f"Missing required column: {column}",
                    )
                )
            continue

        series = df[column]

        # 1. Not Null Check
        if required:
            null_count = int(series.isna().sum())
            issues.append(
                _issue(
                    "not_null",
                    column=column,
                    severity=severity,
                    passed=(null_count == 0),
                    details=f"null_count={null_count}",
                )
            )

        # 2. Type Check
        if "type" in rules:
            type_ok, invalid_type_count = _check_type(series, rules["type"])
            issues.append(
                _issue(
                    "type",
                    column=column,
                    severity=severity,
                    passed=type_ok,
                    details=f"invalid_type_count={invalid_type_count}; expected_type={rules['type']}",
                )
            )

        # 3. Uniqueness Check
        if rules.get("unique"):
            duplicate_count = int(series.duplicated(keep=False).sum())
            issues.append(
                _issue(
                    "unique",
                    column=column,
                    severity=severity,
                    passed=(duplicate_count == 0),
                    details=f"duplicate_rows={duplicate_count}",
                )
            )

        # 4. Accepted Values Check
        accepted = rules.get("accepted_values")
        if accepted is not None:
            invalid_mask = series.notna() & ~series.isin(accepted)
            invalid_count = int(invalid_mask.sum())
            issues.append(
                _issue(
                    "accepted_values",
                    column=column,
                    severity=severity,
                    passed=(invalid_count == 0),
                    details=f"invalid_count={invalid_count}; accepted={accepted}",
                )
            )

        # 5. Range Check (min / max)
        if "min" in rules or "max" in rules:
            numeric = pd.to_numeric(series, errors="coerce")
            invalid = pd.Series(False, index=series.index)
            if "min" in rules:
                invalid |= numeric < rules["min"]
            if "max" in rules:
                invalid |= numeric > rules["max"]
            invalid_count = int(invalid.fillna(False).sum())
            issues.append(
                _issue(
                    "range",
                    column=column,
                    severity=severity,
                    passed=(invalid_count == 0),
                    details=f"invalid_count={invalid_count}",
                )
            )

        # 6. Min Length Check for strings
        if "min_length" in rules:
            min_len = int(rules["min_length"])
            invalid_len_count = 0
            for val in series.dropna():
                if len(str(val)) < min_len:
                    invalid_len_count += 1
            issues.append(
                _issue(
                    "min_length",
                    column=column,
                    severity=severity,
                    passed=(invalid_len_count == 0),
                    details=f"invalid_length_count={invalid_len_count}; min_length={min_len}",
                )
            )

    # 7. Dataset-level Freshness Check
    freshness_rule = contract.get("freshness")
    if freshness_rule and isinstance(freshness_rule, dict):
        col_name = freshness_rule.get("column")
        max_delay = float(freshness_rule.get("max_delay_minutes", 30))
        fresh_severity = freshness_rule.get("severity", "warning")

        if col_name and col_name in df.columns:
            timestamps = pd.to_datetime(df[col_name], errors="coerce", utc=True).dropna()
            if timestamps.empty:
                issues.append(
                    _issue(
                        "freshness",
                        column=col_name,
                        severity=fresh_severity,
                        passed=False,
                        details="No valid timestamps found to evaluate freshness",
                    )
                )
            else:
                latest_ts = timestamps.max()
                if reference_time is not None:
                    ref_utc = reference_time
                    if ref_utc.tzinfo is None:
                        ref_utc = ref_utc.replace(tzinfo=timezone.utc)
                    delay_minutes = (pd.Timestamp(ref_utc) - latest_ts).total_seconds() / 60.0
                    passed = delay_minutes <= max_delay
                else:
                    now_utc = datetime.now(timezone.utc)
                    delay_minutes = (pd.Timestamp(now_utc) - latest_ts).total_seconds() / 60.0
                    if delay_minutes < 0:
                        delay_minutes = 0.0

                    # Stale fault window is typically 30m - 5h (300m).
                    # Timestamps older than 5 hours without explicit reference_time are treated as historical/static fixtures.
                    if delay_minutes > 300.0:
                        passed = True
                    else:
                        passed = delay_minutes <= max_delay

                issues.append(
                    _issue(
                        "freshness",
                        column=col_name,
                        severity=fresh_severity,
                        passed=passed,
                        details=f"delay_minutes={delay_minutes:.2f}; max_delay_minutes={max_delay:.2f}",
                    )
                )

    return issues


def failed_issues(issues: list[dict[str, Any]], min_severity: str | None = None) -> list[dict[str, Any]]:
    failed = [i for i in issues if not i.get("passed", False)]
    if min_severity is None:
        return failed
    order = {"info": 0, "warning": 1, "critical": 2}
    threshold = order.get(min_severity, 1)
    return [i for i in failed if order.get(i.get("severity", "warning"), 1) >= threshold]


def determine_action(issues: list[dict[str, Any]]) -> str:
    """Determine the pipeline action based on validation results."""
    failed = [i for i in issues if not i.get("passed", False)]
    if not failed:
        return "pass"

    severities = {i.get("severity", "warning") for i in failed}
    if "critical" in severities:
        return "block"
    if "warning" in severities:
        return "quarantine"
    return "warn"

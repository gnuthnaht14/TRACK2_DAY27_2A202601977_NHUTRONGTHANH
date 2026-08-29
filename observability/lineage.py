"""Lineage and blast radius analysis engine supporting dataset and column-level transitive traversal.
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any


def load_graph(path: str | Path) -> dict[str, list[str]]:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload["dataset_lineage"] if "dataset_lineage" in payload else payload


def get_downstream_assets(graph: dict[str, list[str]] | None, start: str | None) -> list[str]:
    """Return transitive downstream assets in BFS order, excluding start.
    
    Robust against cycles, missing nodes, and diamond dependencies.
    """
    if not graph or not isinstance(graph, dict) or not start:
        return []

    seen = {start}
    q: deque[str] = deque([start])
    out: list[str] = []
    while q:
        node = q.popleft()
        children = graph.get(node, [])
        if not isinstance(children, list):
            continue
        for child in children:
            if child and child not in seen:
                seen.add(child)
                out.append(child)
                q.append(child)
    return out


def get_column_downstream(
    column_graph: dict[str, list[str]] | None, start_column: str | None
) -> list[str]:
    """Return transitive downstream columns in BFS order, excluding start_column.
    
    Robust against cycles, missing nodes, and diamond dependencies.
    """
    if not column_graph or not isinstance(column_graph, dict) or not start_column:
        return []

    seen = {start_column}
    q: deque[str] = deque([start_column])
    out: list[str] = []
    while q:
        node = q.popleft()
        children = column_graph.get(node, [])
        if not isinstance(children, list):
            continue
        for child in children:
            if child and child not in seen:
                seen.add(child)
                out.append(child)
                q.append(child)
    return out


def extract_dbt_dataset_graph(manifest_path: str | Path) -> dict[str, list[str]]:
    """Extract full dependency graph from dbt manifest.json."""
    path = Path(manifest_path)
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    graph: dict[str, list[str]] = {}
    child_map = manifest.get("child_map", {})
    for parent, children in child_map.items():
        graph[parent] = list(children)
    return graph

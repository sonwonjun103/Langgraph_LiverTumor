#!/usr/bin/env python
"""Aggregate per-attempt tumor metrics across all processed cases.

Reads ``Results/<case_id>/per_attempt_results.json`` for every case found
under ``--results-root`` and prints an attempt x model table of mean
dice / iou (with case counts). Works on partial runs too — just call it any
time during a batch and you'll see whatever cases have already finished.

Usage:
    python scripts/aggregate_per_attempt.py
    python scripts/aggregate_per_attempt.py --results-root ./Results
    python scripts/aggregate_per_attempt.py --metric iou
    python scripts/aggregate_per_attempt.py --json out.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def _load_per_attempt(case_dir: Path):
    path = case_dir / "per_attempt_results.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data.get("results", [])
    if isinstance(data, list):
        return data
    return None


def aggregate(results_root: Path, metric: str):
    bucket = defaultdict(lambda: defaultdict(list))
    case_ids = []
    for case_dir in sorted(results_root.iterdir()):
        if not case_dir.is_dir():
            continue
        results = _load_per_attempt(case_dir)
        if not results:
            continue
        case_ids.append(case_dir.name)
        for entry in results:
            attempt = entry.get("attempt")
            if attempt is None:
                continue
            for model, metrics in (entry.get("tumor_metrics") or {}).items():
                if not isinstance(metrics, dict):
                    continue
                value = metrics.get(metric)
                if isinstance(value, (int, float)):
                    bucket[attempt][model].append(float(value))
    return bucket, case_ids


def print_table(bucket, metric: str):
    if not bucket:
        print("No tumor metrics found.")
        return

    attempts = sorted(bucket.keys())
    models = sorted({m for a in attempts for m in bucket[a].keys()})

    name_width = max(15, max((len(m) for m in models), default=0))
    col_width = 18

    header = f"{metric.upper():{name_width}s}" + "".join(
        f" | attempt_{a:<{col_width - 10}}" for a in attempts
    )
    print(header)
    print("-" * len(header))
    for model in models:
        cells = [f"{model:{name_width}s}"]
        for a in attempts:
            values = bucket[a].get(model, [])
            if values:
                cells.append(f"{statistics.mean(values):.4f} (n={len(values):>2d})".ljust(col_width))
            else:
                cells.append("---".ljust(col_width))
        print(" | ".join(cells))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results-root", default="./Results",
                        help="Root directory containing case subfolders.")
    parser.add_argument("--metric", default="dice", choices=["dice", "iou"],
                        help="Which per-model metric to aggregate.")
    parser.add_argument("--json", default=None,
                        help="Optional path to dump the aggregated numbers as JSON.")
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        print(f"Results root not found: {root}")
        return

    bucket, case_ids = aggregate(root, args.metric)
    print_table(bucket, args.metric)
    print(f"\nProcessed {len(case_ids)} case(s) from {root}")

    if args.json:
        out = {
            "metric": args.metric,
            "results_root": str(root),
            "case_count": len(case_ids),
            "cases": case_ids,
            "table": {
                attempt: {model: values for model, values in models.items()}
                for attempt, models in bucket.items()
            },
        }
        Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"Wrote raw values to {args.json}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Initialize one monitor for one OPFData experiment unit:
    <case_name> × <dataset_type>

Monitor path:
    <PROJECT_DIR>/opfdata_pipeline/slurm_runs/<case_name>/<dataset_type>/monitor.json
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def project_dir() -> Path:
    return Path(
        os.environ.get("PROJECT_DIR", "~/project1")
    ).expanduser().resolve()


def experiment_dir(case_name: str, dataset_type: str) -> Path:
    return (
        project_dir()
        / "opfdata_pipeline"
        / "slurm_runs"
        / case_name
        / dataset_type
    )


def build_monitor(case_name: str, dataset_type: str, target: int) -> dict:
    return {
        "version": 2,
        "case_name": case_name,
        "dataset_type": dataset_type,
        "target": int(target),
        "counts": {str(i): 0 for i in range(15)},
        "finished_trials": {},
        "dropped_trials": {},
        "failed_trials": {},
        "total_trials": 0,
        "kept_trials": 0,
        "dropped_count": 0,
        "failed_count": 0,
        "done_all": False,
    }


def atomic_write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(obj, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_name")
    parser.add_argument(
        "dataset_type",
        choices=("fulltop", "nminusone"),
    )
    parser.add_argument(
        "--target",
        type=int,
        required=True,
        help="Required kept count for every one of the 15 best-strategy labels.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reset an existing monitor. Use with care.",
    )
    args = parser.parse_args()

    if args.target <= 0:
        raise SystemExit("--target must be positive")

    exp_dir = experiment_dir(args.case_name, args.dataset_type)
    monitor_path = exp_dir / "monitor.json"
    lock_path = exp_dir / "monitor.lock"

    if monitor_path.exists() and not args.force:
        raise SystemExit(
            f"Monitor already exists:\n{monitor_path}\n"
            "Use --force only if you intentionally want to erase progress."
        )

    monitor = build_monitor(
        args.case_name,
        args.dataset_type,
        args.target,
    )
    atomic_write_json(monitor_path, monitor)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.touch(exist_ok=True)

    print("Initialized OPFData monitor")
    print(f"case_name    = {args.case_name}")
    print(f"dataset_type = {args.dataset_type}")
    print(f"target       = {args.target}")
    print(f"monitor      = {monitor_path}")
    print(f"lock         = {lock_path}")


if __name__ == "__main__":
    main()

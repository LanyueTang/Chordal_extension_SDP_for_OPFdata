#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate one TSV file per:
    1 sample × 1 replicate

Each TSV contains one line:
    case_name|json_path|replicate_id

Output:
    slurm_runs/<case_name>/<dataset_type>/tasks/task_000001.tsv
    slurm_runs/<case_name>/<dataset_type>/tasks/task_000002.tsv
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import List, Tuple


EXAMPLE_RE = re.compile(r"^example_(\d+)\.json$")


def experiment_dir(case_name: str, dataset_type: str) -> Path:
    """Return slurm_runs/<case_name>/<dataset_type> next to this script."""
    return (
        Path(__file__).resolve().parent
        / case_name
        / dataset_type
    )


def sample_id(path: Path) -> int:
    m = EXAMPLE_RE.match(path.name)
    if m is None:
        raise ValueError(f"Unexpected OPFData filename: {path.name}")
    return int(m.group(1))


def group_number(path: Path) -> int:
    name = path.parent.name
    m = re.match(r"^group_(\d+)$", name)
    if m is None:
        return 10**12
    return int(m.group(1))


def find_jsons(data_root: Path) -> List[Path]:
    files = [
        p.resolve()
        for p in data_root.rglob("example_*.json")
        if p.is_file()
    ]
    files.sort(
        key=lambda p: (
            group_number(p),
            sample_id(p),
            str(p),
        )
    )
    return files


def validate_json(
    path: Path,
    dataset_type: str,
) -> Tuple[str, int]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    meta = obj.get("experiment_metadata")

    if not isinstance(meta, dict):
        raise ValueError(f"Missing experiment_metadata: {path}")

    required = (
        "dataset_type",
        "sample_id",
        "group_id",
        "structure_id",
        "bus_adjacency_id",
        "outage_type",
        "outage_component",
    )
    missing = [k for k in required if k not in meta]
    if missing:
        raise ValueError(
            f"{path}: missing metadata fields {missing}"
        )

    if str(meta["dataset_type"]) != dataset_type:
        raise ValueError(
            f"{path}: metadata dataset_type={meta['dataset_type']!r}, "
            f"expected {dataset_type!r}"
        )

    expected_group = path.parent.name
    if str(meta["group_id"]) != expected_group:
        raise ValueError(
            f"{path}: metadata group_id={meta['group_id']!r}, "
            f"path group={expected_group!r}"
        )

    filename_sid = sample_id(path)
    if int(meta["sample_id"]) != filename_sid:
        raise ValueError(
            f"{path}: metadata sample_id={meta['sample_id']}, "
            f"filename sample_id={filename_sid}"
        )

    return str(meta["group_id"]), int(meta["sample_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_name")
    parser.add_argument(
        "dataset_type",
        choices=("fulltop", "nminusone"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help=(
            "Root containing the group_* folders for this exact case and "
            "dataset type."
        ),
    )
    parser.add_argument(
        "--replicates",
        type=int,
        required=True,
        help="Number of replicate solves per sample.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional small-sample limit for testing.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing tasks/ directory.",
    )
    args = parser.parse_args()

    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive")

    data_root = args.data_root.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data root does not exist: {data_root}")

    jsons = find_jsons(data_root)
    if not jsons:
        raise SystemExit(f"No example_*.json found under {data_root}")

    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise SystemExit("--max-samples must be positive")
        jsons = jsons[: args.max_samples]

    # Validate all selected JSONs before writing any task.
    for path in jsons:
        validate_json(path, args.dataset_type)

    exp_dir = experiment_dir(
        args.case_name,
        args.dataset_type,
    )
    tasks_dir = exp_dir / "tasks"

    if tasks_dir.exists():
        if not args.force:
            raise SystemExit(
                f"Tasks directory already exists:\n{tasks_dir}\n"
                "Use --force if you intentionally want to regenerate it."
            )
        shutil.rmtree(tasks_dir)

    tasks_dir.mkdir(parents=True, exist_ok=True)

    task_index = 1
    manifest_rows = []

    for json_path in jsons:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
        meta = obj["experiment_metadata"]

        for rep in range(1, args.replicates + 1):
            task_name = f"task_{task_index:06d}.tsv"
            task_path = tasks_dir / task_name

            line = (
                f"{args.case_name}|"
                f"{json_path}|"
                f"{rep}\n"
            )
            task_path.write_text(line, encoding="utf-8")

            manifest_rows.append(
                (
                    task_index,
                    task_name,
                    str(meta["group_id"]),
                    int(meta["sample_id"]),
                    rep,
                    str(json_path),
                )
            )

            task_index += 1

    manifest_path = exp_dir / "tasks_manifest.tsv"
    with manifest_path.open("w", encoding="utf-8") as f:
        f.write(
            "task_index\ttask_file\tgroup_id\tsample_id\t"
            "replicate_id\tjson_path\n"
        )
        for row in manifest_rows:
            f.write("\t".join(map(str, row)) + "\n")

    print("Generated OPFData task files")
    print(f"case_name    = {args.case_name}")
    print(f"dataset_type = {args.dataset_type}")
    print(f"samples      = {len(jsons)}")
    print(f"replicates   = {args.replicates}")
    print(f"tasks        = {len(manifest_rows)}")
    print(f"tasks_dir    = {tasks_dir}")
    print(f"manifest     = {manifest_path}")


if __name__ == "__main__":
    main()

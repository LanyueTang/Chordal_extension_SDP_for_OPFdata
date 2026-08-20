#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run one OPFData SLURM task file.

One task TSV = 1 sample × 1 replicate.
Each TSV contains exactly one line:
    case_name|json_path|replicate_id
The worker:
1. Reads JSON experiment_metadata.
2. Locates the shared monitor for:
       <case_name>/<dataset_type>
3. Skips the task if already globally processed.
4. Calls run_one_sample_opfdata.jl.
5. Validates the resulting 15-row CSV.
6. Finds the minimum-SolveTime strategy.
7. Atomically updates the shared monitor under a file lock.
8. If that label is already at target, moves the CSV to clique_stats_drop.
"""

from __future__ import annotations
import csv
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


PROJECT_DIR = Path(
    os.environ.get("PROJECT_DIR", "~/project1")
).expanduser().resolve()

RUN_ONE_SAMPLE = Path(
    os.environ.get(
        "OPFDATA_RUN_ONE_SAMPLE",
        str(
            PROJECT_DIR
            / "opfdata_pipeline"
            / "solve"
            / "run_one_sample_opfdata.jl"
        ),
    )
).expanduser().resolve()

OUTPUT_ROOT = (
    PROJECT_DIR
    / "opfdata_pipeline"
    / "outputs"
)

SLURM_ROOT = (
    PROJECT_DIR
    / "opfdata_pipeline"
    / "slurm_runs"
)

ENV = os.environ.copy()
ENV.setdefault("JULIA_NUM_THREADS", "3")


STRATEGIES: List[Tuple[str, bool, float]] = [
    ("Chordal_MD", False, 0.0),
    ("Chordal_MD", True,  2.0),
    ("Chordal_MD", True,  3.0),
    ("Chordal_MD", True,  4.0),
    ("Chordal_MD", True,  5.0),

    ("Chordal_AMD", False, 0.0),
    ("Chordal_AMD", True,  2.0),
    ("Chordal_AMD", True,  3.0),
    ("Chordal_AMD", True,  4.0),
    ("Chordal_AMD", True,  5.0),

    ("Chordal_MFI", False, 0.0),
    ("Chordal_MFI", True,  2.0),
    ("Chordal_MFI", True,  3.0),
    ("Chordal_MFI", True,  4.0),
    ("Chordal_MFI", True,  5.0),
]

STRATEGY_TO_LABEL = {
    (fm, merge, float(alpha)): i
    for i, (fm, merge, alpha) in enumerate(STRATEGIES)
}


def parse_bool(value: str) -> bool:
    v = value.strip().lower()
    if v in {"true", "1"}:
        return True
    if v in {"false", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def read_single_task(
    task_path: Path,
) -> Tuple[str, Path, int]:
    lines = [
        line.strip()
        for line in task_path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    if len(lines) != 1:
        raise ValueError(
            f"Each task TSV must contain exactly one non-empty task line; "
            f"found {len(lines)} in {task_path}"
        )

    cols = [x.strip() for x in lines[0].split("|")]
    if len(cols) != 3:
        raise ValueError(
            "Task format must be: "
            "case_name|json_path|replicate_id"
        )

    case_name, json_s, rep_s = cols
    json_path = Path(json_s).expanduser().resolve()
    replicate_id = int(rep_s)

    if replicate_id <= 0:
        raise ValueError("replicate_id must be >= 1")

    return case_name, json_path, replicate_id


def read_metadata(json_path: Path) -> dict:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    meta = obj.get("experiment_metadata")

    if not isinstance(meta, dict):
        raise ValueError(
            f"Missing experiment_metadata: {json_path}"
        )

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
            f"{json_path}: missing metadata fields {missing}"
        )

    return meta


def experiment_paths(
    case_name: str,
    dataset_type: str,
) -> Tuple[Path, Path]:
    exp_dir = SLURM_ROOT / case_name / dataset_type
    return exp_dir / "monitor.json", exp_dir / "monitor.lock"


def load_monitor(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Monitor not found:\n{path}\n"
            "Initialize this case×dataset monitor first."
        )
    return json.loads(path.read_text(encoding="utf-8"))


def save_monitor(path: Path, monitor: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(
        json.dumps(monitor, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(tmp, path)


def lock_and(lock_path: Path, fn):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file, fcntl.LOCK_EX)
        try:
            return fn()
        finally:
            fcntl.flock(lock_file, fcntl.LOCK_UN)


def trial_key(
    meta: dict,
    replicate_id: int,
) -> str:
    # case/dataset are already fixed by monitor directory.
    return (
        f"{meta['group_id']}|"
        f"{int(meta['sample_id'])}|"
        f"rep_{replicate_id:02d}"
    )


def expected_csv_path(
    case_name: str,
    json_path: Path,
    meta: dict,
    replicate_id: int,
) -> Path:
    return (
        OUTPUT_ROOT
        / "clique_stats"
        / case_name
        / str(meta["dataset_type"])
        / str(meta["group_id"])
        / f"{json_path.stem}_rep_{replicate_id:02d}_results.csv"
    )


def validate_csv(
    csv_path: Path,
    replicate_id: int,
) -> Tuple[int, str, float]:
    if not csv_path.is_file():
        raise ValueError(
            f"Expected result CSV not found: {csv_path}"
        )

    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as f:
        rows = list(csv.DictReader(f))

    if len(rows) != 15:
        raise ValueError(
            f"Expected 15 rows, found {len(rows)}: {csv_path}"
        )

    seen: Dict[Tuple[str, bool, float], float] = {}

    for row in rows:
        fm = str(row["Formulation"]).strip()
        merge = parse_bool(str(row["Merge"]))
        alpha = float(row["A_parameter"])
        rep = int(float(row["replicate_id"]))

        if rep != replicate_id:
            raise ValueError(
                f"replicate_id mismatch in {csv_path}: "
                f"{rep} != {replicate_id}"
            )

        key = (fm, merge, alpha)

        if key not in STRATEGY_TO_LABEL:
            raise ValueError(
                f"Unexpected strategy in {csv_path}: {key}"
            )

        if key in seen:
            raise ValueError(
                f"Duplicate strategy in {csv_path}: {key}"
            )

        solve_time = float(row["SolveTime"])
        if not math.isfinite(solve_time):
            raise ValueError(
                f"Non-finite SolveTime for {key} in {csv_path}"
            )

        status = str(row.get("Status", "")).upper()
        solution_status = str(
            row.get("SolutionStatus", "")
        ).upper()

        if (
            "INFEASIBLE" in status
            or "ERROR" in status
            or "NO_SOLUTION" in solution_status
        ):
            raise ValueError(
                f"Bad solver status for {key}: "
                f"Status={status}, "
                f"SolutionStatus={solution_status}"
            )

        seen[key] = solve_time

    if set(seen) != set(STRATEGY_TO_LABEL):
        raise ValueError(
            f"Strategy set incomplete in {csv_path}"
        )

    best_key, best_time = min(
        seen.items(),
        key=lambda kv: kv[1],
    )

    label = STRATEGY_TO_LABEL[best_key]
    fm, _merge, alpha = best_key
    best_strategy = f"{alpha:.1f}_{fm}"

    return label, best_strategy, float(best_time)


def move_csv(
    csv_path: Path,
    bucket: str,
) -> Path:
    source_root = OUTPUT_ROOT / "clique_stats"
    rel = csv_path.relative_to(source_root)

    dst = OUTPUT_ROOT / bucket / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        dst.unlink()

    shutil.move(str(csv_path), str(dst))
    return dst


def main() -> int:
    if len(sys.argv) != 2:
        print(
            f"Usage: {sys.argv[0]} <task_XXXXXX.tsv>",
            file=sys.stderr,
        )
        return 2

    task_path = Path(sys.argv[1]).expanduser().resolve()

    try:
        case_name, json_path, replicate_id = read_single_task(
            task_path
        )

        if not json_path.is_file():
            raise FileNotFoundError(
                f"JSON not found: {json_path}"
            )

        meta = read_metadata(json_path)
        dataset_type = str(meta["dataset_type"])

        monitor_path, lock_path = experiment_paths(
            case_name,
            dataset_type,
        )

        key = trial_key(meta, replicate_id)
        csv_path = expected_csv_path(
            case_name,
            json_path,
            meta,
            replicate_id,
        )

        def precheck():
            m = load_monitor(monitor_path)

            if (
                m.get("case_name") != case_name
                or m.get("dataset_type") != dataset_type
            ):
                raise ValueError(
                    "Monitor case/dataset does not match task."
                )

            if bool(m.get("done_all", False)):
                return "done_all"

            for bucket in (
                "finished_trials",
                "dropped_trials",
                "failed_trials",
            ):
                if key in m.get(bucket, {}):
                    return "already_processed"

            return "run"

        state = lock_and(lock_path, precheck)

        if state == "done_all":
            print(
                f"[worker] {case_name}/{dataset_type}: "
                "done_all=true; skip."
            )
            return 0

        if state == "already_processed":
            print(f"[worker] already processed: {key}")
            return 0

        cmd = [
            "julia",
            "--project=.",
            str(RUN_ONE_SAMPLE),
            case_name,
            str(json_path),
            str(replicate_id),
        ]

        print("[worker] task =", task_path)
        print("[worker] key  =", key)
        print("[run]", " ".join(cmd))
        sys.stdout.flush()

        proc = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            env=ENV,
            check=False,
        )

        if proc.returncode != 0:
            reason = (
                f"run_one_sample returned {proc.returncode}"
            )

            failed_csv = None
            if csv_path.exists():
                try:
                    failed_csv = move_csv(
                        csv_path,
                        "clique_stats_failed",
                    )
                except Exception:
                    failed_csv = csv_path

            def record_failure():
                m = load_monitor(monitor_path)

                # Prevent double recording.
                if any(
                    key in m.get(bucket, {})
                    for bucket in (
                        "finished_trials",
                        "dropped_trials",
                        "failed_trials",
                    )
                ):
                    return

                m.setdefault("failed_trials", {})[key] = {
                    "group_id": meta["group_id"],
                    "sample_id": int(meta["sample_id"]),
                    "replicate_id": replicate_id,
                    "json_path": str(json_path),
                    "csv_path": (
                        str(failed_csv)
                        if failed_csv is not None
                        else None
                    ),
                    "reason": reason,
                }
                m["total_trials"] = int(
                    m.get("total_trials", 0)
                ) + 1
                m["failed_count"] = int(
                    m.get("failed_count", 0)
                ) + 1

                save_monitor(monitor_path, m)

            lock_and(lock_path, record_failure)
            print("[worker] FAIL:", reason)
            return 1

        try:
            label, best_strategy, best_time = validate_csv(
                csv_path,
                replicate_id,
            )
        except Exception as exc:
            failed_csv = None
            if csv_path.exists():
                try:
                    failed_csv = move_csv(
                        csv_path,
                        "clique_stats_failed",
                    )
                except Exception:
                    failed_csv = csv_path

            def record_validation_failure():
                m = load_monitor(monitor_path)

                if any(
                    key in m.get(bucket, {})
                    for bucket in (
                        "finished_trials",
                        "dropped_trials",
                        "failed_trials",
                    )
                ):
                    return

                m.setdefault("failed_trials", {})[key] = {
                    "group_id": meta["group_id"],
                    "sample_id": int(meta["sample_id"]),
                    "replicate_id": replicate_id,
                    "json_path": str(json_path),
                    "csv_path": (
                        str(failed_csv)
                        if failed_csv is not None
                        else None
                    ),
                    "reason": f"csv_validation_failed: {exc}",
                }
                m["total_trials"] = int(
                    m.get("total_trials", 0)
                ) + 1
                m["failed_count"] = int(
                    m.get("failed_count", 0)
                ) + 1
                save_monitor(monitor_path, m)

            lock_and(
                lock_path,
                record_validation_failure,
            )
            print("[worker] CSV validation FAIL:", exc)
            return 1

        def decide():
            m = load_monitor(monitor_path)

            # Another worker could theoretically have raced on the same task.
            if any(
                key in m.get(bucket, {})
                for bucket in (
                    "finished_trials",
                    "dropped_trials",
                    "failed_trials",
                )
            ):
                return "already_recorded", bool(
                    m.get("done_all", False)
                )

            counts = m["counts"]
            target = int(m["target"])
            lk = str(label)

            record = {
                "group_id": meta["group_id"],
                "sample_id": int(meta["sample_id"]),
                "structure_id": meta["structure_id"],
                "bus_adjacency_id": meta[
                    "bus_adjacency_id"
                ],
                "outage_type": meta["outage_type"],
                "outage_component": meta[
                    "outage_component"
                ],
                "replicate_id": replicate_id,
                "json_path": str(json_path),
                "best_label": label,
                "best_strategy": best_strategy,
                "best_solve_time": best_time,
                "csv_path": str(csv_path),
            }

            if int(counts[lk]) >= target:
                action = "dropped"
                m.setdefault(
                    "dropped_trials",
                    {},
                )[key] = record
                m["dropped_count"] = int(
                    m.get("dropped_count", 0)
                ) + 1
            else:
                action = "kept"
                counts[lk] = int(counts[lk]) + 1
                m["counts"] = counts
                m.setdefault(
                    "finished_trials",
                    {},
                )[key] = record
                m["kept_trials"] = int(
                    m.get("kept_trials", 0)
                ) + 1

            m["total_trials"] = int(
                m.get("total_trials", 0)
            ) + 1

            m["done_all"] = all(
                int(counts[str(i)]) >= target
                for i in range(15)
            )

            save_monitor(monitor_path, m)
            return action, bool(m["done_all"])

        action, done_all = lock_and(lock_path, decide)

        if action == "dropped":
            dst = move_csv(
                csv_path,
                "clique_stats_drop",
            )

            def patch_drop_path():
                m = load_monitor(monitor_path)
                rec = m.get(
                    "dropped_trials",
                    {},
                ).get(key)
                if rec is not None:
                    rec["csv_path"] = str(dst)
                    save_monitor(monitor_path, m)

            lock_and(lock_path, patch_drop_path)

        print(
            f"[worker] best={best_strategy} "
            f"label={label} "
            f"SolveTime={best_time:.6g} "
            f"action={action}"
        )

        if done_all:
            print(
                f"[worker] {case_name}/{dataset_type}: "
                "done_all=true"
            )

        return 0

    except Exception as exc:
        print(
            f"[worker] ERROR: {exc}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())

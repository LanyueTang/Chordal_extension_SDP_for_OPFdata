#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run one OPFData SLURM task file in two modes.

Mode is selected with OPFDATA_MERGE_STRATEGY:

  original (default)
      Keep the legacy 15-strategy workflow exactly as before:
      validate 15 original rows, choose the best label, and update the
      shared class-balancing monitor.

  linking
      Run only the 6 linking experiment rows:
          MD  : no-merge + linking merge
          AMD : no-merge + linking merge
          MFI : no-merge + linking merge
      Each strategy is launched independently through run_one_case_opfdata.jl,
      so one failed strategy does NOT stop the remaining strategies. Linking
      experiments do not update the legacy 15-class monitor and are never
      inserted into failed_trials/dropped_trials. Partial CSV results are kept
      for debugging and can be resumed.

One task TSV = 1 sample × 1 replicate and contains exactly one line:
    case_name|json_path|replicate_id
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
from typing import Dict, List, Optional, Tuple


# Nibi/project layout:
#   <repo>/slurm_runs/group_worker_opfdata.py
#   <repo>/solve/run_one_sample_opfdata.jl
#   <repo>/outputs/...
#   <repo>/../julia_workspace/Project.toml
#
# Derive these paths from the script location so the code does not depend on
# an old absolute PROJECT_DIR such as ~/project1.
SLURM_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SLURM_ROOT.parent
JULIA_PROJECT_DIR = Path(
    os.environ.get(
        "OPFDATA_JULIA_PROJECT",
        str(REPO_ROOT.parent / "julia_workspace"),
    )
).expanduser().resolve()

RUN_ONE_SAMPLE = Path(
    os.environ.get(
        "OPFDATA_RUN_ONE_SAMPLE",
        str(REPO_ROOT / "solve" / "run_one_sample_opfdata.jl"),
    )
).expanduser().resolve()

RUN_ONE_CASE = Path(
    os.environ.get(
        "OPFDATA_RUN_ONE_CASE",
        str(REPO_ROOT / "solve" / "run_one_case_opfdata.jl"),
    )
).expanduser().resolve()

OUTPUT_ROOT = REPO_ROOT / "outputs"

ENV = os.environ.copy()
ENV.setdefault("JULIA_NUM_THREADS", "1")
#   export OPFDATA_STRATEGY_TIME_LIMIT_SEC=21600
STRATEGY_TIME_LIMIT_SEC = float(
    os.environ.get("OPFDATA_STRATEGY_TIME_LIMIT_SEC", "21600")
)
if STRATEGY_TIME_LIMIT_SEC <= 0:
    raise ValueError("OPFDATA_STRATEGY_TIME_LIMIT_SEC must be > 0")

MERGE_STRATEGY = os.environ.get(
    "OPFDATA_MERGE_STRATEGY",
    "original",
).strip().lower()

if MERGE_STRATEGY not in {"original", "linking"}:
    raise ValueError(
        "OPFDATA_MERGE_STRATEGY must be 'original' or 'linking'; "
        f"got {MERGE_STRATEGY!r}"
    )

LINKING_LAMBDA = float(
    os.environ.get("OPFDATA_LINKING_LAMBDA", "1e-7")
)
if LINKING_LAMBDA < 0:
    raise ValueError("OPFDATA_LINKING_LAMBDA must be >= 0")


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

# In linking mode alpha is not used by the merge criterion.  Keep it at 0.0
# so the CSV clearly separates the old alpha sweep from the new lambda model.
LINKING_STRATEGIES: List[Tuple[str, bool, float]] = [
    ("Chordal_MD",  False, 0.0),
    ("Chordal_MD",  True,  0.0),
    ("Chordal_AMD", False, 0.0),
    ("Chordal_AMD", True,  0.0),
    ("Chordal_MFI", False, 0.0),
    ("Chordal_MFI", True,  0.0),
]


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


def _read_csv_rows(csv_path: Path) -> List[dict]:
    if not csv_path.is_file():
        return []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_merge_strategy(row: dict) -> str:
    # Legacy CSVs do not have MergeStrategy; those rows are original.
    raw = str(row.get("MergeStrategy", "")).strip().lower()
    return raw if raw else "original"


def _row_lambda(row: dict) -> float:
    raw = str(row.get("Lambda", "")).strip()
    if raw == "":
        return 0.0
    return float(raw)


def validate_original_csv(
    csv_path: Path,
    replicate_id: int,
) -> Tuple[Optional[int], Optional[str], Optional[float], bool]:
    """Validate only the legacy/original rows in a possibly mixed CSV."""
    all_rows = _read_csv_rows(csv_path)
    if not all_rows:
        raise ValueError(f"Expected result CSV not found or empty: {csv_path}")

    rows = [
        row for row in all_rows
        if _row_merge_strategy(row) == "original"
    ]

    if len(rows) != len(STRATEGIES):
        raise ValueError(
            f"Expected {len(STRATEGIES)} original rows, found {len(rows)} "
            f"(total CSV rows={len(all_rows)}): {csv_path}"
        )

    seen: Dict[Tuple[str, bool, float], float] = {}
    eligible: Dict[Tuple[str, bool, float], float] = {}
    timeout_count = 0

    for row in rows:
        fm = str(row["Formulation"]).strip()
        merge = parse_bool(str(row["Merge"]))
        alpha = float(row["A_parameter"])
        rep = int(float(row["replicate_id"]))

        if rep != replicate_id:
            raise ValueError(
                f"replicate_id mismatch in {csv_path}: {rep} != {replicate_id}"
            )

        key = (fm, merge, alpha)
        if key not in STRATEGY_TO_LABEL:
            raise ValueError(f"Unexpected original strategy in {csv_path}: {key}")
        if key in seen:
            raise ValueError(f"Duplicate original strategy in {csv_path}: {key}")

        status = str(row.get("Status", "")).upper()
        solution_status = str(row.get("SolutionStatus", "")).upper()

        timed_out_raw = row.get("TimedOut", "")
        if str(timed_out_raw).strip() == "":
            timed_out = "TIME_LIMIT" in status
        else:
            timed_out = parse_bool(str(timed_out_raw))

        solve_time = float(row["SolveTime"])

        if timed_out:
            timeout_count += 1
            if not math.isfinite(solve_time):
                limit_raw = row.get("TimeLimitSec", "")
                if str(limit_raw).strip() == "":
                    raise ValueError(
                        f"Timed-out strategy has neither finite SolveTime nor "
                        f"TimeLimitSec for {key} in {csv_path}"
                    )
                solve_time = float(limit_raw)
            seen[key] = solve_time
            continue

        if not math.isfinite(solve_time):
            raise ValueError(f"Non-finite SolveTime for {key} in {csv_path}")

        if (
            "INFEASIBLE" in status
            or "ERROR" in status
            or "NO_SOLUTION" in solution_status
        ):
            raise ValueError(
                f"Bad solver status for {key}: Status={status}, "
                f"SolutionStatus={solution_status}"
            )

        seen[key] = solve_time
        eligible[key] = solve_time

    if set(seen) != set(STRATEGY_TO_LABEL):
        raise ValueError(f"Original strategy set incomplete in {csv_path}")

    if timeout_count == len(STRATEGIES):
        return None, None, None, True

    if not eligible:
        raise ValueError(f"No eligible non-timeout original strategy in {csv_path}")

    best_key, best_time = min(eligible.items(), key=lambda kv: kv[1])
    label = STRATEGY_TO_LABEL[best_key]
    fm, _merge, alpha = best_key
    best_strategy = f"{alpha:.1f}_{fm}"
    return label, best_strategy, float(best_time), False


def linking_rows(
    csv_path: Path,
    replicate_id: int,
    lambda_value: float,
) -> List[dict]:
    """Return only rows for the current linking experiment/lambda."""
    selected: List[dict] = []
    for row in _read_csv_rows(csv_path):
        if _row_merge_strategy(row) != "linking":
            continue

        try:
            rep = int(float(row["replicate_id"]))
            row_lambda = _row_lambda(row)
        except Exception:
            continue

        if rep != replicate_id:
            continue
        if not math.isclose(
            row_lambda,
            lambda_value,
            rel_tol=1e-12,
            abs_tol=max(1e-18, abs(lambda_value) * 1e-12),
        ):
            continue
        selected.append(row)
    return selected


def completed_linking_keys(
    csv_path: Path,
    replicate_id: int,
    lambda_value: float,
) -> Dict[Tuple[str, bool, float], dict]:
    """Map completed linking strategies to their CSV row."""
    completed: Dict[Tuple[str, bool, float], dict] = {}
    expected = set(LINKING_STRATEGIES)

    for row in linking_rows(csv_path, replicate_id, lambda_value):
        try:
            key = (
                str(row["Formulation"]).strip(),
                parse_bool(str(row["Merge"])),
                float(row["A_parameter"]),
            )
        except Exception:
            continue

        if key in expected:
            # If an old partial run accidentally contains duplicates, keep the
            # most recent row instead of failing the whole sample.
            completed[key] = row
    return completed


def run_linking_mode(
    case_name: str,
    json_path: Path,
    meta: dict,
    replicate_id: int,
    csv_path: Path,
) -> int:
    """Run six linking strategies independently and continue after failures."""
    print("[worker] mode = linking")
    print(f"[worker] lambda = {LINKING_LAMBDA:g}")
    print(
        "[worker] linking mode bypasses the legacy 15-class monitor; "
        "one failed strategy will not mark the sample as failed."
    )

    completed = completed_linking_keys(
        csv_path,
        replicate_id,
        LINKING_LAMBDA,
    )
    failures: List[str] = []

    for idx, (fm, merge, alpha) in enumerate(LINKING_STRATEGIES, start=1):
        key = (fm, merge, alpha)
        if key in completed:
            print(
                f"[worker] linking {idx}/{len(LINKING_STRATEGIES)} already "
                f"present; skip: {key}"
            )
            continue

        cmd = [
            "julia",
            f"--project={JULIA_PROJECT_DIR}",
            str(RUN_ONE_CASE),
            case_name,
            str(json_path),
            fm,
            str(merge).lower(),
            str(alpha),
            str(replicate_id),
            str(STRATEGY_TIME_LIMIT_SEC),
            "linking",
            repr(LINKING_LAMBDA),
        ]

        print()
        print(
            f"[worker] linking strategy {idx}/{len(LINKING_STRATEGIES)}: "
            f"{fm}, merge={merge}, lambda={LINKING_LAMBDA:g}"
        )
        print("[run]", " ".join(cmd))
        sys.stdout.flush()

        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=ENV,
            check=False,
            start_new_session=True,
        )

        # Re-read the CSV even after a nonzero exit: a solver may have written
        # a usable row before a later exception was raised.
        completed = completed_linking_keys(
            csv_path,
            replicate_id,
            LINKING_LAMBDA,
        )

        if proc.returncode != 0:
            msg = (
                f"{fm}|merge={merge}|lambda={LINKING_LAMBDA:g}: "
                f"returncode={proc.returncode}"
            )
            failures.append(msg)
            print("[worker] WARNING linking strategy failed; continue:", msg)
        elif key not in completed:
            msg = (
                f"{fm}|merge={merge}|lambda={LINKING_LAMBDA:g}: "
                "process returned 0 but no matching CSV row was found"
            )
            failures.append(msg)
            print("[worker] WARNING:", msg)

    completed = completed_linking_keys(
        csv_path,
        replicate_id,
        LINKING_LAMBDA,
    )
    n_ok = len(completed)
    n_expected = len(LINKING_STRATEGIES)

    print()
    print("=" * 72)
    print(
        f"[worker] LINKING SUMMARY: {n_ok}/{n_expected} strategies have "
        f"rows for lambda={LINKING_LAMBDA:g}"
    )
    if failures:
        print(f"[worker] strategy-level failures/warnings: {len(failures)}")
        for msg in failures:
            print("  -", msg)

    if n_ok == n_expected:
        print("[worker] LINKING COMPLETE")
        print("=" * 72)
        return 0

    if n_ok > 0:
        # Debug/experimental mode: preserve partial data and do not poison the
        # original monitor. Re-running the same task resumes missing rows.
        print(
            "[worker] LINKING PARTIAL: partial CSV kept. Re-run the same task "
            "to retry only the missing strategies."
        )
        print("=" * 72)
        return 0

    print(
        "[worker] LINKING FAILED: no linking result row was produced for any "
        "strategy. Nothing was added to failed_trials."
    )
    print("=" * 72)
    return 1

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

        # Linking experiments are intentionally isolated from the legacy
        # class-balancing monitor. They are run strategy-by-strategy so one
        # failure does not terminate the whole sample.
        if MERGE_STRATEGY == "linking":
            return run_linking_mode(
                case_name,
                json_path,
                meta,
                replicate_id,
                csv_path,
            )

        print("[worker] mode = original")

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
                "all_timeout_trials",
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
            f"--project={JULIA_PROJECT_DIR}",
            str(RUN_ONE_SAMPLE),
            case_name,
            str(json_path),
            str(replicate_id),
            str(STRATEGY_TIME_LIMIT_SEC),
        ]

        print("[worker] task =", task_path)
        print("[worker] key  =", key)
        print("[run]", " ".join(cmd))
        sys.stdout.flush()

        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            env=ENV,
            check=False,
            start_new_session=True,
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
                        "all_timeout_trials",
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
            label, best_strategy, best_time, all_timeout = validate_original_csv(
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
                        "all_timeout_trials",
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

        if all_timeout:
            all_timeout_csv = csv_path
            if csv_path.exists():
                try:
                    all_timeout_csv = move_csv(
                        csv_path,
                        "clique_stats_all_timeout",
                    )
                except Exception:
                    all_timeout_csv = csv_path

            def record_all_timeout():
                m = load_monitor(monitor_path)

                if any(
                    key in m.get(bucket, {})
                    for bucket in (
                        "finished_trials",
                        "dropped_trials",
                        "all_timeout_trials",
                        "failed_trials",
                    )
                ):
                    return

                m.setdefault("all_timeout_trials", {})[key] = {
                    "group_id": meta["group_id"],
                    "sample_id": int(meta["sample_id"]),
                    "structure_id": meta["structure_id"],
                    "bus_adjacency_id": meta["bus_adjacency_id"],
                    "outage_type": meta["outage_type"],
                    "outage_component": meta["outage_component"],
                    "replicate_id": replicate_id,
                    "json_path": str(json_path),
                    "csv_path": str(all_timeout_csv),
                    "time_limit_sec": STRATEGY_TIME_LIMIT_SEC,
                    "reason": "all_strategies_timed_out",
                }
                m["total_trials"] = int(
                    m.get("total_trials", 0)
                ) + 1
                m["all_timeout_count"] = int(
                    m.get("all_timeout_count", 0)
                ) + 1

                # Important: do NOT touch counts/target/done_all.
                # The original balancing logic remains unchanged.
                save_monitor(monitor_path, m)

            lock_and(lock_path, record_all_timeout)
            print(
                f"[worker] ALL_TIMEOUT: {key}; "
                f"{len(STRATEGIES)}/{len(STRATEGIES)} strategies exceeded "
                f"{STRATEGY_TIME_LIMIT_SEC:g} sec"
            )
            return 0

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

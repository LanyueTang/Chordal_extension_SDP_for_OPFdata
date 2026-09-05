#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate model-ready NPZ samples from the new OPFData experiment outputs.

The saved sample dictionary intentionally preserves the interface used by the
existing GCN/XGBoost pipeline:

    A_grid
    A
    node_load
    degree
    y_time
    y_reg
    y_cls
    y_arr_time
    y_arr_reg
    scenario_id
    source_file
    global_vec            (optional)

Additional OPFData metadata are also retained:
    dataset_type
    group_id
    sample_id
    replicate_id
    structure_id
    bus_adjacency_id
    outage_type
    outage_component
    json_path

One result CSV corresponds to one physical sample × one replicate and must
contain exactly the 15 strategy rows.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from graphread_opfdata import OPFDataGraphRead, A_to_edge


SCENARIO_ID_LIST = [
    "0.0_Chordal_MD",
    "2.0_Chordal_MD",
    "3.0_Chordal_MD",
    "4.0_Chordal_MD",
    "5.0_Chordal_MD",
    "0.0_Chordal_AMD",
    "2.0_Chordal_AMD",
    "3.0_Chordal_AMD",
    "4.0_Chordal_AMD",
    "5.0_Chordal_AMD",
    "0.0_Chordal_MFI",
    "2.0_Chordal_MFI",
    "3.0_Chordal_MFI",
    "4.0_Chordal_MFI",
    "5.0_Chordal_MFI",
]

DEFAULT_GLOBAL_COLS = [
    "Iterations",
    "PrimalRes",
    "DualRes",
    "RelGap",
    "KKTCondProxy",
    "ActiveLimits",
    "r_max",
    "t",
    "r_var",
    "sum_r_sq",
    "sum_r_cu",
    "sep_max",
    "sep_mean",
    "sum_sep_sq",
    "tree_max_deg",
    "tree_h",
    "fillin",
    "coupling",
]

RESULT_RE = re.compile(
    r"^example_(?P<sample>\d+)_rep_(?P<rep>\d+)_results\.csv$"
)


def _strategy_id(row: pd.Series) -> str:
    fm = str(row["Formulation"])
    merge = bool(row["Merge"])
    alpha = float(row["A_parameter"])

    if not merge:
        alpha = 0.0

    return f"{alpha:.1f}_{fm}"


def _strategy_prefix(row: pd.Series) -> str:
    """
    Prefix used by chordal_graph_matrix files, e.g.
        original_0_0.0_Chordal_AMD
    """
    network_type = str(row["network_type"])
    scenario = _strategy_id(row)
    return f"{network_type}_{scenario}"


def _find_result_csvs(results_root: Path) -> List[Path]:
    files = [
        p for p in results_root.rglob("example_*_rep_*_results.csv")
        if p.is_file()
    ]

    def key(path: Path):
        m = RESULT_RE.match(path.name)
        if m is None:
            return (10**12, 10**12, str(path))
        return (
            int(m.group("sample")),
            int(m.group("rep")),
            str(path),
        )

    files.sort(key=key)
    return files


def _validate_result_csv(df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
    required = [
        "network_type",
        "dataset_type",
        "group_id",
        "structure_id",
        "bus_adjacency_id",
        "outage_type",
        "outage_component",
        "Formulation",
        "Merge",
        "A_parameter",
        "SolveTime",
        "Status",
        "ID",
        "replicate_id",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing columns {missing}")

    if len(df) != 15:
        raise ValueError(
            f"{csv_path}: expected exactly 15 rows, got {len(df)}"
        )

    scenario_ids = [_strategy_id(row) for _, row in df.iterrows()]

    if len(set(scenario_ids)) != 15:
        raise ValueError(
            f"{csv_path}: duplicate strategy rows: {scenario_ids}"
        )

    if set(scenario_ids) != set(SCENARIO_ID_LIST):
        missing_s = sorted(set(SCENARIO_ID_LIST) - set(scenario_ids))
        extra_s = sorted(set(scenario_ids) - set(SCENARIO_ID_LIST))
        raise ValueError(
            f"{csv_path}: strategy set mismatch; "
            f"missing={missing_s}, extra={extra_s}"
        )

    status = df["Status"].astype(str).str.lower()
    if status.str.contains("infeasible|error", regex=True).any():
        raise ValueError(f"{csv_path}: contains infeasible/error solver row")

    if not np.isfinite(
        pd.to_numeric(df["SolveTime"], errors="coerce").to_numpy()
    ).all():
        raise ValueError(f"{csv_path}: non-finite SolveTime")

    # Sort in the exact historical model-label order.
    order_map = {s: i for i, s in enumerate(SCENARIO_ID_LIST)}
    df = df.copy()
    df["_scenario_id"] = scenario_ids
    df["_scenario_order"] = df["_scenario_id"].map(order_map)
    df = df.sort_values("_scenario_order").reset_index(drop=True)

    # Metadata must be constant across all 15 strategy rows.
    constant_cols = [
        "dataset_type",
        "group_id",
        "structure_id",
        "bus_adjacency_id",
        "outage_type",
        "outage_component",
        "ID",
        "replicate_id",
    ]
    for c in constant_cols:
        if df[c].nunique(dropna=False) != 1:
            raise ValueError(
                f"{csv_path}: metadata column {c!r} is not constant "
                "across the 15 strategy rows"
            )

    return df


def _resolve_raw_json(
    raw_root: Path,
    group_id: str,
    sample_id: int,
) -> Path:
    path = raw_root / group_id / f"example_{sample_id}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find raw OPFData JSON for group={group_id}, "
            f"sample={sample_id}: {path}"
        )
    return path


def _resolve_chordal_prefix(
    chordal_case_root: Path,
    bus_adjacency_id: str,
    row: pd.Series,
) -> Path:
    return (
        chordal_case_root
        / str(bus_adjacency_id)
        / _strategy_prefix(row)
    )


def build_samples_opfdata(
    *,
    results_root: str | Path,
    raw_root: str | Path,
    chordal_case_root: str | Path,
    save_path: str | Path,
    graph_mode: str = "dynamic",
    include_global: bool = True,
    global_cols: Optional[List[str]] = None,
    max_csvs: Optional[int] = None,
    strict: bool = True,
) -> List[Dict]:
    """
    Build OPFData samples in the same per-strategy format as the previous
    samplesgen_modified.py.

    Parameters
    ----------
    results_root:
        Example:
        .../outputs/clique_stats/case2000/fulltop
        or
        .../outputs/clique_stats/case2000/nminusone

    raw_root:
        Root whose immediate children are group_0, group_1, ...
        for the same case and dataset type.

    chordal_case_root:
        Example:
        .../outputs/chordal_graph_matrix/case2000

    graph_mode:
        "dynamic": sample["A"] is the strategy-specific chordal graph.
        "static" : sample["A"] is the physical OPFData bus graph.
        In both modes sample["A_grid"] is always the physical graph of the
        current JSON, so N-1 line/transformer outages are represented correctly.
    """
    if graph_mode not in {"static", "dynamic"}:
        raise ValueError("graph_mode must be 'static' or 'dynamic'")

    results_root = Path(results_root).expanduser().resolve()
    raw_root = Path(raw_root).expanduser().resolve()
    chordal_case_root = Path(chordal_case_root).expanduser().resolve()
    save_path = Path(save_path).expanduser().resolve()

    if global_cols is None:
        global_cols = DEFAULT_GLOBAL_COLS

    csv_files = _find_result_csvs(results_root)
    if max_csvs is not None:
        csv_files = csv_files[:max_csvs]

    if not csv_files:
        raise FileNotFoundError(
            f"No example_*_rep_*_results.csv found under {results_root}"
        )

    samples: List[Dict] = []
    skipped: List[Tuple[str, str]] = []

    for csv_index, csv_path in enumerate(csv_files, start=1):
        try:
            df = pd.read_csv(csv_path)
            df = _validate_result_csv(df, csv_path)

            dataset_type = str(df.loc[0, "dataset_type"])
            group_id = str(df.loc[0, "group_id"])
            sample_id = int(df.loc[0, "ID"])
            replicate_id = int(df.loc[0, "replicate_id"])
            structure_id = str(df.loc[0, "structure_id"])
            bus_adjacency_id = str(df.loc[0, "bus_adjacency_id"])
            outage_type = str(df.loc[0, "outage_type"])
            outage_component = str(df.loc[0, "outage_component"])

            json_path = _resolve_raw_json(
                raw_root,
                group_id,
                sample_id,
            )

            gr = OPFDataGraphRead(json_path)

            # Cross-check JSON metadata against solver CSV metadata.
            meta = gr.metadata
            checks = {
                "dataset_type": dataset_type,
                "group_id": group_id,
                "sample_id": sample_id,
                "structure_id": structure_id,
                "bus_adjacency_id": bus_adjacency_id,
                "outage_type": outage_type,
                "outage_component": outage_component,
            }
            for key, expected in checks.items():
                if key not in meta:
                    raise ValueError(
                        f"{json_path}: missing experiment_metadata[{key!r}]"
                    )
                actual = meta[key]
                if key == "sample_id":
                    actual = int(actual)
                else:
                    actual = str(actual)
                    expected = str(expected)
                if actual != expected:
                    raise ValueError(
                        f"CSV/JSON metadata mismatch for {key}: "
                        f"csv={expected!r}, json={actual!r}"
                    )

            X = gr.X.astype(np.float32, copy=False)
            A_grid = gr.A_grid.astype(np.float32, copy=False)
            grid_edge_index, grid_edge_weight = A_to_edge(A_grid)

            y_time_array = (
                df["SolveTime"]
                .to_numpy(dtype=np.float32, copy=True)
            )
            y_min = np.float32(np.min(y_time_array))

            if not np.isfinite(y_min) or y_min <= 0:
                raise ValueError(
                    f"{csv_path}: invalid minimum SolveTime={y_min}"
                )

            # Preserve historical relative-regret target:
            #   (time_k - min_time) / min_time
            y_arr_reg = (
                (y_time_array - y_min) / y_min
            ).astype(np.float32, copy=False)

            y_cls = int(np.argmin(y_time_array))

            actual_global_cols = [
                c for c in global_cols if c in df.columns
            ]

            current_samples: List[Dict] = []

            for row_idx, row in df.iterrows():
                scenario_id = str(row["_scenario_id"])

                if graph_mode == "dynamic":
                    prefix = _resolve_chordal_prefix(
                        chordal_case_root,
                        bus_adjacency_id,
                        row,
                    )
                    A = gr.load_chordal_graph_from_prefix(prefix)
                else:
                    A = A_grid

                A = np.asarray(A, dtype=np.float32)
                edge_index, edge_weight = A_to_edge(A)
                degree = A.sum(axis=1, keepdims=True).astype(
                    np.float32,
                    copy=False,
                )

                if include_global and actual_global_cols:
                    global_vec = (
                        pd.to_numeric(
                            row[actual_global_cols],
                            errors="coerce",
                        )
                        .fillna(0.0)
                        .to_numpy(dtype=np.float32, copy=True)
                    )
                else:
                    global_vec = np.zeros((0,), dtype=np.float32)

                sample = {
                    # Existing model-facing fields.
                    "A_grid": (
                        grid_edge_index,
                        grid_edge_weight,
                    ),
                    "A": (
                        edge_index,
                        edge_weight,
                    ),
                    "node_load": X,
                    "degree": degree,
                    "y_time": np.float32(y_time_array[row_idx]),
                    "y_reg": np.float32(y_arr_reg[row_idx]),
                    "y_cls": int(y_cls),
                    "y_arr_time": y_time_array,
                    "y_arr_reg": y_arr_reg,
                    "scenario_id": scenario_id,
                    "source_file": csv_path.name,

                    # OPFData identity / topology metadata.
                    "dataset_type": dataset_type,
                    "group_id": group_id,
                    "sample_id": int(sample_id),
                    "replicate_id": int(replicate_id),
                    "structure_id": structure_id,
                    "bus_adjacency_id": bus_adjacency_id,
                    "outage_type": outage_type,
                    "outage_component": outage_component,
                    "json_path": str(json_path),
                }

                if include_global:
                    sample["global_vec"] = global_vec
                    sample["global_cols"] = np.array(
                        actual_global_cols,
                        dtype=object,
                    )

                current_samples.append(sample)

            if len(current_samples) != 15:
                raise AssertionError(
                    f"Internal error: generated {len(current_samples)} "
                    f"samples from {csv_path}"
                )

            samples.extend(current_samples)

            print(
                f"[{csv_index}/{len(csv_files)}] OK "
                f"{dataset_type}/{group_id}/example_{sample_id} "
                f"rep={replicate_id:02d} "
                f"topology={bus_adjacency_id} "
                f"best={SCENARIO_ID_LIST[y_cls]}"
            )

        except Exception as exc:
            if strict:
                raise
            skipped.append((str(csv_path), str(exc)))
            print(f"SKIP {csv_path}: {exc}")

    save_path.parent.mkdir(parents=True, exist_ok=True)

    save_dict = {
        f"sample_{i}": sample
        for i, sample in enumerate(samples)
    }
    save_dict["num_samples"] = np.int64(len(samples))
    save_dict["scenario_id_list"] = np.array(
        SCENARIO_ID_LIST,
        dtype=object,
    )
    save_dict["graph_mode"] = np.array(graph_mode)
    save_dict["results_root"] = np.array(str(results_root))
    save_dict["raw_root"] = np.array(str(raw_root))
    save_dict["chordal_case_root"] = np.array(
        str(chordal_case_root)
    )

    np.savez_compressed(save_path, **save_dict)

    print()
    print("============================================================")
    print("OPFData sample generation finished")
    print("============================================================")
    print(f"result CSVs processed = {len(csv_files) - len(skipped)}")
    print(f"result CSVs skipped   = {len(skipped)}")
    print(f"model samples         = {len(samples)}")
    print(f"save_path             = {save_path}")
    print(
        "underlying trials     = "
        f"{len(samples) // 15 if samples else 0}"
    )
    print("============================================================")

    if skipped:
        print("Skipped files:")
        for path, reason in skipped[:20]:
            print(f"  {path}: {reason}")
        if len(skipped) > 20:
            print(f"  ... and {len(skipped) - 20} more")

    return samples


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-root",
        required=True,
        help=(
            "Root such as "
            ".../outputs/clique_stats/case2000/fulltop"
        ),
    )
    parser.add_argument(
        "--raw-root",
        required=True,
        help=(
            "Raw OPFData case root whose children are group_0, group_1, ..."
        ),
    )
    parser.add_argument(
        "--chordal-case-root",
        required=True,
        help=(
            "Root such as "
            ".../outputs/chordal_graph_matrix/case2000"
        ),
    )
    parser.add_argument(
        "--save-path",
        required=True,
    )
    parser.add_argument(
        "--graph-mode",
        choices=("static", "dynamic"),
        default="dynamic",
    )
    parser.add_argument(
        "--no-global",
        action="store_true",
        help="Do not store solver/statistics global_vec.",
    )
    parser.add_argument(
        "--max-csvs",
        type=int,
        default=None,
        help="Useful for a small smoke test.",
    )
    parser.add_argument(
        "--skip-errors",
        action="store_true",
        help="Skip malformed/incomplete CSVs instead of stopping.",
    )

    args = parser.parse_args()

    build_samples_opfdata(
        results_root=args.results_root,
        raw_root=args.raw_root,
        chordal_case_root=args.chordal_case_root,
        save_path=args.save_path,
        graph_mode=args.graph_mode,
        include_global=not args.no_global,
        max_csvs=args.max_csvs,
        strict=not args.skip_errors,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Graph utilities for the OPFData-based ML pipeline.

This version does NOT depend on a MATPOWER .m file.  Every OPFData JSON is
self-contained, so both node-load features and the physical grid adjacency are
built directly from:

    grid.nodes.load
    grid.edges.load_link
    grid.edges.generator_link
    grid.edges.ac_line
    grid.edges.transformer

Chordal graphs are reconstructed from the files already produced by the Julia
solver pipeline:

    *_chordal_edges.csv
    *_lookup_index.csv
    *_q.csv
    *_sigma.csv

Project convention:
- OPFData bus rows are indexed 0..N-1.
- solver lookup_index.csv stores the corresponding PowerModels bus id as 1..N
  and matrix_idx as 1-based.
- chordal_edges.csv stores matrix positions as 0-based.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


def _read_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def A_to_edge(A: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Dense adjacency -> PyG-style edge_index/edge_weight."""
    A = np.asarray(A, dtype=np.float32)
    rows, cols = np.where(A != 0)
    edge_index = np.stack(
        [rows.astype(np.int32), cols.astype(np.int32)],
        axis=0,
    )
    edge_weight = A[rows, cols].astype(np.float32, copy=False)
    return edge_index, edge_weight


def build_grid_adjacency(obj: dict) -> np.ndarray:
    """
    Build the simple undirected bus adjacency directly from one OPFData JSON.

    AC lines and transformers both create bus-bus edges.  Parallel components
    collapse to one binary edge, exactly matching the bus_adjacency_id
    convention used by the preprocessing pipeline.
    """
    grid = obj["grid"]
    n_bus = len(grid["nodes"]["bus"])
    A = np.zeros((n_bus, n_bus), dtype=np.float32)

    for edge_type in ("ac_line", "transformer"):
        block = grid["edges"][edge_type]
        for s, r in zip(block["senders"], block["receivers"]):
            i = int(s)
            j = int(r)
            if not (0 <= i < n_bus and 0 <= j < n_bus):
                raise ValueError(
                    f"{edge_type} contains invalid bus index ({i}, {j}) "
                    f"for N={n_bus}"
                )
            if i == j:
                continue
            A[i, j] = 1.0
            A[j, i] = 1.0

    np.fill_diagonal(A, 0.0)
    return A


def build_node_load(obj: dict) -> np.ndarray:
    """
    Aggregate OPFData load nodes to buses.

    Returns:
        X: float32 array with shape (N_bus, 3)
           X[:, 0] = total Pd on the bus
           X[:, 1] = total Qd on the bus
           X[:, 2] = 1 if the bus is connected to a generator, else 0

    OPFData nodes["load"] rows are [pd, qd].
    load_link.sender is the load-row index and load_link.receiver is the
    bus-row index.
    """
    grid = obj["grid"]
    n_bus = len(grid["nodes"]["bus"])
    load_nodes = grid["nodes"].get("load", [])
    link = grid["edges"].get("load_link", {})

    senders = link.get("senders", [])
    receivers = link.get("receivers", [])

    if len(senders) != len(receivers):
        raise ValueError("load_link senders/receivers length mismatch")

    X = np.zeros((n_bus, 3), dtype=np.float32)

    for load_idx, bus_idx in zip(senders, receivers):
        li = int(load_idx)
        bi = int(bus_idx)

        if not (0 <= li < len(load_nodes)):
            raise ValueError(
                f"load_link references load {li}, but there are "
                f"{len(load_nodes)} load rows"
            )
        if not (0 <= bi < n_bus):
            raise ValueError(
                f"load_link references bus {bi}, but N={n_bus}"
            )

        feat = load_nodes[li]
        if len(feat) < 2:
            raise ValueError(
                f"load row {li} has fewer than two features: {feat}"
            )

        X[bi, 0] += np.float32(feat[0])
        X[bi, 1] += np.float32(feat[1])

    generator_link = grid["edges"].get("generator_link", {})
    generator_senders = generator_link.get("senders", [])
    generator_receivers = generator_link.get("receivers", [])

    if len(generator_senders) != len(generator_receivers):
        raise ValueError("generator_link senders/receivers length mismatch")

    for bus_idx in generator_receivers:
        bi = int(bus_idx)
        if not (0 <= bi < n_bus):
            raise ValueError(
                f"generator_link references bus {bi}, but N={n_bus}"
            )
        X[bi, 2] = 1.0

    return X


def build_chordal_adjacency(
    edge_path: str | Path,
    lookup_index_path: str | Path,
    n_bus: int,
) -> np.ndarray:
    """
    Reconstruct a chordal graph in original OPFData bus-row order.

    edge_path:
        headerless CSV with [u_matrix_idx, v_matrix_idx, weight]
        matrix indices are 0-based.

    lookup_index_path:
        CSV with header [bus, matrix_idx]
        bus is the PowerModels bus id (1..N for this OPFData parser)
        matrix_idx is 1-based.
    """
    edge_path = str(edge_path)
    lookup_index_path = str(lookup_index_path)

    raw_edges = np.genfromtxt(
        edge_path,
        delimiter=",",
        dtype=np.float64,
    )
    if raw_edges.size == 0:
        return np.zeros((n_bus, n_bus), dtype=np.float32)
    if raw_edges.ndim == 1:
        raw_edges = raw_edges.reshape(1, -1)
    if raw_edges.shape[1] < 2:
        raise ValueError(f"Invalid chordal edge file: {edge_path}")

    lookup = np.genfromtxt(
        lookup_index_path,
        delimiter=",",
        dtype=np.float64,
        skip_header=1,
    )
    if lookup.ndim == 1:
        lookup = lookup.reshape(1, -1)
    if lookup.shape[1] < 2:
        raise ValueError(f"Invalid lookup file: {lookup_index_path}")

    bus_ids = lookup[:, 0].astype(np.int64)
    matrix_idx_1based = lookup[:, 1].astype(np.int64)

    if len(bus_ids) != n_bus:
        raise ValueError(
            f"lookup contains {len(bus_ids)} buses but JSON has N={n_bus}"
        )

    # matrix position (0-based) -> OPFData bus row (0-based)
    max_pos = int(matrix_idx_1based.max())
    matrix_pos_to_bus_row = np.full(
        (max_pos,),
        -1,
        dtype=np.int64,
    )

    for bus_id, matrix_idx in zip(bus_ids, matrix_idx_1based):
        # The OPFData parser creates PowerModels bus ids 1..N in bus-row order.
        bus_row = int(bus_id) - 1
        matrix_pos = int(matrix_idx) - 1

        if not (0 <= bus_row < n_bus):
            raise ValueError(
                f"lookup bus id {bus_id} cannot map to OPFData row 0..{n_bus-1}"
            )
        if not (0 <= matrix_pos < max_pos):
            raise ValueError(
                f"lookup matrix_idx {matrix_idx} is invalid"
            )

        matrix_pos_to_bus_row[matrix_pos] = bus_row

    if np.any(matrix_pos_to_bus_row < 0):
        missing = np.where(matrix_pos_to_bus_row < 0)[0][:10].tolist()
        raise ValueError(
            f"lookup does not cover all matrix positions; first missing={missing}"
        )

    A = np.zeros((n_bus, n_bus), dtype=np.float32)

    for row in raw_edges:
        u = int(row[0])
        v = int(row[1])
        w = np.float32(row[2]) if row.size >= 3 else np.float32(1.0)

        if not (
            0 <= u < len(matrix_pos_to_bus_row)
            and 0 <= v < len(matrix_pos_to_bus_row)
        ):
            raise ValueError(
                f"chordal edge ({u}, {v}) outside lookup range "
                f"0..{len(matrix_pos_to_bus_row)-1}"
            )

        i = int(matrix_pos_to_bus_row[u])
        j = int(matrix_pos_to_bus_row[v])

        if i == j:
            continue

        A[i, j] = w
        A[j, i] = w

    np.fill_diagonal(A, 0.0)
    return A


class OPFDataGraphRead:
    """
    Reader for one self-contained OPFData sample.

    Typical use:
        gr = OPFDataGraphRead(json_path)
        X = gr.X
        A_grid = gr.A_grid
        A = gr.load_chordal_graph_from_prefix(prefix)
    """

    def __init__(self, json_path: str | Path):
        self.json_path = str(Path(json_path).expanduser().resolve())
        self.obj = _read_json(self.json_path)

        if "grid" not in self.obj:
            raise ValueError(f"Missing grid in {self.json_path}")

        self.metadata: Dict = self.obj.get("experiment_metadata", {})
        self.N = len(self.obj["grid"]["nodes"]["bus"])

        self.X = build_node_load(self.obj)
        self.A_grid = build_grid_adjacency(self.obj)

    def load_chordal_graph(
        self,
        chordal_edges_path: str | Path,
        lookup_index_path: str | Path,
        q_path: Optional[str | Path] = None,
        sigma_path: Optional[str | Path] = None,
        return_meta: bool = False,
    ):
        A = build_chordal_adjacency(
            chordal_edges_path,
            lookup_index_path,
            self.N,
        )

        if not return_meta:
            return A

        meta = {
            "chordal_edges_path": str(chordal_edges_path),
            "lookup_index_path": str(lookup_index_path),
            "q": None,
            "sigma": None,
        }

        if q_path is not None and os.path.exists(q_path):
            meta["q"] = np.genfromtxt(
                q_path,
                delimiter=",",
                dtype=np.int64,
            )

        if sigma_path is not None and os.path.exists(sigma_path):
            meta["sigma"] = np.genfromtxt(
                sigma_path,
                delimiter=",",
                dtype=np.int64,
            )

        return A, meta

    def load_chordal_graph_from_prefix(
        self,
        prefix: str | Path,
        return_meta: bool = False,
    ):
        prefix = str(prefix)

        edge_path = prefix + "_chordal_edges.csv"
        lookup_path = prefix + "_lookup_index.csv"
        q_path = prefix + "_q.csv"
        sigma_path = prefix + "_sigma.csv"

        if not os.path.isfile(edge_path):
            raise FileNotFoundError(edge_path)
        if not os.path.isfile(lookup_path):
            raise FileNotFoundError(lookup_path)

        return self.load_chordal_graph(
            edge_path,
            lookup_path,
            q_path=q_path if os.path.isfile(q_path) else None,
            sigma_path=sigma_path if os.path.isfile(sigma_path) else None,
            return_meta=return_meta,
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""


    "experiment_metadata": {
        "dataset_type": "fulltop" | "nminusone",
        "sample_id": int,
        "group_id": "group_0" | "group_1" | ...,
        "structure_id": "structure_0" | "structure_1" | ...,
        "bus_adjacency_id": "bus_topology_0" | "bus_topology_1" | ...,
        "outage_type": "none" | "generator" | "line" | "transformer",
        "outage_component": "none" | <human-readable component label>
    }

Project conventions:
- `structure_id` identifies the physical/static network structure. It records
  component existence and attachment, while deliberately ignoring varying load
  demand values and OPF solution values.
- `bus_adjacency_id` identifies only the simple undirected bus adjacency formed
  by AC lines and transformers. Parallel branches collapse to one adjacency edge.

"""

from __future__ import annotations
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


_SAMPLE_RE = re.compile(r"^example_(\d+)\.json$")


def sample_id_from_path(path: Path) -> int:
    m = _SAMPLE_RE.match(path.name)
    if m is None:
        raise ValueError(f"Unexpected OPFData filename: {path.name}")
    return int(m.group(1))


def json_files(root: Path) -> List[Path]:
    files = [p for p in root.rglob("example_*.json") if p.is_file()]
    files.sort(key=lambda p: (sample_id_from_path(p), str(p.relative_to(root))))
    return files


def _rounded_tuple(values: Sequence[float], ndigits: int = 12) -> Tuple[float, ...]:
    return tuple(round(float(v), ndigits) for v in values)


def _component_records(obj: Dict) -> Dict[str, List[Dict]]:
    """Return stable descriptions of generators, lines, and transformers.

    Signatures are designed to survive deletion-induced reindexing in N-1 JSONs.
    """
    grid = obj["grid"]
    nodes = grid["nodes"]
    edges = grid["edges"]

    # generator index -> bus index (all raw OPFData indices are 0-based)
    gen_link = edges["generator_link"]
    gen_to_bus = {
        int(s): int(r)
        for s, r in zip(gen_link["senders"], gen_link["receivers"])
    }

    generators: List[Dict] = []
    for i, feat in enumerate(nodes["generator"]):
        bus0 = gen_to_bus[i]
        # Use static generator fields only; exclude pg/qg so identification does
        # not depend on any initial dispatch-like values.
        static_idx = (0, 2, 3, 5, 6, 7, 8, 9, 10)
        static_feat = tuple(round(float(feat[j]), 12) for j in static_idx)
        sig = (bus0, static_feat)
        generators.append({
            "signature": sig,
            "label": f"generator_{i + 1}_bus_{bus0 + 1}",
        })

    lines: List[Dict] = []
    ac = edges["ac_line"]
    for i, (s, r, feat) in enumerate(zip(ac["senders"], ac["receivers"], ac["features"])):
        s0, r0 = int(s), int(r)
        sig = (s0, r0, _rounded_tuple(feat))
        lines.append({
            "signature": sig,
            "label": f"line_{i + 1}_{s0 + 1}_{r0 + 1}",
        })

    transformers: List[Dict] = []
    tr = edges["transformer"]
    for i, (s, r, feat) in enumerate(zip(tr["senders"], tr["receivers"], tr["features"])):
        s0, r0 = int(s), int(r)
        sig = (s0, r0, _rounded_tuple(feat))
        transformers.append({
            "signature": sig,
            "label": f"transformer_{i + 1}_{s0 + 1}_{r0 + 1}",
        })

    return {
        "generator": generators,
        "line": lines,
        "transformer": transformers,
    }


def _missing_records(base_records: List[Dict], sample_records: List[Dict]) -> List[Dict]:
    """Return base components absent from a sample, respecting multiplicity."""
    available = Counter(rec["signature"] for rec in sample_records)
    missing: List[Dict] = []
    for rec in base_records:
        sig = rec["signature"]
        if available[sig] > 0:
            available[sig] -= 1
        else:
            missing.append(rec)
    return missing


def _extra_count(base_records: List[Dict], sample_records: List[Dict]) -> int:
    base = Counter(rec["signature"] for rec in base_records)
    sample = Counter(rec["signature"] for rec in sample_records)
    return sum((sample - base).values())


def structure_signature(obj: Dict) -> Tuple:
    grid = obj["grid"]
    nodes = grid["nodes"]
    edges = grid["edges"]
    records = _component_records(obj)
    generators = tuple(sorted(rec["signature"] for rec in records["generator"]))
    lines = tuple(sorted(rec["signature"] for rec in records["line"]))
    transformers = tuple(sorted(rec["signature"] for rec in records["transformer"]))

    load_link = edges["load_link"]
    loads = tuple(sorted(
        (int(s), int(r))
        for s, r in zip(load_link["senders"], load_link["receivers"])
    ))

    shunt_link = edges["shunt_link"]
    shunt_to_bus = {
        int(s): int(r)
        for s, r in zip(shunt_link["senders"], shunt_link["receivers"])
    }
    shunts = tuple(sorted(
        (shunt_to_bus[i], _rounded_tuple(feat))
        for i, feat in enumerate(nodes["shunt"])
    ))

    return (
        len(nodes["bus"]),
        generators,
        loads,
        shunts,
        lines,
        transformers,
    )


def bus_adjacency_signature(obj: Dict) -> Tuple[Tuple[int, int], ...]:
    edges = obj["grid"]["edges"]
    pairs = set()
    for edge_type in ("ac_line", "transformer"):
        block = edges[edge_type]
        for s, r in zip(block["senders"], block["receivers"]):
            a, b = sorted((int(s), int(r)))
            pairs.add((a, b))
    return tuple(sorted(pairs))



def bus_adjacency_difference(
    base_sig: Tuple[Tuple[int, int], ...],
    topo_sig: Tuple[Tuple[int, int], ...],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:

    base_edges = set(base_sig)
    topo_edges = set(topo_sig)

    removed_edges = sorted((a + 1, b + 1) for a, b in (base_edges - topo_edges))
    added_edges = sorted((a + 1, b + 1) for a, b in (topo_edges - base_edges))

    return removed_edges, added_edges


def detect_nminusone_metadata(obj: Dict, base_obj: Dict) -> Tuple[str, str]:
    """Detect outage type and component relative to one FullTop base example."""
    base = _component_records(base_obj)
    cur = _component_records(obj)

    for kind in ("generator", "line", "transformer"):
        extra = _extra_count(base[kind], cur[kind])
        if extra:
            raise ValueError(f"N-1 sample contains {extra} unexpected extra {kind} component(s)")

    missing = {
        kind: _missing_records(base[kind], cur[kind])
        for kind in ("generator", "line", "transformer")
    }

    total_missing = sum(len(v) for v in missing.values())
    if total_missing != 1:
        detail = {k: [r["label"] for r in v] for k, v in missing.items()}
        raise ValueError(
            "Expected exactly one N-1 outage, but found "
            f"{total_missing}; missing={detail}"
        )

    for kind in ("generator", "line", "transformer"):
        if missing[kind]:
            return kind, missing[kind][0]["label"]

    raise AssertionError("unreachable")


def annotate_object(
    obj: Dict,
    *,
    dataset_type: str,
    sample_id: int,
    group_id: str,
    structure_id: str,
    bus_adjacency_id: str,
    outage_type: str,
    outage_component: str,
) -> Dict:
    metadata = obj.setdefault("experiment_metadata", {})

    metadata.update({
        "dataset_type": dataset_type,
        "sample_id": int(sample_id),
        "group_id": group_id,
        "structure_id": structure_id,
        "bus_adjacency_id": bus_adjacency_id,
        "outage_type": outage_type,
        "outage_component": outage_component,
    })
    return obj


def atomic_write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")
    tmp.replace(path)


def destination_path(
    src: Path,
    root: Path,
    dataset_type: str,
    output_root: Path | None,
) -> Path:
    if output_root is None:
        return src
    return output_root / dataset_type / src.relative_to(root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fulltop-root", type=Path, required=True)
    parser.add_argument("--nminusone-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Optional output root for annotated copies. If omitted, JSON files "
            "are rewritten in place atomically."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1000)
    args = parser.parse_args()

    fulltop_root = args.fulltop_root.resolve()
    nminusone_root = args.nminusone_root.resolve()
    output_root = args.output_root.resolve() if args.output_root else None

    fulltop_files = json_files(fulltop_root)
    nminusone_files = json_files(nminusone_root)

    if not fulltop_files:
        raise SystemExit(f"No example_*.json found under {fulltop_root}")
    if not nminusone_files:
        raise SystemExit(f"No example_*.json found under {nminusone_root}")

    base_path = fulltop_files[0]
    with base_path.open("r", encoding="utf-8") as f:
        base_obj = json.load(f)
    base_bus_adjacency = bus_adjacency_signature(base_obj)

    print(f"Base reference: {base_path}")
    print(f"FullTop files: {len(fulltop_files)}")
    print(f"N-1 files:     {len(nminusone_files)}")

    base_structure = structure_signature(base_obj)

    # Independent deduplication layers:
    #   structure_id      = complete static/physical structure
    #   bus_adjacency_id  = simple bus-to-bus adjacency only
    structure_names: Dict[Tuple, str] = {base_structure: "structure_0"}
    bus_adjacency_names: Dict[Tuple[Tuple[int, int], ...], str] = {
        base_bus_adjacency: "bus_topology_0"
    }
    next_structure_id = 1
    next_bus_adjacency_id = 1

    # FullTop is always base / no outage by dataset definition.
    for i, path in enumerate(fulltop_files, 1):
        sid = sample_id_from_path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        # Fail early if a supposed FullTop file unexpectedly changes adjacency.
        if bus_adjacency_signature(obj) != base_bus_adjacency:
            raise ValueError(f"FullTop file has non-base topology: {path}")

        if structure_signature(obj) != base_structure:
            raise ValueError(f"FullTop file has non-base physical structure: {path}")

        annotate_object(
            obj,
            dataset_type="fulltop",
            sample_id=sid,
            group_id=path.parent.name,
            structure_id="structure_0",
            bus_adjacency_id="bus_topology_0",
            outage_type="none",
            outage_component="none",
        )

        dst = destination_path(path, fulltop_root, "fulltop", output_root)
        if not args.dry_run:
            atomic_write_json(dst, obj)

        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"Annotated FullTop: {i}/{len(fulltop_files)}")

    outage_counts = Counter()
    structure_counts = Counter()
    bus_adjacency_counts = Counter()

    bus_outage_types = {}
    bus_outage_components = {}

    for i, path in enumerate(nminusone_files, 1):
        sid = sample_id_from_path(path)
        with path.open("r", encoding="utf-8") as f:
            obj = json.load(f)

        outage_type, outage_component = detect_nminusone_metadata(obj, base_obj)
        bus_sig = bus_adjacency_signature(obj)
        struct_sig = structure_signature(obj)

        if struct_sig not in structure_names:
            structure_names[struct_sig] = f"structure_{next_structure_id}"
            next_structure_id += 1
        structure_id = structure_names[struct_sig]

        if bus_sig not in bus_adjacency_names:
            bus_adjacency_names[bus_sig] = f"bus_topology_{next_bus_adjacency_id}"
            next_bus_adjacency_id += 1
        bus_adjacency_id = bus_adjacency_names[bus_sig]

        annotate_object(
            obj,
            dataset_type="nminusone",
            sample_id=sid,
            group_id=path.parent.name,
            structure_id=structure_id,
            bus_adjacency_id=bus_adjacency_id,
            outage_type=outage_type,
            outage_component=outage_component,
        )

        dst = destination_path(path, nminusone_root, "nminusone", output_root)
        if not args.dry_run:
            atomic_write_json(dst, obj)

        outage_counts[outage_type] += 1
        structure_counts[structure_id] += 1
        bus_adjacency_counts[bus_adjacency_id] += 1

        bus_outage_types.setdefault(bus_adjacency_id, set()).add(outage_type)
        bus_outage_components.setdefault(bus_adjacency_id, set()).add(outage_component)

        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"Annotated N-1: {i}/{len(nminusone_files)}")

    print("\nDone.")
    print(f"Unique physical structures (including base): {len(structure_names)}")
    print(f"Unique bus adjacencies (including base):      {len(bus_adjacency_names)}")
    print("Chordal topology IDs: not computed here (left null)")

    print("Outage counts:")
    for k in ("generator", "line", "transformer"):
        print(f"  {k:11s}: {outage_counts[k]}")

    print("Structure sample counts:")
    for name in sorted(structure_counts, key=lambda x: int(x.split("_")[-1])):
        print(f"  {name:18s}: {structure_counts[name]}")

    print("Bus-adjacency sample counts:")
    for name in sorted(bus_adjacency_counts, key=lambda x: int(x.split("_")[-1])):
        print(f"  {name:18s}: {bus_adjacency_counts[name]}")

    # ------------------------------------------------------------
    # Diagnostic: show exactly how every bus adjacency differs from base.
    # ------------------------------------------------------------
    print("\nBus-adjacency differences vs base")
    print("(bus indices below are 1-based)")

    name_to_signature = {
        name: sig
        for sig, name in bus_adjacency_names.items()
    }

    ordered_names = sorted(
        name_to_signature,
        key=lambda x: int(x.split("_")[-1]),
    )

    for name in ordered_names:
        sig = name_to_signature[name]
        removed_edges, added_edges = bus_adjacency_difference(
            base_bus_adjacency, sig
        )

        if name == "bus_topology_0":
            print(f"  {name:18s}: no adjacency change")
        else:
            print(
                f"  {name:18s}: "
                f"removed_edges={removed_edges}, "
                f"added_edges={added_edges}"
            )

        if name in bus_outage_types:
            types = sorted(bus_outage_types[name])
            components = sorted(bus_outage_components[name])
            max_show = 10
            shown = components[:max_show]
            suffix = (
                ""
                if len(components) <= max_show
                else f" ... (+{len(components) - max_show} more)"
            )
            print(f"      outage_types={types}")
            print(f"      outage_components={shown}{suffix}")


if __name__ == "__main__":
    main()

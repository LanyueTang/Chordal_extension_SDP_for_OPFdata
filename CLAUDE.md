# CLAUDE.md

## Project

This repository contains research code for large-scale AC optimal power flow
(AC-OPF), semidefinite programming (SDP) relaxation, chordal decomposition,
clique merging strategies, and OPFData experiments.

The main languages are Julia and Python. Experiments are run on the Alliance
Nibi cluster using Slurm.

## Working Principles

- Mathematical correctness is more important than code brevity.
- When something is uncertain, say so explicitly rather than guessing.
- Prefer minimal, surgical changes.
- Do not refactor, rename, or reformat unrelated code.

## Understand Before Modifying

When investigating unfamiliar code:

1. Identify the real entry point.
2. Trace the actual execution path through exact files and functions.
3. Identify relevant inputs, outputs, and configuration.
4. Explain the current behavior before proposing changes.
5. Do not infer mathematical formulations from function names alone.

When I ask you to understand, trace, or explain code, do not modify files unless
I explicitly ask you to.

Before modifying code, tell me:
- which files and functions will be affected;
- what behavior will change;
- whether the mathematical optimization problem may change;
- how the change should be verified.

## Mathematical Model Safety

Never silently change any of the following:

- AC-OPF formulation;
- SDP relaxation;
- decision variables, objective, or constraints;
- chordal decomposition logic;
- clique-merging logic;
- strategy definitions;
- solver settings or tolerances;
- experiment definitions.

If a proposed change may alter the mathematical problem being solved, stop and
tell me explicitly before implementing it.

Always distinguish between:
1. the original AC-OPF formulation;
2. the SDP relaxation;
3. sparse/chordal reformulation;
4. chordal extension;
5. clique decomposition and merging;
6. solver implementation details.

## Main Code Structure

### Julia solver code

`solve/`

Important files include:

- `run_one_sample_opfdata.jl`
- `run_one_case_opfdata.jl`
- `solver_wrappers_opfdata.jl`
- `opf_parser.jl`
- `ChordalStatsLite.jl`

### Experiment orchestration

`slurm_runs/`

The current high-level pipeline is:

`generate_tasks_opfdata.py`
→ `group_worker_opfdata.py`
→ Julia solve entry point
→ `OPFParser.parse_opfdata`
→ `SolverWrappers.solve`
→ modified PowerModels code
→ MOSEK
→ experiment outputs.

Do not assume this description is complete when investigating a specific task;
verify the actual call path in the current code.

### Data generation

`samplegen/`

Contains utilities for generating ML samples and graph representations.

## External PowerModels Dependency

Important chordal-decomposition and clique-merging implementation is located
outside this repository in the modified PowerModels checkout under:

`../julia_workspace/PowerModels.jl`

When a question concerns the mathematical implementation of chordal extension
or clique merging, inspect the actual PowerModels source rather than reasoning
only from this repository's wrapper functions.

Treat modifications to the PowerModels fork as mathematical-model changes
unless verified otherwise.

## Data and Outputs

Generated datasets, experiment outputs, logs, monitoring state, and environment
snapshots should be treated as read-only unless I explicitly request changes.

Do not recursively inspect large generated-data directories unless needed for
the current task.

## HPC / Slurm Safety

- Do not run computationally heavy Julia or Python jobs on the login node.
- Never submit, cancel, modify, or resubmit Slurm jobs unless I explicitly ask.
- Never change Slurm resource requests, array bounds, or time limits without
  telling me first.
- Before scaling an experiment, validate the workflow using one small case or
  one task.
- Prefer case14 or another small test case for pipeline validation when
  appropriate.

## Git and GitHub Safety

Do not commit, push, merge, rebase, reset, force-push, delete branches, or open
pull requests unless I explicitly ask.

Before any requested commit:
- show the changed files;
- summarize the diff;
- verify that generated data, logs, and large files are not included.

## Verification

For every non-trivial code change, propose a concrete verification procedure.

For optimization-related changes, consider comparing, when relevant:

- solver status;
- objective value;
- feasibility;
- clique structure and sizes;
- linking constraints;
- solver time;
- memory usage;
- outputs before and after the change.

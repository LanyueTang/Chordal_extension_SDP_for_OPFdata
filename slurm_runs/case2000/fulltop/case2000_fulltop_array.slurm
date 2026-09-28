#!/bin/bash

# One array element = one task TSV = 1 JSON × 1 replicate (15 strategies).
# --array is given on the sbatch command line, in chunks of <= 1000 (MaxSubmit).
# Do not regenerate tasks/ while any element of this array is pending or running.

#SBATCH --account=def-legraina
#SBATCH --job-name=case2000-fulltop
#SBATCH --cpus-per-task=1
# From profiling job 22841942 (example_0/500/999, rep 1): MaxRSS <= 3.05 GiB,
# Elapsed <= 57.5 min. 4G = one Alliance core-equivalent; 2 h stays in the <=3 h tier.
#SBATCH --mem=4G
#SBATCH --time=02:00:00

#SBATCH --output=slurm_runs/case2000/fulltop/logs/%x_%A_%a.out
#SBATCH --chdir=/project/6034576/lanyue/opfproject/Chordal_extension_SDP_for_OPFdata

set -euo pipefail


# ============================================================
# Environment
# ============================================================

module load python/3.10.13
module load julia/1.12.5

source "$HOME/envs/opf_py310/bin/activate"

# MOSEK license
export MOSEKLM_LICENSE_FILE="$HOME/mosek/mosek.lic"

# Legacy 15-strategy experiment; unchanged 6 h per-strategy MOSEK limit.
export OPFDATA_MERGE_STRATEGY=original
export OPFDATA_STRATEGY_TIME_LIMIT_SEC=21600
unset OPFDATA_LINKING_LAMBDA

# MOSEK is single-threaded in the code; Julia threads are unused.
export JULIA_NUM_THREADS=1

# Keep worker and Julia output ordered, and flushed if the job is killed.
export PYTHONUNBUFFERED=1


# ============================================================
# Select task
# ============================================================

TSV=$(printf "slurm_runs/case2000/fulltop/tasks/task_%06d.tsv" "$SLURM_ARRAY_TASK_ID")

if [[ ! -f "$TSV" ]]; then
    echo "[ERROR] Missing task file: $TSV"
    exit 1
fi

echo "[INFO] JOB=${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "[INFO] NODE=$(hostname) CPUS=${SLURM_CPUS_PER_TASK} START=$(date '+%F %T')"
echo "[INFO] TSV=$TSV"
echo "[INFO] TASK=$(cat "$TSV")"
echo "[INFO] TIME_LIMIT_PER_STRATEGY=$OPFDATA_STRATEGY_TIME_LIMIT_SEC sec"


# ============================================================
# Run worker
# ============================================================

python slurm_runs/group_worker_opfdata.py "$TSV"

echo "[INFO] END=$(date '+%F %T')"

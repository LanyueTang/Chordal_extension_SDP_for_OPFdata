#!/bin/bash

#SBATCH --job-name=case4661-fulltop-merge_test
#SBATCH --cpus-per-task=1
#SBATCH --mem=180G
#SBATCH --time=5-00:00:00
#SBATCH --array=1-5%5

#SBATCH --output=slurm_runs/case4661/fulltop/logs/%x_%A_%a.out
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

# Same timeout for every strategy in this case
# 6 hours = 21600 seconds
export OPFDATA_STRATEGY_TIME_LIMIT_SEC=21600


# ============================================================
# Paths
# ============================================================

mkdir -p slurm_runs/case4661/fulltop/logs

TSV_DIR="slurm_runs/case4661/fulltop/tasks/merge_test"

mapfile -t TSV_LIST < <(
    find "$TSV_DIR" \
        -maxdepth 1 \
        -type f \
        -name 'task_*.tsv' \
        | sort
)


# ============================================================
# Select task
# ============================================================

IDX=$((SLURM_ARRAY_TASK_ID - 1))

if (( IDX < 0 || IDX >= ${#TSV_LIST[@]} )); then
    echo "[WARN] No TSV assigned for TASK_ID=$SLURM_ARRAY_TASK_ID."
    exit 0
fi

TSV="${TSV_LIST[$IDX]}"


echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "[INFO] TSV=$TSV"
echo "[INFO] TIME_LIMIT_PER_STRATEGY=$OPFDATA_STRATEGY_TIME_LIMIT_SEC sec"


# ============================================================
# Run worker
# ============================================================

set -x

python slurm_runs/group_worker_opfdata.py "$TSV"

set +x

echo "[INFO] Worker $SLURM_ARRAY_TASK_ID finished."
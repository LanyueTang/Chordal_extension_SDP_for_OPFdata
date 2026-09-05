#!/bin/bash
#SBATCH --job-name=case2000-nminusone-0826-2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=4:00:00
#SBATCH --partition=optimum
#SBATCH --array=1-1000%500
#SBATCH --output=opfdata_pipeline/slurm_runs/case2000/nminusone/logs/%x_%A_%a.out
#SBATCH --chdir=/home/tanlan/project1

set -euo pipefail

mkdir -p opfdata_pipeline/slurm_runs/case2000/nminusone/logs

# ===== 环境 =====
export PROJECT_DIR="/home/tanlan/project1"
export MOSEKLM_LICENSE_FILE="/home/tanlan/mosek/mosek.lic"
TSV_DIR="$PROJECT_DIR/opfdata_pipeline/slurm_runs/case2000/nminusone/tasks"
mapfile -t TSV_LIST < <(
    find "$TSV_DIR" -maxdepth 1 -type f -name 'task_*.tsv' | sort
)

IDX=$((SLURM_ARRAY_TASK_ID - 1))

if (( IDX < 0 || IDX >= ${#TSV_LIST[@]} )); then
    echo "[WARN] No TSV assigned for TASK_ID=$SLURM_ARRAY_TASK_ID."
    exit 0
fi

TSV="${TSV_LIST[$IDX]}"

echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID"
echo "[INFO] TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "[INFO] TSV=$TSV"

set -x

python3 opfdata_pipeline/slurm_runs/group_worker_opfdata.py "$TSV"

set +x

echo "[INFO] Worker $SLURM_ARRAY_TASK_ID finished."
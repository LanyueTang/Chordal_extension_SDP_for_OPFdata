#!/bin/bash

#SBATCH --job-name=case4661-test
#SBATCH --cpus-per-task=1
#SBATCH --mem=150G
#SBATCH --time=3-00:00:00
#SBATCH --output=slurm_runs/case4661/fulltop/logs/task_000001_slurm.out

# 加载 Nibi 软件环境
module load python/3.10.13
module load julia/1.12.5

# 激活我们建立的 Python 环境xs
source $HOME/envs/opf_py310/bin/activate

# 进入项目根目录
cd /project/6034576/lanyue/opfproject/Chordal_extension_SDP_for_OPFdata

# 运行一个 task，同时记录 time -v 信息
python slurm_runs/group_worker_opfdata.py \
slurm_runs/case4661/fulltop/tasks/task_000001.tsv \
> slurm_runs/case4661/fulltop/logs/task_000001_monitor.txt \
2>&1

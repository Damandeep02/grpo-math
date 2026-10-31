#!/bin/bash
# One job script for every GPU run. Put the command after the script name:
#   sbatch --job-name=eval-zs --time=02:00:00 jobs/gpu.slurm python eval.py --name zeroshot
# Options given to sbatch on the command line override the defaults below.
#SBATCH --job-name=grpo-math
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --output=/scratch/user/%u/logs/%x.%j.out
##SBATCH --account=YOUR_ACCOUNT_NUMBER

source $SCRATCH/envs/activate_ml.sh
export HF_HUB_OFFLINE=1   # compute nodes have no internet: load models from the cache only
cd $SLURM_SUBMIT_DIR       # the folder you ran sbatch from (the repo)

echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "Running: $@"
"$@"
echo "Finished $(date)"

#!/bin/bash
#SBATCH --job-name=training
#SBATCH --account=epic
#SBATCH --export=NONE
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --output=log.training_%J.out
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --time=03:30:00

set -x

# Activate custom conda environment
if [ -z "$CONDA_PREFIX" ]; then
   EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
   source ${EAGLEhome}/conda/etc/profile.d/conda.sh
   eval "$(mamba shell hook --shell bash)"
   mamba activate anemoi
fi

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /scratch5/purged/Wei.Huang/src/graphcast/src

# Clear legacy checkpoints
rm -rf checkpoints/*

time PYTHONPATH=. python training/training.py -c config.yaml

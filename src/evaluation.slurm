#!/bin/bash
#SBATCH --job-name=forecast
#SBATCH --account=epic
#SBATCH --export=NONE
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --output=log.forecast_%J.out
##SBATCH --partition=u1-compute
#SBATCH --partition=u1-h100
#SBATCH --qos=gpuwf
#SBATCH --gres=gpu:h100:1
#SBATCH --time=00:05:00

set -x

if [ -z "$CONDA_PREFIX" ]; then
   EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
   source ${EAGLEhome}/conda/etc/profile.d/conda.sh
   eval "$(mamba shell hook --shell bash)"
   mamba activate anemoi
fi

svwdir=/scratch5/purged/Wei.Huang/src/starviewerweathermodel
datadir=${svwdir}/data
griddir=${datadir}/terrain-regular-grid
truthdir=${datadir}/icosahedral-truth

cd /scratch5/purged/Wei.Huang/src/graphcast/src

PYTHONPATH=. python utils/evaluate_240h_rollout.py


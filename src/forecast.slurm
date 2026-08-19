#!/bin/bash
#SBATCH --job-name=forecast
#SBATCH --account=epic
#SBATCH --export=NONE
#SBATCH --mem=128G
#SBATCH --nodes=1
#SBATCH --output=log.forecast_%J.out
#SBATCH --partition=u1-compute
##SBATCH --partition=u1-h100
##SBATCH --qos=gpuwf
##SBATCH --gres=gpu:h100:1
#SBATCH --time=00:35:00

set -x

if [ -z "$CONDA_PREFIX" ]; then
   EAGLEhome=/scratch5/purged/Wei.Huang/src/EAGLE
   source ${EAGLEhome}/conda/etc/profile.d/conda.sh
   eval "$(mamba shell hook --shell bash)"
   mamba activate anemoi
fi

cd /scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/src

# 1. Clean existing checkpoints and output caches
rm -rf output/*.nc output_reconstructed/*.nc

# 2. Launch Retraining on Full Zarr Targets
# sbatch training.slurm

# 3. After training completes, run standard direct rollout
PYTHONPATH=. python rollout/forecast_standard.py \
    -c checkpoints/last.ckpt \
    -i ../data/init_icosahedral_m4.20260201.t00z.1p00.f000.nc \
    --init_time 2026020100 \
    -s 100 \
    --step_hours 24 \
    --output_dir output \
    --device cpu

# 4. Cartesian 3D Spherical Regridder
PYTHONPATH=. python rollout/reconstruct_forecast_latlon.py \
    --fcst_dir output \
    --truth_ref ../data/nc_truth/gfs.20260201.t06z.1p00.f000.nc \
    --grid_ref ../data/starviewergraphcast-grid/global_icosahedral_m4.20220101.t00z.1p00.f000.nc \
    --out_dir output_reconstructed

# 5. Verify Metrics and Display Diagnostic Maps
python ../utils/plot_forecast_vs_truth.py \
    --fcst_dir output_reconstructed \
    --level_idx 10 \
    --truth_dir ../data/nc_truth \
    --init_time 2026020100 \
    --lead_times 480 960 1440 1920 2400 \
    --out_dir verification_plots \
    --vars t p u v q --show

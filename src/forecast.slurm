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

svwdir=/scratch5/purged/Wei.Huang/src/starviewerweathermodel
datadir=${svwdir}/data
griddir=${datadir}/terrain-regular-grid
truthdir=${datadir}/icosahedral-truth

cd /scratch5/purged/Wei.Huang/src/graphcast/src

# 1. Clean existing checkpoints and output caches
rm -rf output/*.nc output_reconstructed/*.nc

# 2. Launch Retraining on Full Zarr Targets
# sbatch training.slurm

# 3. After training completes, run standard direct rollout
# Example Rollout using separate t0 (06Z) and t-1 (00Z) single-step NetCDF files
PYTHONPATH=. python rollout/forecast.py \
    -c checkpoints/last.ckpt \
    --input_file_t0 ${truthdir}/icosahedral_logstate_m6.20260201.t06z.0p25.f000.nc \
    --input_file_tm1 ${truthdir}/icosahedral_logstate_m6.20260201.t00z.0p25.f000.nc \
    --init_time 2026020106 \
    -s 20 \
    --step_hours 6 \
    --output_dir output \
    --device cpu

#   --alpha_wind 2.2 \
#   --alpha_w 2.5 \
# 4. Cartesian 3D Spherical Regridder
PYTHONPATH=. python utils/reconstruct_forecast_latlon.py \
    --fcst_dir output \
    --grid_ref ${truthdir}/icosahedral_logstate_m6.20260201.t06z.0p25.f000.nc \
    --truth_ref ${griddir}/gfs.20260201.t06z.0p25.f000.nc \
    --out_dir output_reconstructed

# 5. Verify Metrics and Display Diagnostic Maps
python utils/plot_forecast_vs_truth.py \
    --fcst_dir output_reconstructed \
    --level_idx 10 \
    --truth_dir ${griddir} \
    --init_time 2026020100 \
    --lead_times 24 48 72 96 120 \
    --out_dir verification_plots \
    --vars t p u v q

#!/bin/bash

set -x

python utils/generate_verification_metrics_and_plots.py \
    --fcst_dir output/20260201/06 \
    --truth_dir /scratch5/purged/Wei.Huang/src/starviewerweathermodel/data/icosahedral-truth \
    --init_time 2026020106 \
    --level_idx 10 \
    --out_dir verification_results \
    --show

#   --lead_hours "list(range(6, 726, 6))" \

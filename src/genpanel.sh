#!/bin/bash

set -x

python utils/plot_forecast_panel_comparison.py \
  -f output/forecast_standard_f0024h.nc \
  -t /scratch5/purged/Wei.Huang/src/starviewerweathermodel/data/icosahedral-truth/icosahedral_logstate_m6.20260202.t00z.0p25.f000.nc \
  -l 10 \
  -o forecast_panel.png


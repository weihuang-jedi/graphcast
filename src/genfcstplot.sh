#!/bin/bash

set -x

python utils/plot_icosahedral_panel.py \
  -i output/forecast_standard_f0024h.nc \
  -l 10 -s \
  -o forecast_f024_level_11.png


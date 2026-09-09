#!/bin/bash

set -x

init_date=20260201
init_hour=06
fcst_hour=024
plot_level=10

python utils/plot_icosahedral_panel.py \
  -i output/${init_date}/${init_hour}/forecast_standard_f0${fcst_hour}h.nc \
  -l ${plot_level} -s \
  -o forecast_f0${fcst_hour}_level_${plot_level}.png


### This script runs konrad with interactive ozone and line-by-line calculations at each step.

import sys,os,os.path
os.environ['KONRAD_LOOKUP_TABLE_LW']=path/to/lw/lut
os.environ['KONRAD_LOOKUP_TABLE_SW']=path/to/sw/lut
os.environ['ARTS_DATA_PATH'] = path/to/arts_cat_data
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light as c
import xarray as xr
import seaborn as sns
from typhon import plots

import konrad


# define surface temperatures for which to run the simulation
range_T = np.array([275, 285, 295, 305])

# temperature loop
for T_idx, Ts in enumerate(range_T):
    phlev = konrad.utils.get_quadratic_pgrid(top_pressure=5, num=64)

    if Ts == 305:
        phlev = konrad.utils.get_quadratic_pgrid(top_pressure=5, num=128)

    arts_atmosphere = konrad.atmosphere.Atmosphere(phlev)

    arts = konrad.RCE(
        arts_atmosphere,
        surface=konrad.surface.FixedTemperature(temperature=Ts),  # Run with a fixed surface temperature.
        ozone=konrad.ozone.Cariolle(), # interactive ozone
        timestep='2h',  # Set timestep in model time.
        max_duration='150d',  # Set maximum runtime.
        radiation = konrad.radiation.ARTS() # run ARTS online
    )
    arts.run()

    nc = konrad.netcdf.NetcdfHandler('arts_output_' + str(Ts) + '_ozone_plev.nc', arts)  # create output file
    nc.write()  # write (append) current RCE state to file'''

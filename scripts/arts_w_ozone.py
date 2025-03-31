import sys,os,os.path
os.environ['KONRAD_LOOKUP_TABLE_LW']='/home/pc2943/konrad23/abs_lookup_lw.xml'
os.environ['KONRAD_LOOKUP_TABLE_SW']='/home/pc2943/konrad23/abs_lookup_sw.xml'
os.environ['ARTS_DATA_PATH'] = '/home/pc2943/arts-cat-data/'
import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import speed_of_light as c
import xarray as xr
import seaborn as sns
from typhon import plots

import konrad

phlev = konrad.utils.get_quadratic_pgrid(top_pressure=5, num=64)

range_T = np.array([285, 295, 275])

for T_idx, Ts in enumerate(range_T):
    arts_atmosphere = konrad.atmosphere.Atmosphere(phlev)

    arts = konrad.RCE(
        arts_atmosphere,
        surface=konrad.surface.FixedTemperature(temperature=Ts),  # Run with a fixed surface temperature.
        ozone=konrad.ozone.Cariolle(),
        timestep='2h',  # Set timestep in model time.
        max_duration='150d',  # Set maximum runtime.
        radiation = konrad.radiation.ARTS()
    )
    arts.run()

    nc = konrad.netcdf.NetcdfHandler('arts_output' + str(Ts) + '_ozone.nc', arts)  # create output file
    nc.write()  # write (append) current RCE state to file'''

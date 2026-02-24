### This script computes the error metrics of DDQ compared to ARTS line-by-line output, from file
## both reference ARTS files and DDQ predictions are generated as in the Tutorial, with the
## FluxSimulator module

import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.constants import speed_of_light as c
import xarray as xr
import seaborn as sns

import aux_funcs as aux


filenames = [] ## reference results filenames for present-day, if saved across multiple files

present = xr.open_mfdataset(filenames, engine = "netcdf4", combine = 'nested', concat_dim = 'column').compute()

### example of file structure expected from reference data 

#present = xr.Dataset(
#    data_vars = dict(
#        spectral_fluxes = (["half_level", "column", "spectral_coord"], spectral_fluxes), # spectral fluxes, in W/m^2/cm^-1
#        reference_fluxes = (["half_level", "column"], reference_fluxes), # integrated fluxes, in W/m^2
#        reference_heating = (["column", "level"], reference_heating), # reference heating rate, in K/day
#        pressures = (["column", "half_level"], pressures), # pressure, in Pa or hPa, consistent with heating rate calculation
#    ),
#    coords = dict(
#        half_level = half_level, # half level index
#        level = half_level, # level index
#        column = column, # atmospheric profile index
#        spectral_coord = wavenumber, # wavenumber, in cm^-1
#    ),
#)

### similar procedure for our perturbations from the present day: n x present day gas concentration for
# ch4, n2o, o3, cfc11, cfc12, co2
n_array = np.array([1/8, 1/4, 1/2, 2, 4, 8])

## need to define filenames
# expected structure is as defined above,
# assuming an additional dimension 'n' which contains the gas perturbations from present-day
cfc12 = xr.open_mfdataset(filenames_cfc12, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').compute()
cfc11 = xr.open_mfdataset(filenames_cfc11, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').compute()
co2 = xr.open_mfdataset(filenames_co2, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').compute()
ch4 = xr.open_mfdataset(filenames_ch4, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').compute()
n2o = xr.open_mfdataset(filenames_n2o, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').compute()
o3 = xr.open_mfdataset(filenames_o3, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').compute()


# how many different optimizations are there?
## e.g., in Czarnecki and Brath, we perform 10 independent optimizations, and we want to compute
## error metrics for each one
indeces = np.arange(10)

for idx in indeces:
    ### Present day, training set
    # column computations
    NUM_HL = 55
    present_flux_mean, present_flux_std, present_heat_mean, present_heat_std = aux.profile_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename', np.array([64]), np.array([idx]), present)

    # set arrays for all perturbations
    ch4_flux_means = np.empty((len(n_array), NUM_HL))
    ch4_flux_stds = np.empty((len(n_array), NUM_HL))
    ch4_heat_means = np.empty((len(n_array), NUM_HL))
    ch4_heat_stds = np.empty((len(n_array), NUM_HL))

    n2o_flux_means = np.empty((len(n_array), NUM_HL))
    n2o_flux_stds = np.empty((len(n_array), NUM_HL))
    n2o_heat_means = np.empty((len(n_array), NUM_HL))
    n2o_heat_stds = np.empty((len(n_array), NUM_HL))

    o3_flux_means = np.empty((len(n_array), NUM_HL))
    o3_flux_stds = np.empty((len(n_array), NUM_HL))
    o3_heat_means = np.empty((len(n_array), NUM_HL))
    o3_heat_stds = np.empty((len(n_array), NUM_HL))

    cfc11_flux_means = np.empty((len(n_array), NUM_HL))
    cfc11_flux_stds = np.empty((len(n_array), NUM_HL))
    cfc11_heat_means = np.empty((len(n_array), NUM_HL))
    cfc11_heat_stds = np.empty((len(n_array), NUM_HL))

    cfc12_flux_means = np.empty((len(n_array), NUM_HL))
    cfc12_flux_stds = np.empty((len(n_array), NUM_HL))
    cfc12_heat_means = np.empty((len(n_array), NUM_HL))
    cfc12_heat_stds = np.empty((len(n_array), NUM_HL))

    co2_flux_means = np.empty((len(n_array), NUM_HL))
    co2_flux_stds = np.empty((len(n_array), NUM_HL))
    co2_heat_means = np.empty((len(n_array), NUM_HL))
    co2_heat_stds = np.empty((len(n_array), NUM_HL))


    # loop through scenarios to do these computations

    ### flux and heating rate computations
    for n_idx, n in enumerate(n_array):
        ch4_flux_means[n_idx, :], ch4_flux_stds[n_idx, :], ch4_heat_means[n_idx, :], ch4_heat_stds[n_idx, :] = aux.profile_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename', np.array([64]), np.array([idx]), ch4.isel(n = n_idx))
        
        n2o_flux_means[n_idx, :], n2o_flux_stds[n_idx, :], n2o_heat_means[n_idx, :], n2o_heat_stds[n_idx, :] = aux.profile_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename', np.array([64]), np.array([idx]), n2o.isel(n = n_idx))
        
        o3_flux_means[n_idx, :], o3_flux_stds[n_idx, :], o3_heat_means[n_idx, :], o3_heat_stds[n_idx, :] = aux.profile_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename', np.array([64]), np.array([idx]), o3.isel(n = n_idx))
        
        cfc11_flux_means[n_idx, :], cfc11_flux_stds[n_idx, :], cfc11_heat_means[n_idx, :], cfc11_heat_stds[n_idx, :] = aux.profile_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename', np.array([64]), np.array([idx]), cfc11.isel(n = n_idx))

        cfc12_flux_means[n_idx, :], cfc12_flux_stds[n_idx, :], cfc12_heat_means[n_idx, :], cfc12_heat_stds[n_idx, :] = aux.profile_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename', np.array([64]), np.array([idx]), cfc12.isel(n = n_idx))
        
        co2_flux_means[n_idx, :], co2_flux_stds[n_idx, :], co2_heat_means[n_idx, :], co2_heat_stds[n_idx, :] = aux.profile_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename', np.array([64]), np.array([idx]), co2.isel(n = n_idx))

    # ch4, n2o, o3, cfc11, cfc12, co2
    n_array = np.array([1/8, 1/4, 1/2, 2, 4, 8])

    # set arrays for all perturbations
    ch4_forcing_means = np.empty((len(n_array)))
    ch4_forcing_stds = np.empty((len(n_array)))

    n2o_forcing_means = np.empty((len(n_array)))
    n2o_forcing_stds = np.empty((len(n_array)))

    o3_forcing_means = np.empty((len(n_array)))
    o3_forcing_stds = np.empty((len(n_array)))

    cfc11_forcing_means = np.empty((len(n_array)))
    cfc11_forcing_stds = np.empty((len(n_array)))

    cfc12_forcing_means = np.empty((len(n_array)))
    cfc12_forcing_stds = np.empty((len(n_array)))

    co2_forcing_means = np.empty((len(n_array)))
    co2_forcing_stds = np.empty((len(n_array)))

    ### forcing calculations
    for n_idx, n in enumerate(n_array):
        ch4_forcing_means[n_idx], ch4_forcing_stds[n_idx] = aux.forcing_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename',  np.array([64]), np.array([idx]), present, 
                                                                ch4.isel(n = n_idx, half_level = 0).spectral_fluxes, ch4.isel(n = n_idx, half_level = 0).reference_fluxes)

        n2o_forcing_means[n_idx], n2o_forcing_stds[n_idx] = aux.forcing_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename',  np.array([64]), np.array([idx]), present, 
                                                                n2o.isel(n = n_idx, half_level = 0).spectral_fluxes, n2o.isel(n = n_idx, half_level = 0).reference_fluxes)

        o3_forcing_means[n_idx], o3_forcing_stds[n_idx] = aux.forcing_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename', np.array([64]), np.array([idx]), present, 
                                                                o3.isel(n = n_idx, half_level = 0).spectral_fluxes, o3.isel(n = n_idx, half_level = 0).reference_fluxes)

        cfc11_forcing_means[n_idx], cfc11_forcing_stds[n_idx] = aux.forcing_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename',  np.array([64]), np.array([idx]), present, 
                                                                cfc11.isel(n = n_idx, half_level = 0).spectral_fluxes, cfc11.isel(n = n_idx, half_level = 0).reference_fluxes)

        cfc12_forcing_means[n_idx], cfc12_forcing_stds[n_idx] = aux.forcing_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename',  np.array([64]), np.array([idx]), present, 
                                                                cfc12.isel(n = n_idx, half_level = 0).spectral_fluxes, cfc12.isel(n = n_idx, half_level = 0).reference_fluxes)

        co2_forcing_means[n_idx], co2_forcing_stds[n_idx] = aux.forcing_computations_rrms('dir/to/DDQ/configs', 
                                                                'ddq_filename',  np.array([64]), np.array([idx]), present, 
                                                                co2.isel(n = n_idx, half_level = 0).spectral_fluxes, co2.isel(n = n_idx, half_level = 0).reference_fluxes)


    ### data structure expected for plotting
    rrms_errors = xr.Dataset(
        data_vars = dict(
            ch4_flux_means = (["n","half_level"], ch4_flux_means),
            ch4_heat_means = (["n","half_level"], ch4_heat_means),
            n2o_flux_means = (["n","half_level"], n2o_flux_means),
            n2o_heat_means = (["n","half_level"], n2o_heat_means),
            o3_flux_means = (["n","half_level"], o3_flux_means),
            o3_heat_means = (["n","half_level"], o3_heat_means),
            cfc11_flux_means = (["n","half_level"], cfc11_flux_means),
            cfc11_heat_means = (["n","half_level"], cfc11_heat_means),
            cfc12_flux_means = (["n","half_level"], cfc12_flux_means),
            cfc12_heat_means = (["n","half_level"], cfc12_heat_means),
            co2_flux_means = (["n","half_level"], co2_flux_means),
            co2_heat_means = (["n","half_level"], co2_heat_means),
            ch4_forcing_means = (["n"], ch4_forcing_means),
            n2o_forcing_means = (["n"], n2o_forcing_means),
            o3_forcing_means = (["n"], o3_forcing_means),
            cfc11_forcing_means = (["n"], cfc11_forcing_means),
            cfc12_forcing_means = (["n"], cfc12_forcing_means),
            co2_forcing_means = (["n"], co2_forcing_means),
            present_flux_means = (["half_level"], present_flux_mean[0]),
            present_heat_means = (["half_level"], present_heat_mean[0]),
            ),
        coords = dict(
            half_level = np.arange(55),
            n = np.array([1/8, 1/4, 1/2, 2, 4, 8]),
        ),
    )

    rrms_errors.to_netcdf('savename.nc')
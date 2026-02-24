### This script computes the error metrics of RRTMGP compared to ARTS line-by-line output
# RRTMGP results are generated using the pyRTE-RRTMGP Python wrapper for RRTMGP

import numpy as np

from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.data_types import (
    GasOpticsFiles,
    OpticsProblemTypes,
    RFMIPExampleFiles,
)
from pyrte_rrtmgp.rte_solver import rte_solve
from pyrte_rrtmgp.utils import load_rrtmgp_file

import xarray as xr

import aux_funcs as aux

### note that RRTMGP will produce/expect an atmosphere with the TOA at the first index
# this is the opposite of ARTS
ref = xr.open_mfdataset('path/to/reference_data')

## convert CKDMIP format data to RRTMGP format data
mults = [1, 1, 1, 1] ## multiples of [ch4, n2o, o3, co2] concentration; here, present-day values
cos_sza = np.array([0.1, 0.3, 0.52, 0.7, 0.9]) # cosie of the solar zenith angle
albedo = np.array([0.15, 0.30, 0.45, 0.6, 0.75])
concs = aux.concs_to_rrtmgp(mults, cos_sza, albedo)

present_flux_errs_lw = np.zeros((25, 2, 53))
present_heat_errs_lw = np.zeros((25, 2, 53))
present_fluxes_lw = np.zeros((25, 49, 53))

present_flux_errs_sw = np.zeros((25, 2, 53))
present_heat_errs_sw = np.zeros((25, 2, 53))
present_fluxes_sw = np.zeros((25, 49, 53))


# a "scenario" here is a combination of zenith angle and albedo; we sample 5 of each
for scenario_idx in np.arange(25):
    scenario_cols_ref = ref.column.data[scenario_idx:][::25]

    present_fluxes_lw[scenario_idx, :, :], _, present_flux_errs_lw[scenario_idx, :, :], present_heat_errs_lw[scenario_idx, :, :] = aux.compute_rrtmgp_quantities_lw(concs.isel(column = scenario_cols_ref), 
                                                                                                                                                        ref.isel(column = scenario_cols_ref))
    present_fluxes_sw[scenario_idx, :, :], _, present_flux_errs_sw[scenario_idx, :, :], present_heat_errs_sw[scenario_idx, :, :] = aux.compute_rrtmgp_quantities_sw(concs.isel(column = scenario_cols_ref), 
                                                                                                                                                        ref.isel(column = scenario_cols_ref))

### save to file
data = xr.Dataset(
    data_vars = dict(
        flux_errs = (["scenario", "idx", "half_level"], present_flux_errs_sw),
        heat_errs = (["scenario", "idx", "half_level"], present_heat_errs_sw),
        ),
    coords = dict(
        half_level = np.arange(53),
        scenario = np.arange(25),
        idx = np.arange(2),
    ),
)

data.to_netcdf(save/to/filename)

data = xr.Dataset(
    data_vars = dict(
        flux_errs = (["scenario", "idx", "half_level"], present_flux_errs_lw),
        heat_errs = (["scenario", "idx", "half_level"], present_heat_errs_lw),
        ),
    coords = dict(
        half_level = np.arange(53),
        scenario = np.arange(25),
        idx = np.arange(2),
    ),
)

data.to_netcdf(save/to/filename)



### calculate fluxes, heating rates, and forcing in different climates
n_array = np.array([1/8, 1/4, 1/2, int(2), int(4), int(8)]) ## perturbations

## we use only 10 columns of CKDMIP dataset for each perturbation, to avoid having to compute
# n_sza * n_albedos * 50 * n_perturbations * n_gases line-by-line profiles.
col_indeces = [np.arange(40, 50), np.arange(10), np.array([10, 11, 12, 13, 15, 16, 17, 18, 19]), np.arange(30, 40), np.arange(20, 30), np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])]

for n_idx, n in enumerate(n_array):
    mults = np.array([[n, 1, 1, 1, 1, 1], [1, n, 1, 1, 1, 1], [1, 1, n, 1, 1, 1], [1, 1, 1, n, 1, 1], [1, 1, 1, 1, n, 1], [1, 1, 1, 1, 1, n]])
    gas_label = ['ch4', 'n2o', 'o3', 'co2', 'cfc11', 'cfc12']
    idx = col_indeces[n_idx]

    for gas_idx in range(mults.shape[0]):
        concs = aux.concs_to_rrtmgp(mults, cos_sza, albedo)
        ref_perturbed = xr.open_mfdataset('path/to/reference_data') ## assuming perturbed filenames are saved in individual files


        flux_errs_lw = np.zeros((25, 2, 53))
        heat_errs_lw = np.zeros((25, 2, 53))
        forcing_errs_lw = np.zeros((25, 2))

        flux_errs_sw = np.zeros((25, 2, 53))
        heat_errs_sw = np.zeros((25, 2, 53))
        forcing_errs_sw = np.zeros((25, 2))

        for scenario_idx in np.arange(25):
            scenario_cols_ref = ref.column.data[scenario_idx:][::25][idx]
            scenario_cols_pert = ref.column.data[scenario_idx:][::25]

            ### LW
            flux, _, flux_errs_lw[scenario_idx, :, :], heat_errs_lw[scenario_idx, :, :] = aux.compute_rrtmgp_quantities_lw(concs.isel(column = scenario_cols_ref), ref_perturbed.isel(column = scenario_cols_pert))
            forcing_est = present_fluxes_lw[scenario_idx, idx, 0] - flux[:, 0]
            forcing_ref = ref.reference_fluxes.isel(column = scenario_cols_ref, half_level = 0).data - ref_perturbed.reference_fluxes.isel(column = scenario_cols_pert, half_level = 0).data
            forcing_errs_lw[scenario_idx, :] = aux.rel_rms(np.array([forcing_est]), np.array([forcing_ref]))

            ### SW
            flux, _, flux_errs_sw[scenario_idx, :, :], heat_errs_sw[scenario_idx, :, :] = aux.compute_rrtmgp_quantities_sw(concs.isel(column = scenario_cols_ref), ref_perturbed.isel(column = scenario_cols_pert))
            forcing_est = present_fluxes_sw[scenario_idx, idx, 0] - flux[:, 0]
            forcing_ref = ref.reference_fluxes.isel(column = scenario_cols_ref, half_level = 0).data - ref_perturbed.reference_fluxes.isel(column = scenario_cols_pert, half_level = 0).data
            forcing_errs_sw[scenario_idx, :] = aux.rel_rms(np.array([forcing_est]), np.array([forcing_ref]))


        data = xr.Dataset(
            data_vars = dict(
                flux_errs = (["scenario", "idx", "half_level"], flux_errs_lw),
                heat_errs = (["scenario", "idx", "half_level"], heat_errs_lw),
                forcing_errs = (["scenario", "idx"], forcing_errs_lw),
                ),
            coords = dict(
                half_level = np.arange(53),
                scenario = np.arange(25),
                idx = np.arange(2),
            ),
        )
        data.to_netcdf(lw/savename.nc)

        data = xr.Dataset(
            data_vars = dict(
                flux_errs = (["scenario", "idx", "half_level"], flux_errs_sw),
                heat_errs = (["scenario", "idx", "half_level"], heat_errs_sw),
                forcing_errs = (["scenario", "idx"], forcing_errs_sw),
                ),
            coords = dict(
                half_level = np.arange(53),
                scenario = np.arange(25),
                idx = np.arange(2),
            ),
        )
        data.to_netcdf(sw/savename.nc)
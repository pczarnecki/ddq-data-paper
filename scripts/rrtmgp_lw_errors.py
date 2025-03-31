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

import matplotlib.pyplot as plt

def rel_rms(estimate, reference):
    # relative root mean squared error across all ensembles
    # Buehler 2010 eqn 3
    ## Introduce floor for heating
    
    return np.sqrt((((estimate - reference)/np.maximum(np.abs(reference), 10**(-1)))**2).mean(axis = 1))


def compute_rrtmgp_quantities(atm, ref):
    internal_atm = atm.copy()

    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    gas_mapping = {
        "h2o": "h2o_mole_fraction_fl",
        "co2": "co2_mole_fraction_fl",
        "o3": "o3_mole_fraction_fl",
        "co": "co_mole_fraction_fl",
        "n2o": "n2o_mole_fraction_fl",
        "ch4": "ch4_mole_fraction_fl",
        "o2": "o2_mole_fraction_fl",
        "n2": "n2_mole_fraction_fl",
        "cfc11": "cfc11_mole_fraction_fl",
        "cfc12": "cfc12_mole_fraction_fl",
    }

    gas_optics_lw.compute_gas_optics(
        internal_atm,
        problem_type=OpticsProblemTypes.ABSORPTION,
        gas_name_map=gas_mapping,
    )

    fluxes = rte_solve(internal_atm, add_to_input=False)
    net_flux = fluxes.lw_flux_down - fluxes.lw_flux_up

    g = 9.81
    scaling = 3600*24
    heating = -scaling*(np.gradient((net_flux), axis = -1)*g/np.gradient(internal_atm.pres_level.data, axis = -1)/1004)

    flux_rrms_rrtmgp = rel_rms(net_flux.data.T, ref.reference_fluxes.data)

    heat_rrms_rrtmgp = rel_rms(heating.T, ref.reference_heating.data.T)

    return net_flux, heating, flux_rrms_rrtmgp, heat_rrms_rrtmgp



concs = xr.open_dataset('/data/robertp/CKDMIP_LBL/evaluation2/conc/ckdmip_evaluation2_concentrations_present.nc', engine = "netcdf4")
concs = concs.drop_isel(column = 14)


def define_reference_file(filenames):
    pyarts_data = xr.open_mfdataset(filenames, engine = "netcdf4", combine = 'nested', concat_dim = 'column').isel(half_level = np.arange(2, 55)).compute()
    #pyarts_data = pyarts_data.drop_sel(column = 14)

    # transpose into appropriate shape for spectral and reference fluxes, open files
    spectral_fluxes = pyarts_data.spec_net_flux.transpose('half_level', 'column', 'wavenumber').compute()
    reference_fluxes = pyarts_data.int_net_flux.compute()

    # include reference heating rate and pressure data
    reps = len(pyarts_data.column.data)/49
    pressures = np.repeat(col_concs.pres_level.data, reps, axis = 0)
    g = 9.81 # gravity (m/s^2)
    scaling = 3600*24 # heating rate conversion factor: seconds * minutes * hours in a day 

    reference_heating = -scaling*(np.gradient(reference_fluxes.T, axis = 0)*g/np.gradient(pressures.T, axis = 0)/1004)

    ### compose into xarray
    data_eval2_LW_present = xr.Dataset(
        data_vars = dict(
            spectral_fluxes = (["half_level", "column", "spectral_coord"], spectral_fluxes.data),
            reference_fluxes = (["half_level", "column"], reference_fluxes.T.data),
            reference_heating = (["column", "level"], reference_heating.T),
            pressures = (["column", "half_level"], pressures.data),
        ),
        coords = dict(
            half_level = spectral_fluxes.half_level.data,
            level = spectral_fluxes.half_level.data,
            column = spectral_fluxes.column.data,
            spectral_coord = spectral_fluxes.wavenumber.data,
        ),
    )

    return data_eval2_LW_present


## set up default data
col_concs = concs.isel(half_level = np.arange(2, 55), level = np.arange(2, 54)).copy()
col_concs = col_concs.rename({"level":"layer", "half_level":"level"})
col_concs = col_concs.rename({"temperature_hl":"temp_level", "temperature_fl":"temp_layer", "pressure_hl":"pres_level", "pressure_fl":"pres_layer"})
col_concs = col_concs.assign({"surface_temperature":(["column"], col_concs.temp_level.data[:, -1])})
col_concs = col_concs.assign({"co_mole_fraction_fl": (["column", "layer"], 0*np.ones((np.shape(col_concs.h2o_mole_fraction_fl.data)))), "co_mole_fraction_hl": (["column", "level"], 0*np.ones((np.shape(col_concs.h2o_mole_fraction_hl.data))))})
col_concs = col_concs.drop_vars('level')
col_concs = col_concs.drop_vars('layer')

### computations

present_filenames = []
for i in np.arange(50):
    if i != 14:
        present_filenames.append('/data/pc2943/eval2_present_col_' + str(int(i)) + '.h5')

present_ref = define_reference_file(present_filenames)
present_flux, present_heating, present_flux_error, present_heat_error = compute_rrtmgp_quantities(col_concs, present_ref)

NUM_HL = 53

# ch4, n2o, o3, cfc11, cfc12, co2
n_array = np.array([1/8, 1/4, 1/2, 2, 4, 8])

filenames_cfc12 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_cfc12.append('/data/pc2943/eval2_' + str(n) + 'x_cfc12_col_' + str(i) + '.h5')
cfc12_data = define_reference_file(filenames_cfc12)

filenames_cfc11 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_cfc11.append('/data/pc2943/eval2_' + str(n) + 'x_cfc11_col_' + str(i) + '.h5')
cfc11_data = define_reference_file(filenames_cfc11)

filenames_co2 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_co2.append('/data/pc2943/eval2_' + str(n) + 'x_co2_col_' + str(i) + '.h5')
co2_data = define_reference_file(filenames_co2)

filenames_ch4 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_ch4.append('/data/pc2943/eval2_' + str(n) + 'x_ch4_col_' + str(i) + '.h5')
ch4_data = define_reference_file(filenames_ch4)

filenames_n2o = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_n2o.append('/data/pc2943/eval2_' + str(n) + 'x_n2o_col_' + str(i) + '.h5')
n2o_data = define_reference_file(filenames_n2o)

filenames_o3 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_o3.append('/data/pc2943/eval2_' + str(n) + 'x_o3_col_' + str(i) + '.h5')
o3_data = define_reference_file(filenames_o3)

# set arrays for all perturbations
ch4_flux_means = np.empty((len(n_array), NUM_HL))
ch4_heat_means = np.empty((len(n_array), NUM_HL))

n2o_flux_means = np.empty((len(n_array), NUM_HL))
n2o_heat_means = np.empty((len(n_array), NUM_HL))

o3_flux_means = np.empty((len(n_array), NUM_HL))
o3_heat_means = np.empty((len(n_array), NUM_HL))

cfc11_flux_means = np.empty((len(n_array), NUM_HL))
cfc11_heat_means = np.empty((len(n_array), NUM_HL))

cfc12_flux_means = np.empty((len(n_array), NUM_HL))
cfc12_heat_means = np.empty((len(n_array), NUM_HL))

co2_flux_means = np.empty((len(n_array), NUM_HL))
co2_heat_means = np.empty((len(n_array), NUM_HL))

# set arrays for all perturbations
ch4_forcing_means = np.empty((len(n_array)))

n2o_forcing_means = np.empty((len(n_array)))

o3_forcing_means = np.empty((len(n_array)))

cfc11_forcing_means = np.empty((len(n_array)))

cfc12_forcing_means = np.empty((len(n_array)))

co2_forcing_means = np.empty((len(n_array)))


for n_idx, n in enumerate(n_array):
    ## CH4
    col_concs_ch4 = col_concs.copy()
    col_concs_ch4 = col_concs.assign({"ch4_mole_fraction_fl": (["column", "layer"], n*col_concs.ch4_mole_fraction_fl.data), "ch4_mole_fraction_hl": (["column", "level"], n*col_concs.ch4_mole_fraction_hl.data)})

    flux, _, ch4_flux_means[n_idx, :], ch4_heat_means[n_idx, :] = compute_rrtmgp_quantities(col_concs_ch4, ch4_data.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))

    forcing_ch4 = present_flux[:, 0] - flux[:, 0]
    forcing_ref = present_ref.reference_fluxes.isel(half_level = 0).data - ch4_data.reference_fluxes.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).data
    ch4_forcing_means[n_idx] = rel_rms(np.array([forcing_ch4]), np.array([forcing_ref]))

    ## N2O
    col_concs_n2o = col_concs.copy()                                                                             
    col_concs_n2o = col_concs.assign({"n2o_mole_fraction_fl": (["column", "layer"], n*col_concs.n2o_mole_fraction_fl.data), "n2o_mole_fraction_hl": (["column", "level"], n*col_concs.n2o_mole_fraction_hl.data)})

    flux, _, n2o_flux_means[n_idx, :], n2o_heat_means[n_idx, :] = compute_rrtmgp_quantities(col_concs_n2o, n2o_data.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))

    forcing_n2o = present_flux[:, 0] - flux[:, 0]
    forcing_ref = present_ref.reference_fluxes.isel(half_level = 0).data - n2o_data.reference_fluxes.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).data
    n2o_forcing_means[n_idx] = rel_rms(np.array([forcing_n2o]), np.array([forcing_ref]))

    ## O3
    col_concs_o3 = col_concs.copy()                                                                               
    col_concs_o3 = col_concs.assign({"o3_mole_fraction_fl": (["column", "layer"], n*col_concs.o3_mole_fraction_fl.data), "o3_mole_fraction_hl": (["column", "level"], n*col_concs.o3_mole_fraction_hl.data)})

    flux, _, o3_flux_means[n_idx, :], o3_heat_means[n_idx, :] = compute_rrtmgp_quantities(col_concs_o3, o3_data.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))

    forcing_o3 = present_flux[:, 0] - flux[:, 0]
    forcing_ref = present_ref.reference_fluxes.isel(half_level = 0).data - o3_data.reference_fluxes.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).data
    o3_forcing_means[n_idx] = rel_rms(np.array([forcing_o3]), np.array([forcing_ref]))

    ## CFC11
    col_concs_cfc11 = col_concs.copy()                                                                          
    col_concs_cfc11 = col_concs.assign({"cfc11_mole_fraction_fl": (["column", "layer"], n*col_concs.cfc11_mole_fraction_fl.data), "cfc11_mole_fraction_hl": (["column", "level"], n*col_concs.cfc11_mole_fraction_hl.data)})

    flux, _, cfc11_flux_means[n_idx, :], cfc11_heat_means[n_idx, :] = compute_rrtmgp_quantities(col_concs_cfc11, cfc11_data.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))

    forcing_cfc11 = present_flux[:, 0] - flux[:, 0]
    forcing_ref = present_ref.reference_fluxes.isel(half_level = 0).data - cfc11_data.reference_fluxes.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).data
    cfc11_forcing_means[n_idx] = rel_rms(np.array([forcing_cfc11]), np.array([forcing_ref]))

    ## CFC12
    col_concs_cfc12 = col_concs.copy()                                                               
    col_concs_cfc12 = col_concs.assign({"cfc12_mole_fraction_fl": (["column", "layer"], n*col_concs.cfc12_mole_fraction_fl.data), "cfc12_mole_fraction_hl": (["column", "level"], n*col_concs.cfc12_mole_fraction_hl.data)})

    flux, _, cfc12_flux_means[n_idx, :], cfc12_heat_means[n_idx, :] = compute_rrtmgp_quantities(col_concs_cfc12, cfc12_data.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))

    forcing_cfc12 = present_flux[:, 0] - flux[:, 0]
    forcing_ref = present_ref.reference_fluxes.isel(half_level = 0).data - cfc12_data.reference_fluxes.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).data
    cfc12_forcing_means[n_idx] = rel_rms(np.array([forcing_cfc12]), np.array([forcing_ref]))

    ## CO2
    col_concs_co2 = col_concs.copy()                                                       
    col_concs_co2 = col_concs.assign({"co2_mole_fraction_fl": (["column", "layer"], n*col_concs.co2_mole_fraction_fl.data), "co2_mole_fraction_hl": (["column", "level"], n*col_concs.co2_mole_fraction_hl.data)})

    flux, _, co2_flux_means[n_idx, :], co2_heat_means[n_idx, :] = compute_rrtmgp_quantities(col_concs_co2, co2_data.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))

    forcing_co2 = present_flux[:, 0] - flux[:, 0]
    forcing_ref = present_ref.reference_fluxes.isel(half_level = 0).data - co2_data.reference_fluxes.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).data
    co2_forcing_means[n_idx] = rel_rms(np.array([forcing_co2]), np.array([forcing_ref]))


rrms_errors = xr.Dataset(
    data_vars = dict(
        present_flux_error = (["half_level"], present_flux_error),
        present_heat_error = (["half_level"], present_heat_error),
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

        ),
    coords = dict(
        half_level = np.arange(53),
        n = np.array([1/8, 1/4, 1/2, 2, 4, 8]),
    ),
)

rrms_errors.to_netcdf('all_forcing_lw_rrms_rrtmgp.h5')
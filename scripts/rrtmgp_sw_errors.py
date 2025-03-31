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

def rel_rms(estimate, reference):
    # relative root mean squared error across all ensembles
    # Buehler 2010 eqn 3
    ## Introduce floor for heating
    
    return np.sqrt((((estimate - reference)/np.maximum(np.abs(reference), 10**(-1)))**2).mean(axis = 1))

def compute_rrtmgp_quantities(atm, ref):
    internal_atm = atm.copy()

    gas_optics_sw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.SW_G224
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

    gas_optics_sw.compute_gas_optics(
        internal_atm,
        problem_type=OpticsProblemTypes.TWO_STREAM,
        gas_name_map=gas_mapping,
    )


    fluxes = rte_solve(internal_atm, add_to_input=False)

    net_flux = fluxes.sw_flux_down - fluxes.sw_flux_up

    g = 9.81
    scaling = 3600*24
    heating = -scaling*(np.gradient((net_flux), axis = -1)*g/np.gradient(internal_atm.pres_level.data, axis = -1)/1004)

    flux_rrms_rrtmgp = rel_rms(net_flux.data.T, ref.reference_fluxes.data)

    heat_rrms_rrtmgp = rel_rms(heating.T, ref.reference_heating.data.T)

    return net_flux, heating, flux_rrms_rrtmgp, heat_rrms_rrtmgp


def define_reference_file(filenames):
    perturbed_data = xr.open_mfdataset(filenames, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').compute()

    if len(perturbed_data.half_level) > 53:
    
        perturbed_data_order = xr.Dataset(
            data_vars = dict(
                #spectral_fluxes = (["half_level", "column", "spectral_coord"], perturbed_data.spectral_fluxes.transpose('half_level', 'column', 'spectral_coord').data[2:, :, :]),
                reference_fluxes = (["half_level", "column"], perturbed_data.reference_fluxes.data.T[::-1, :][2:, :]),
                #reference_heating = (["column", "half_level"], perturbed_data.reference_heating[::-1, :].data[:, 2:]),
                pressures = (["column", "half_level"], perturbed_data.pressures[:, ::-1].data[:, 2:]),

                ),
            coords = dict(
                half_level = np.arange(53),
                level = np.arange(53),
                column = perturbed_data.column.data,
                #spectral_coord = perturbed_data.spectral_coord.data,
            ),
        )

        g = 9.81
        scaling = 3600*24
        reference_heating = -scaling*(np.gradient((perturbed_data_order.reference_fluxes.T), axis = -1)*g/np.gradient(perturbed_data_order.pressures.data, axis = -1)/1004)
        perturbed_data_order= perturbed_data_order.assign({"reference_heating":(["column", "half_level"], reference_heating)})


    else:
        perturbed_data_order = xr.Dataset(
            data_vars = dict(
                #spectral_fluxes = (["half_level", "column", "spectral_coord"], perturbed_data.spectral_fluxes.transpose('half_level', 'column', 'spectral_coord').data),
                reference_fluxes = (["half_level", "column"], perturbed_data.reference_fluxes.data.T[::-1, :]),
                #reference_heating = (["column", "half_level"], perturbed_data.reference_heating[::-1, :].data),
                pressures = (["column", "half_level"], perturbed_data.pressures[:, ::-1].data),

                ),
            coords = dict(
                half_level = np.arange(53),
                level = np.arange(53),
                column = perturbed_data.column.data,
                #spectral_coord = perturbed_data.spectral_coord.data,
            ),
        )

        g = 9.81
        scaling = 3600*24
        reference_heating = -scaling*(np.gradient((perturbed_data_order.reference_fluxes.T), axis = -1)*g/np.gradient(perturbed_data_order.pressures.data, axis = -1)/1004)
        perturbed_data_order = perturbed_data_order.assign({"reference_heating":(["column", "half_level"], reference_heating)})

    return perturbed_data_order

# ch4, n2o, o3, cfc11, cfc12, co2
n_array = np.array([1/8, 1/4, 1/2, int(2), int(4), int(8)])
col_indeces = [np.arange(40, 50), np.arange(10), np.array([10, 11, 12, 13, 15, 16, 17, 18, 19]), np.arange(30, 40), np.arange(20, 30), np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])]
present_indeces = [np.arange(39, 49), np.arange(10), np.arange(10, 19), np.arange(29, 39), np.arange(19, 29), np.array([0, 5, 10, 14, 19, 24, 29, 34, 39, 44])]

concs = xr.open_dataset('/data/robertp/CKDMIP_LBL/evaluation2/conc/ckdmip_evaluation2_concentrations_present.nc', engine = "netcdf4")
concs = concs.drop_isel(column = 14)

## set up default data
col_concs = concs.isel(half_level = np.arange(2, 55), level = np.arange(2, 54)).copy()
col_concs = col_concs.rename({"level":"layer", "half_level":"level"})
col_concs = col_concs.rename({"temperature_hl":"temp_level", "temperature_fl":"temp_layer", "pressure_hl":"pres_level", "pressure_fl":"pres_layer"})
col_concs = col_concs.assign({"surface_temperature":(["column"], col_concs.temp_level.data[:, -1])})
col_concs = col_concs.assign({"co_mole_fraction_fl": (["column", "layer"], 0*np.ones((np.shape(col_concs.h2o_mole_fraction_fl.data)))), "co_mole_fraction_hl": (["column", "level"], 0*np.ones((np.shape(col_concs.h2o_mole_fraction_hl.data))))})

ref_data = xr.open_dataset('/data/pc2943/eval2_sw_full_sun_mu.h5', engine = "netcdf4")
ref_data_order = xr.Dataset(
    data_vars = dict(
        #spectral_fluxes = (["half_level", "column", "spectral_coord"], ref_data.spectral_fluxes.transpose('half_level', 'column', 'spectral_coord').data),
        reference_fluxes = (["half_level", "column"], ref_data.reference_fluxes.data.T[::-1, :].data),
        pressures = (["column", "half_level"], ref_data.pressures[:, ::-1].data),
        #reference_heating = (["column", "half_level"], ref_data.reference_heating[::-1, :].data),
    ),
    coords = dict(
        half_level = ref_data.half_level.data,
        column = ref_data.column.data,
        #spectral_coord = ref_data.spectral_coord.data,
    ),
)
g = 9.81
scaling = 3600*24
reference_heating = -scaling*(np.gradient((ref_data_order.reference_fluxes.T), axis = -1)*g/np.gradient(ref_data_order.pressures.data, axis = -1)/1004)
ref_data_order = ref_data_order.assign({"reference_heating":(["column", "half_level"], reference_heating)})


ref_data_order = ref_data_order.isel(half_level = np.arange(2, 55)).compute()
new_idx = np.arange(49)
new_idx = new_idx.repeat(25)

col_concs = col_concs.reindex(column = np.arange(49))
col_concs = col_concs.reindex(column = new_idx, method = 'ffill')

col_concs = col_concs.drop_vars('level')
col_concs = col_concs.drop_vars('layer')
col_concs = col_concs.drop_vars('column')


lats = np.array([0.1, 0.3, 0.52, 0.7, 0.9])
albedos = np.array([0.15, 0.30, 0.45, 0.6, 0.75])

present_zenith = np.zeros((len(ref_data_order.column.data)))
present_albedos = np.zeros((len(ref_data_order.column.data)))

j = 0
for i in np.arange(len(concs.column.data)):
    for lat in lats:
        for reflectivity in albedos:
            present_zenith[j] = lat
            present_albedos[j] = reflectivity
            j += 1

col_concs = col_concs.assign({"surface_albedo": (["column"], present_albedos), "mu0": (["column"], present_zenith)})

present_concs = col_concs.copy()

present_flux_errs = np.zeros((25, 2, 53))
present_heat_errs = np.zeros((25, 2, 53))
present_fluxes = np.zeros((25, 49, 53))

for scenario_idx in np.arange(25):
    scenario_cols_ref = ref_data_order.column.data[scenario_idx:][::25]

    present_fluxes[scenario_idx, :, :], _, present_flux_errs[scenario_idx, :, :], present_heat_errs[scenario_idx, :, :] = compute_rrtmgp_quantities(present_concs.isel(column = scenario_cols_ref), ref_data_order.isel(column = scenario_cols_ref))

data = xr.Dataset(
    data_vars = dict(
        flux_errs = (["scenario", "idx", "half_level"], present_flux_errs),
        heat_errs = (["scenario", "idx", "half_level"], present_heat_errs),
        ),
    coords = dict(
        half_level = np.arange(53),
        scenario = np.arange(25),
        idx = np.arange(2),
    ),
)

data.to_netcdf('rel_rms_present_rrtmgp_01.h5')

##################### loop
for n_idx, n in enumerate(n_array):
    mults = np.array([[n, 1, 1, 1], [1, n, 1, 1], [1, 1, n, 1], [1, 1, 1, n]])
    gas_label = ['ch4', 'n2o', 'o3', 'co2']
    idx = col_indeces[n_idx]
    present_idx = present_indeces[n_idx]


    for gas_idx in range(mults.shape[0]):

        filenames = []
        zenith_conc = np.zeros((len(lats)*len(albedos)*len(idx)))
        albedo_conc = np.zeros((len(lats)*len(albedos)*len(idx)))
        j = 0

        for i in idx:
            for lat in lats:
                for reflectivity in albedos:

                    zenith_conc[j] = lat
                    albedo_conc[j] = reflectivity
                    j += 1


                    if n == 8 or n == 4 or n == 2:
                        filenames.append('/data/pc2943/eval2_sw_full_sun_' + str(i) + '_mu_' + str(lat) + '_ref_' + str(reflectivity) +'_' + str(int(n)) + 'x_' + gas_label[gas_idx] + '.h5')

                    else:
                        filenames.append('/data/pc2943/eval2_sw_full_sun_' + str(i) + '_mu_' + str(lat) + '_ref_' + str(reflectivity) +'_' + str(n) + 'x_' + gas_label[gas_idx] + '.h5')

        reference_scenario = define_reference_file(filenames)

        scenario_concs = col_concs.copy()
        ### perturb the scenario
        if gas_label[gas_idx] == 'co2':
            scenario_concs = scenario_concs.assign({"co2_mole_fraction_fl": (["column", "layer"], n*col_concs.co2_mole_fraction_fl.data), "co2_mole_fraction_hl": (["column", "level"], n*col_concs.co2_mole_fraction_hl.data)})

        elif gas_label[gas_idx] == 'ch4':
            scenario_concs = scenario_concs.assign({"ch4_mole_fraction_fl": (["column", "layer"], n*col_concs.ch4_mole_fraction_fl.data), "ch4_mole_fraction_hl": (["column", "level"], n*col_concs.ch4_mole_fraction_hl.data)})

        elif gas_label[gas_idx] == 'n2o':
            scenario_concs = scenario_concs.assign({"n2o_mole_fraction_fl": (["column", "layer"], n*col_concs.n2o_mole_fraction_fl.data), "n2o_mole_fraction_hl": (["column", "level"], n*col_concs.n2o_mole_fraction_hl.data)})
        
        elif gas_label[gas_idx] == 'o3':
            scenario_concs = scenario_concs.assign({"o3_mole_fraction_fl": (["column", "layer"], n*col_concs.o3_mole_fraction_fl.data), "o3_mole_fraction_hl": (["column", "level"], n*col_concs.o3_mole_fraction_hl.data)})

        else:
            print("ERROR")



        #scenario_concs = scenario_concs.assign({"surface_albedo": (["column"], zenith_conc), "mu0": (["column"], albedo_conc)})


        flux_errs = np.zeros((25, 2, 53))
        heat_errs = np.zeros((25, 2, 53))
        forcing_errs = np.zeros((25, 2))

        for scenario_idx in np.arange(25):
            scenario_cols_ref = ref_data_order.column.data[scenario_idx:][::25][present_idx]
            scenario_cols_pert = reference_scenario.column.data[scenario_idx:][::25]

            flux, _, flux_errs[scenario_idx, :, :], heat_errs[scenario_idx, :, :] = compute_rrtmgp_quantities(scenario_concs.isel(column = scenario_cols_ref), reference_scenario.isel(column = scenario_cols_pert))

            forcing_est = present_fluxes[scenario_idx, present_idx, 0] - flux[:, 0]
            forcing_ref = ref_data_order.reference_fluxes.isel(column = scenario_cols_ref, half_level = 0).data - reference_scenario.reference_fluxes.isel(column = scenario_cols_pert, half_level = 0).data
            forcing_errs[scenario_idx, :] = rel_rms(np.array([forcing_est]), np.array([forcing_ref]))


        data = xr.Dataset(
            data_vars = dict(
                flux_errs = (["scenario", "idx", "half_level"], flux_errs),
                heat_errs = (["scenario", "idx", "half_level"], heat_errs),
                forcing_errs = (["scenario", "idx"], forcing_errs),
                ),
            coords = dict(
                half_level = np.arange(53),
                scenario = np.arange(25),
                idx = np.arange(2),
            ),
        )
        data.to_netcdf('rel_rms_' + str(n) + 'x_' + gas_label[gas_idx] + '_rrtmgp_01.h5')
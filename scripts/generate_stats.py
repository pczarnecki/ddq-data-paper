import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.constants import speed_of_light as c
import xarray as xr
import random
import seaborn as sns

# Error metrics

def compute_testing(S, w, intercept, spectral_test, reference_test, num_hl, num_cols):
    spec_data = spectral_test.data
    q_test = np.empty((num_hl, num_cols))
    if num_hl == 1:
        q_test = np.matmul(w, spec_data[:, :].T)
        error = rel_rms(np.array([q_test]), np.array([reference_test]))
    else:
        for i in range(num_hl):
            q_test[i, :] = np.matmul(w, spec_data[i, :, :].T)
        
        # compute error of test estimate against test reference calculation
        error = rel_rms(q_test, reference_test)
    
    return error, q_test


def compute_testing_rrms(S, w, intercept, spectral_test, reference_test, num_hl, num_cols):
    spec_data = spectral_test.data
    q_test = np.empty((num_hl, num_cols))
    if num_hl == 1:
        q_test = np.matmul(w, spec_data[:, :].T)
        error = rel_rms(np.array([q_test]), np.array([reference_test]))
    else:
        for i in range(num_hl):
            q_test[i, :] = np.matmul(w, spec_data[i, :, :].T)
        
        # compute error of test estimate against test reference calculation
        error = rel_rms(q_test, reference_test)
    
    return error, q_test


def rel_rms(estimate, reference):
    # relative root mean squared error across all ensembles
    # Buehler 2010 eqn 3
    ## Introduce floor for heating
    
    return np.sqrt((((estimate - reference)/np.maximum(np.abs(reference), 10**(-1)))**2).mean(axis = 1))


def profile_computations_rrms(result_directory, result_filename, subset_sizes, num_reps, ref_data):
    g = 9.81
    scaling = 3600*24
    PRESSURES = ref_data.pressures.data

    reference_heating = -scaling*(np.gradient((ref_data.reference_fluxes.transpose('half_level', 'column').T), axis = -1)*g/np.gradient(PRESSURES, axis = -1)/1004)
    
    # set up empty result arrays

    flux_train_arms_mean = np.zeros((len(subset_sizes), len(ref_data.half_level.data)))
    flux_train_arms_std = np.zeros((len(subset_sizes), len(ref_data.half_level.data)))
    
    heat_train_arms_mean = np.zeros((len(subset_sizes), len(ref_data.half_level.data)))
    heat_train_arms_std = np.zeros((len(subset_sizes), len(ref_data.half_level.data)))
        
    for i in range(len(subset_sizes)):
        
        # set up empty result arrays
        flux_train_arms = np.zeros((len(num_reps), len(ref_data.half_level.data)))
        heat_train_arms = np.zeros((len(num_reps), len(ref_data.half_level.data)))
                
        for j in range(len(num_reps)):
            # open subset and result datasets
            results = xr.open_dataset(result_directory + str(subset_sizes[i]) + result_filename + str(num_reps[j]) + '.h5', engine = "netcdf4")
            subsets = ref_data.isel(spectral_coord = results.S.data)
                        
            # compute flux error            
            flux_train_arms[j, :], total_fluxes_train = compute_testing_rrms(results.S.data, results.W.data, 0, subsets.spectral_fluxes.transpose('half_level', 'column', 'spectral_coord'), ref_data.reference_fluxes.transpose('half_level', 'column'), len(ref_data.half_level.data), len(ref_data.column.data))
            
            train_heating = -scaling*(np.gradient((total_fluxes_train.T), axis = -1)*g/np.gradient(PRESSURES, axis = -1)/1004)

            heat_train_arms[j, :] = rel_rms(train_heating.T, reference_heating.T)
           
        # take mean and std across repetition
        flux_train_arms_mean[i, :] = np.mean(flux_train_arms, axis = 0)
        flux_train_arms_std[i, :] = np.std(flux_train_arms, axis = 0)

        heat_train_arms_mean[i, :] = np.mean(heat_train_arms, axis = 0)
        heat_train_arms_std[i, :] = np.std(heat_train_arms, axis = 0)
                            
    return flux_train_arms_mean, flux_train_arms_std, heat_train_arms_mean, heat_train_arms_std

# absolute RMS for figures

def forcing_computations_rrms(result_directory, result_filename, subset_sizes, num_reps, ref_data, spectral_perturbed_OLR, reference_perturbed_OLR):
    g = 9.81
    scaling = 3600*24
    # set up empty result arrays
        
    forcing_mean = np.zeros(len(subset_sizes))
    forcing_std = np.zeros(len(subset_sizes))

    for i in range(len(subset_sizes)):
        
        # set up empty result arrays
        
        forcing = np.zeros(len(num_reps))

        for j in range(len(num_reps)):
            # open subset and result datasets
            results = xr.open_dataset(result_directory + str(subset_sizes[i]) + result_filename + str(num_reps[j]) + '.h5', engine = "netcdf4")
            subsets = ref_data.isel(spectral_coord = results.S.data)
                        
            # compute flux error            
            _, present_OLR = compute_testing_rrms(results.S.data, results.W.data, 0, ref_data.spectral_fluxes.isel(half_level = 0, spectral_coord = results.S.data), ref_data.reference_fluxes.isel(half_level = 0).T, 1, len(ref_data.column.data))
            _, forced_OLR = compute_testing_rrms(results.S.data, results.W.data, 0, spectral_perturbed_OLR.isel(spectral_coord = results.S.data), reference_perturbed_OLR.T, 1, len(ref_data.column.data))
            
            # compute forcing error
            forcing[j] = rel_rms(np.array([present_OLR - forced_OLR]), np.array([ref_data.reference_fluxes.isel(half_level = 0).data.T - reference_perturbed_OLR.data.T]))

           
        # take mean and std across repetition
        forcing_mean[i] = np.mean(forcing)
        forcing_std[i] = np.std(forcing)
            
    return forcing_mean, forcing_std

def stat_loop(filenames, ref_data, indeces, present_indeces):
    #.isel(half_level = np.arange(53), level = np.arange(53))
    perturbed_data = xr.open_mfdataset(filenames, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').compute()

    if len(perturbed_data.half_level) > 53:
    
        perturbed_data_order = xr.Dataset(
            data_vars = dict(
                spectral_fluxes = (["half_level", "column", "spectral_coord"], perturbed_data.spectral_fluxes.transpose('half_level', 'column', 'spectral_coord').data[2:, :, :]),
                reference_fluxes = (["half_level", "column"], perturbed_data.reference_fluxes.data.T[::-1, :][2:, :]),
                #reference_heating = (["column", "level"], perturbed_data.reference_heating[::-1, :].data[:, 2:]),
                pressures = (["column", "half_level"], perturbed_data.pressures[:, ::-1].data[:, 2:]),

                ),
            coords = dict(
                half_level = np.arange(53),
                level = np.arange(53),
                column = perturbed_data.column.data,
                spectral_coord = perturbed_data.spectral_coord.data,
            ),
        )

    else:
        perturbed_data_order = xr.Dataset(
            data_vars = dict(
                spectral_fluxes = (["half_level", "column", "spectral_coord"], perturbed_data.spectral_fluxes.transpose('half_level', 'column', 'spectral_coord').data),
                reference_fluxes = (["half_level", "column"], perturbed_data.reference_fluxes.data.T[::-1, :]),
                #reference_heating = (["column", "level"], perturbed_data.reference_heating[::-1, :].data),
                pressures = (["column", "half_level"], perturbed_data.pressures[:, ::-1].data),

                ),
            coords = dict(
                half_level = np.arange(53),
                level = np.arange(53),
                column = perturbed_data.column.data,
                spectral_coord = perturbed_data.spectral_coord.data,
            ),
        )



    flux_errs = np.zeros((25, 2, 53))
    heat_errs = np.zeros((25, 2, 53))
    forcing_errs = np.zeros((25, 2))

    for scenario_idx in np.arange(25):
        scenario_cols_ref = ref_data.column.data[scenario_idx:][::25][present_indeces]
        scenario_cols_pert = perturbed_data_order.column.data[scenario_idx:][::25]


        flux_errs[scenario_idx, :, :], eval1_flux_std, heat_errs[scenario_idx, :, :], eval1_heat_std = profile_computations_rrms('/home/pc2943/quadrature21/', '_eval1_fhf_sw_all_forcings_nocfc_cont_', np.array([64]), np.array([0]), perturbed_data_order.isel(column = scenario_cols_pert))

        forcing_errs[scenario_idx, :], _ = forcing_computations_rrms('/home/pc2943/quadrature21/', '_eval1_fhf_sw_all_forcings_nocfc_cont_', np.array([64]), np.array([0]), ref_data.isel(column = scenario_cols_ref), perturbed_data_order.isel(column = scenario_cols_pert, half_level = 0).spectral_fluxes, perturbed_data_order.isel(column = scenario_cols_pert, half_level = 0).reference_fluxes)


    return flux_errs, heat_errs, forcing_errs
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.constants import speed_of_light as c
import xarray as xr
import seaborn as sns


# Error metrics

def rel_rms(estimate, reference):
    # relative root mean squared error across all ensembles
    # Buehler 2010 eqn 3
    ## Introduce floor for heating
    
    return np.sqrt((((estimate - reference)/np.maximum(np.abs(reference), 10**(-1)))**2).mean(axis = 1))


def calc_hr_error(pressure_hl, hr, hr_ref, pressure_range):

    mypow = (1./3)
    pressure_fl = 0.5*(pressure_hl[0:-1, :] + pressure_hl[1:, :])
    #weight = (pressure_hl[1:, :])**mypow - (pressure_hl[0:-1, :])**mypow
    weight = np.log(pressure_hl[1:, :]) - np.log(pressure_hl[0:-1, :])

    weight[np.where((pressure_fl < pressure_range[0]) | (pressure_fl >= pressure_range[1]))] = 0  
    
    nprof = 50
    for ii in range(nprof):
        weight[:, ii] = weight[:, ii] / np.sum(weight[:, ii])
    
    err = np.sqrt(np.sum(weight * ((hr-hr_ref)**2), axis = 1)/nprof)
    
    return err

def hr_bias_error(hr, hr_ref):
    error = hr - hr_ref
    return np.mean(error, axis = 1)

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

def compute_testing_arms(S, w, intercept, spectral_test, reference_test, num_hl, num_cols):
    spec_data = spectral_test.data
    q_test = np.empty((num_hl, num_cols))
    if num_hl == 1:
        q_test = np.matmul(w, spec_data[:, :].T)
        error = abs_rms(np.array([q_test]), np.array([reference_test]))
    else:
        for i in range(num_hl):
            q_test[i, :] = np.matmul(w, spec_data[i, :, :].T)
        
        # compute error of test estimate against test reference calculation
        error = abs_rms(q_test, reference_test)
    
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


def rel_rms_6(estimate, reference):
    # relative root mean squared error across all ensembles
    # Buehler 2010 eqn 3
    ## Introduce floor for heating
    
    return np.sqrt((((estimate - reference)/np.maximum(np.abs(reference), 10**(-6)))**2).mean(axis = 1))

def abs_rms(estimate, reference):
    # absolute root mean squared error across all ensembles
    # Buehler 2010 eqn 3
    return np.sqrt(((estimate - reference)**2).mean(axis = 1))


def profile_computations_rrms(result_directory, result_filename, subset_sizes, num_reps, ref_data):
    g = 9.81
    scaling = 3600*24
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
            forcing[j] = rel_rms(np.array([present_OLR - forced_OLR]), np.array([ref_data.reference_fluxes.isel(half_level = 0).T - reference_perturbed_OLR.T]))
           
        # take mean and std across repetition
        forcing_mean[i] = np.mean(forcing)
        forcing_std[i] = np.std(forcing)
            
    return forcing_mean, forcing_std

concs = xr.open_dataset('/data/robertp/CKDMIP_LBL/evaluation2/conc/ckdmip_evaluation2_concentrations_present.nc', engine = "netcdf4")
concs = concs.drop_isel(column = 14)

PRESSURES = concs.pressure_hl.data


filenames = []
for i in np.arange(50):
    if i != 14:
        filenames.append('/data/pc2943/eval2_present_col_' + str(int(i)) + '.h5')

pyarts_data = xr.open_mfdataset(filenames, engine = "netcdf4", combine = 'nested', concat_dim = 'column').compute()
#pyarts_data = pyarts_data.drop_sel(column = 14)

# transpose into appropriate shape for spectral and reference fluxes, open files
spectral_fluxes = pyarts_data.spec_net_flux.transpose('half_level', 'column', 'wavenumber').compute()
reference_fluxes = pyarts_data.int_net_flux.compute()

# include reference heating rate and pressure data
pressures = concs.pressure_hl.data

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

# ch4, n2o, o3, cfc11, cfc12, co2
n_array = np.array([1/8, 1/4, 1/2, 2, 4, 8])

filenames_cfc12 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_cfc12.append('/data/pc2943/eval2_' + str(n) + 'x_cfc12_col_' + str(int(i)) + '.h5')

filenames_cfc11 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_cfc11.append('/data/pc2943/eval2_' + str(n) + 'x_cfc11_col_' + str(int(i)) + '.h5')

filenames_co2 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_co2.append('/data/pc2943/eval2_' + str(n) + 'x_co2_col_' + str(int(i)) + '.h5')

filenames_ch4 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_ch4.append('/data/pc2943/eval2_' + str(n) + 'x_ch4_col_' + str(int(i)) + '.h5')

filenames_n2o = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_n2o.append('/data/pc2943/eval2_' + str(n) + 'x_n2o_col_' + str(int(i)) + '.h5')

filenames_o3 = []
for n in n_array:
    for i in np.arange(50):
        if i != 14:
            filenames_o3.append('/data/pc2943/eval2_' + str(n) + 'x_o3_col_' + str(int(i)) + '.h5')


cfc12 = xr.open_mfdataset(filenames_cfc12, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').rename({'wavenumber':'spectral_coord', 'spec_net_flux':'spectral_fluxes', 'int_net_flux':'reference_fluxes'}).transpose('half_level', 'column', 'spectral_coord').compute()
cfc11 = xr.open_mfdataset(filenames_cfc11, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').rename({'wavenumber':'spectral_coord', 'spec_net_flux':'spectral_fluxes', 'int_net_flux':'reference_fluxes'}).transpose('half_level', 'column', 'spectral_coord').compute()
co2 = xr.open_mfdataset(filenames_co2, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').rename({'wavenumber':'spectral_coord', 'spec_net_flux':'spectral_fluxes', 'int_net_flux':'reference_fluxes'}).transpose('half_level', 'column', 'spectral_coord').compute()
ch4 = xr.open_mfdataset(filenames_ch4, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').rename({'wavenumber':'spectral_coord', 'spec_net_flux':'spectral_fluxes', 'int_net_flux':'reference_fluxes'}).transpose('half_level', 'column', 'spectral_coord').compute()
n2o = xr.open_mfdataset(filenames_n2o, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').rename({'wavenumber':'spectral_coord', 'spec_net_flux':'spectral_fluxes', 'int_net_flux':'reference_fluxes'}).transpose('half_level', 'column', 'spectral_coord').compute()
o3 = xr.open_mfdataset(filenames_o3, combine = 'nested', concat_dim = 'column', engine = "netcdf4", coords = 'minimal').rename({'wavenumber':'spectral_coord', 'spec_net_flux':'spectral_fluxes', 'int_net_flux':'reference_fluxes'}).transpose('half_level', 'column', 'spectral_coord').compute()

### Present day, training set
# column computations
NUM_HL = 55

present_flux_mean, present_flux_std, present_heat_mean, present_heat_std = profile_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), data_eval2_LW_present)


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

for n_idx, n in enumerate(n_array):
    ch4_flux_means[n_idx, :], ch4_flux_stds[n_idx, :], ch4_heat_means[n_idx, :], ch4_heat_stds[n_idx, :] = profile_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), ch4.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))
    
    n2o_flux_means[n_idx, :], n2o_flux_stds[n_idx, :], n2o_heat_means[n_idx, :], n2o_heat_stds[n_idx, :] = profile_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), n2o.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))
    
    o3_flux_means[n_idx, :], o3_flux_stds[n_idx, :], o3_heat_means[n_idx, :], o3_heat_stds[n_idx, :] = profile_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), o3.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))
    
    cfc11_flux_means[n_idx, :], cfc11_flux_stds[n_idx, :], cfc11_heat_means[n_idx, :], cfc11_heat_stds[n_idx, :] = profile_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), cfc11.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))

    cfc12_flux_means[n_idx, :], cfc12_flux_stds[n_idx, :], cfc12_heat_means[n_idx, :], cfc12_heat_stds[n_idx, :] = profile_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), cfc12.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))
    
    co2_flux_means[n_idx, :], co2_flux_stds[n_idx, :], co2_heat_means[n_idx, :], co2_heat_stds[n_idx, :] = profile_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), co2.isel(column = np.arange(n_idx*49, (n_idx + 1)*49)))

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


for n_idx, n in enumerate(n_array):
    # generate 8x outside of loop because of the way I saved the data
    ch4_forcing_means[n_idx], ch4_forcing_stds[n_idx] = forcing_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), data_eval2_LW_present, ch4.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).spectral_fluxes, ch4.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).reference_fluxes)

    n2o_forcing_means[n_idx], n2o_forcing_stds[n_idx] = forcing_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), data_eval2_LW_present, n2o.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).spectral_fluxes, n2o.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).reference_fluxes)

    o3_forcing_means[n_idx], o3_forcing_stds[n_idx] = forcing_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), data_eval2_LW_present, o3.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).spectral_fluxes, o3.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).reference_fluxes)

    cfc11_forcing_means[n_idx], cfc11_forcing_stds[n_idx] = forcing_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), data_eval2_LW_present, cfc11.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).spectral_fluxes, cfc11.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).reference_fluxes)

    cfc12_forcing_means[n_idx], cfc12_forcing_stds[n_idx] = forcing_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), data_eval2_LW_present, cfc12.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).spectral_fluxes, cfc12.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).reference_fluxes)

    co2_forcing_means[n_idx], co2_forcing_stds[n_idx] = forcing_computations_rrms('/home/pc2943/quadrature21/', '_fhf_present_all_forcings_order_', np.array([64]), np.array([0]), data_eval2_LW_present, co2.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).spectral_fluxes, co2.isel(column = np.arange(n_idx*49, (n_idx + 1)*49), half_level = 0).reference_fluxes)


    
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

rrms_errors.to_netcdf('all_forcing_lw_rrms_order.h5')
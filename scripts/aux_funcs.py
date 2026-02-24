import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc
from scipy.constants import speed_of_light as c
import xarray as xr
import seaborn as sns

from pyrte_rrtmgp import rrtmgp_gas_optics
from pyrte_rrtmgp.data_types import (
    GasOpticsFiles,
    OpticsProblemTypes,
    RFMIPExampleFiles,
)
from pyrte_rrtmgp.rte_solver import rte_solve
from pyrte_rrtmgp.utils import load_rrtmgp_file

### functions to help process the DDQ/ARTS/RRTMGP data, and compute error metrics.

def rel_rms(estimate, reference):
    """
    Relative root mean squared error across all ensembles
    (Buehler et al., 2010, eqn. 3)
    Floor of 0.1 to avoid artificially amplifying small values.

    IN:
        estimate [array]: estimate quantity, ensemble axis is assumed to be 1
        reference [array]: reference quantity, of the same shape as estimate.
    
    OUT:
        the relative root mean squared error
    """
    
    return np.sqrt((((estimate - reference)/np.maximum(np.abs(reference), 10**(-1)))**2).mean(axis = 1))


def compute_testing_rrms(w, spectral_test, reference_test, num_hl, num_cols):
    """
    Function that wraps around rel_rms, ensures that data structures are the correct shape

    IN:
        w [cm-1]: spectral weights, as determined by DDQ
        spectral_test: spectral fluxes/heating rates at optimal wavenumbers
        reference_test: corresponding reference broadband fluxes/heating rates
        num_hl [int]: number of vertical half-levels
        num_cols [int]: number of independent atmospheres

    OUT:
        error: the relative root mean squared error between the estimate and the reference
        q_test: the flux/heating rate as estimated by DDQ
    """
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


def profile_computations_rrms(result_directory, result_filename, subset_sizes, num_reps, ref_data):
    """
    Compute relative root mean squared error between fluxes and heating rates in DDQ output and some reference calculation

    IN:
        result_directory [str]: where to search for results files
        result_filename [str]: naming convention of the result filename
            DDQ configurations are assumed to be named as: num_wavenumbers + result_filename + opt_idx + .h5
            where num_wavenumbers is the number of representative wavenumbers,
                result_filename is a string that describes the configuration (lw/sw, present/forcing)
                and opt_idx is the index of the independent optimization
        subset_sizes [array of int]: how many representative wavenumbers there are
        num_reps [array of int]: what are the indeces of the independent optimizations?
        ref_data [xarray]: a data structure that provides the reference data.
    
    OUT:
        flux_train_rrms_mean [array: (subset_sizes, half_level)]: relative root mean squared error of fluxes, mean across training profiles
        flux_train_rrms_std [array: (subset_sizes, half_level)]: standard deviation of fluxes, mean across training profiles
        heat_train_rrms_mean [array: (subset_sizes, half_level)]: root mean squared error of heating rates, mean across training profiles
        heat_train_rrms_std [array: (subset_sizes, half_level)]: std. of heating rates, mean across training profiles
    """

    g = 9.81 # gravity
    scaling = 3600*24 # convert heating rate to K/day
    reference_heating = -scaling*(np.gradient((ref_data.reference_fluxes.transpose('half_level', 'column').T), axis = -1)*g/np.gradient(ref_data.pressures.data, axis = -1)/1004)
    
    # set up empty result arrays

    flux_train_rrms_mean = np.zeros((len(subset_sizes), len(ref_data.half_level.data)))
    flux_train_rrms_std = np.zeros((len(subset_sizes), len(ref_data.half_level.data)))
    
    heat_train_rrms_mean = np.zeros((len(subset_sizes), len(ref_data.half_level.data)))
    heat_train_rrms_std = np.zeros((len(subset_sizes), len(ref_data.half_level.data)))
        
    for i in range(len(subset_sizes)):
        
        # set up empty result arrays
        flux_train_rrms = np.zeros((len(num_reps), len(ref_data.half_level.data)))
        heat_train_rrms = np.zeros((len(num_reps), len(ref_data.half_level.data)))
                
        for j in range(len(num_reps)):
            # open subset and result datasets
            results = xr.open_dataset(result_directory + str(subset_sizes[i]) + result_filename + str(num_reps[j]) + '.h5', engine = "netcdf4")
            subsets = ref_data.isel(spectral_coord = results.S.data)
                        
            # compute flux error            
            flux_train_rrms[j, :], total_fluxes_train = compute_testing_rrms(results.W.data,
                                                                             subsets.spectral_fluxes.transpose('half_level', 'column', 'spectral_coord'), 
                                                                             ref_data.reference_fluxes.transpose('half_level', 'column'), 
                                                                             len(ref_data.half_level.data), 
                                                                             len(ref_data.column.data))
            
            train_heating = -scaling*(np.gradient((total_fluxes_train.T), axis = -1)*g/np.gradient(ref_data.pressures.data, axis = -1)/1004)

            heat_train_rrms[j, :] = rel_rms(train_heating.T, reference_heating.T)
        
        # take mean and std across repetition
        flux_train_rrms_mean[i, :] = np.mean(flux_train_rrms, axis = 0)
        flux_train_rrms_std[i, :] = np.std(flux_train_rrms, axis = 0)

        heat_train_rrms_mean[i, :] = np.mean(heat_train_rrms, axis = 0)
        heat_train_rrms_std[i, :] = np.std(heat_train_rrms, axis = 0)
                            
    return flux_train_rrms_mean, flux_train_rrms_std, heat_train_rrms_mean, heat_train_rrms_std


def forcing_computations_rrms(result_directory, result_filename, subset_sizes, num_reps, ref_data, spectral_perturbed_OLR, reference_perturbed_OLR):
    """
    Compute relative root mean squared error for forcing in DDQ output and some reference calculation

    IN:
        result_directory [str]: where to search for results files
        result_filename [str]: naming convention of the result filename
            DDQ configurations are assumed to be named as: num_wavenumbers + result_filename + opt_idx + .h5
            where num_wavenumbers is the number of representative wavenumbers,
                result_filename is a string that describes the configuration (lw/sw, present/forcing)
                and opt_idx is the index of the independent optimization
        subset_sizes [array of int]: how many representative wavenumbers there are
        num_reps [array of int]: what are the indeces of the independent optimizations?
        ref_data [xarray]: a data structure that provides the reference data (present-day).
        spectral_perturbed_OLR [W/m^2/cm-1]: spectral outgoing longwave radiation in the perturbed state (n x present-day concentrations)
        reference_perturbed_OLR [W/m^2]: broadband integrated reference OLR in the perturbed state
    
    OUT:
        forcing mean [array: (subset_sizes, half_level)]: relative root mean squared error of forcing, mean across training profiles
        forcing_std [array: (subset_sizes, half_level)]: standard deviation of forcing, mean across training profiles
    """
    # set up empty result arrays
    forcing_mean = np.zeros(len(subset_sizes))
    forcing_std = np.zeros(len(subset_sizes))

    for i in range(len(subset_sizes)):
        
        # set up empty result arrays
        forcing = np.zeros(len(num_reps))

        for j in range(len(num_reps)):
            # open subset and result datasets
            results = xr.open_dataset(result_directory + str(subset_sizes[i]) + result_filename + str(num_reps[j]) + '.h5', engine = "netcdf4")
                        
            # compute flux error            
            _, present_OLR = compute_testing_rrms(results.W.data, ref_data.spectral_fluxes.isel(half_level = 0, spectral_coord = results.S.data), ref_data.reference_fluxes.isel(half_level = 0).T, 1, len(ref_data.column.data))
            _, forced_OLR = compute_testing_rrms(results.W.data, spectral_perturbed_OLR.isel(spectral_coord = results.S.data), reference_perturbed_OLR.T, 1, len(ref_data.column.data))
            
            # compute forcing error
            forcing[j] = rel_rms(np.array([present_OLR - forced_OLR]), np.array([ref_data.reference_fluxes.isel(half_level = 0).T - reference_perturbed_OLR.T]))
        
        # take mean and std across repetition
        forcing_mean[i] = np.mean(forcing)
        forcing_std[i] = np.std(forcing)
            
    return forcing_mean, forcing_std

def compute_rrtmgp_quantities_lw(atm, ref):
    """
    Wrapper around the RRTMGP call to interface between the CKDMIP-defined atmosphere and
    the pyRTE-RRTMGP package in the longwave; also compute error from LBL

    IN: 
        atm: xarray of mole fractions of greenhouse gases, here from CKDMIP, defined on 
            [column, layer]
        ref: reference line-by-line calculations in the following format: 
            xr.Dataset(
                data_vars = dict(
                    reference_fluxes = (["half_level", "column"], reference_fluxes),
                    reference_heating = (["column", "half_level"], reference_heating),
                    pressures = (["column", "half_level"], pressures),
                    ),
                coords = dict(
                    half_level = np.arange(53),
                    level = np.arange(53),
                    column = perturbed_data.column.data,
                    #spectral_coord = perturbed_data.spectral_coord.data,
                ),
            )
            NOTE that RRTMGP expects the opposite vertical orientation of CKDMIP: TOA is first index and surface is last
    OUT:
        net_flux [W/m^2]: net fluxes
        heating [K/day]: atmospheric heating rates
        flux_rrms_rrtmgp: relative root mean squared error in net flux
        heat_rrms_rrtmgp: relative root mean squared error in heating rates
    """

    internal_atm = atm.copy()

    ### Running pyRTE-RRTMGP is described in detail in the 
    # pyRTE-RRTMGP tutorial
    gas_optics_lw = rrtmgp_gas_optics.load_gas_optics(
        gas_optics_file=GasOpticsFiles.LW_G256
    )

    # rename gas concentrations
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


    # use pyRTE-RRTMGP to sove the RT. equations
    fluxes = rte_solve(internal_atm, add_to_input=False)
    net_flux = fluxes.lw_flux_down - fluxes.lw_flux_up

    # calculate heating rates from fluxes
    g = 9.81
    scaling = 3600*24
    heating = -scaling*(np.gradient((net_flux), axis = -1)*g/np.gradient(internal_atm.pres_level.data, axis = -1)/1004)

    flux_rrms_rrtmgp = rel_rms(net_flux.data.T, ref.reference_fluxes.data)

    heat_rrms_rrtmgp = rel_rms(heating.T, ref.reference_heating.data.T)

    return net_flux, heating, flux_rrms_rrtmgp, heat_rrms_rrtmgp



def compute_rrtmgp_quantities_sw(atm, ref):
    """
    Wrapper around the RRTMGP call to interface between the CKDMIP-defined atmosphere and
    the pyRTE-RRTMGP package in the shortwave; also compute error from LBL

    IN: 
        atm: xarray of mole fractions of greenhouse gases, here from CKDMIP, defined on 
            [column, layer]
        ref: reference line-by-line calculations in the following format: 
            xr.Dataset(
                data_vars = dict(
                    reference_fluxes = (["half_level", "column"], reference_fluxes),
                    reference_heating = (["column", "half_level"], reference_heating),
                    pressures = (["column", "half_level"], pressures),
                    ),
                coords = dict(
                    half_level = np.arange(53),
                    level = np.arange(53),
                    column = perturbed_data.column.data,
                    #spectral_coord = perturbed_data.spectral_coord.data,
                ),
            )
            NOTE that RRTMGP expects the opposite vertical orientation of CKDMIP: TOA is first index and surface is last
    OUT:
        net_flux [W/m^2]: net fluxes
        heating [K/day]: atmospheric heating rates
        flux_rrms_rrtmgp: relative root mean squared error in net flux
        heat_rrms_rrtmgp: relative root mean squared error in heating rates
    """
    internal_atm = atm.copy()

    ## set up gas optics
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

    ### solve SW RT. equation
    fluxes = rte_solve(internal_atm, add_to_input=False)

    net_flux = fluxes.sw_flux_down - fluxes.sw_flux_up

    ### calculate heating rates, in K/day
    g = 9.81
    scaling = 3600*24
    heating = -scaling*(np.gradient((net_flux), axis = -1)*g/np.gradient(internal_atm.pres_level.data, axis = -1)/1004)

    flux_rrms_rrtmgp = rel_rms(net_flux.data.T, ref.reference_fluxes.data)

    heat_rrms_rrtmgp = rel_rms(heating.T, ref.reference_heating.data.T)

    return net_flux, heating, flux_rrms_rrtmgp, heat_rrms_rrtmgp


def concs_to_rrtmgp(mults, zenith, albedos):
    """
    Converts CKDMIP's concentration dataset to an RRTMGP-ready set of atmospheres.
    IN:
        mults [array]: array of length 6 where each entry corresponds to the perturbation of the following 
            gases: [ch4, n2o, o3, co2, cfc11, cfc12]
        zenith [array]: array of the cosine of the solar zenith angle to sample
        albedos [array]: array of scalar surface albedo/reflectivity to sample
    OUT:
        perturbed_concs [xarray]: modified data-structure in format acceptable for RRTMGP, 
        potentially with perturbed gas concentrations
    """
    ### CKDMIP concentration files
    concs = xr.open_dataset('path/to/CKDMIP_data/ckdmip_evaluation2_concentrations_present.nc', engine = "netcdf4")

    ### ARTS reference files
    ref = xr.open_mfdataset('path/to/reference_data')

    ### reshape data to structure that RRTMGP expects
    col_concs = concs.isel(half_level = np.arange(2, 55), level = np.arange(2, 54)).copy() # CKDMIP's top pressure levels are too high for RRTMGP
    col_concs = col_concs.rename({"level":"layer", "half_level":"level"}) # rename variables
    col_concs = col_concs.rename({"temperature_hl":"temp_level", "temperature_fl":"temp_layer", "pressure_hl":"pres_level", "pressure_fl":"pres_layer"})
    col_concs = col_concs.assign({"surface_temperature":(["column"], col_concs.temp_level.data[:, -1])}) # set surface temperature
    # RRTMGP requires CO but it can be zero
    col_concs = col_concs.assign({"co_mole_fraction_fl": (["column", "layer"], 0*np.ones((np.shape(col_concs.h2o_mole_fraction_fl.data)))), "co_mole_fraction_hl": (["column", "level"], 0*np.ones((np.shape(col_concs.h2o_mole_fraction_hl.data))))})

    ### add surface_albedo and solar zenith angle mu0 to dataset

    present_zenith = np.zeros((len(ref.column.data)))
    present_albedos = np.zeros((len(ref.column.data)))

    j = 0
    for i in np.arange(len(concs.column.data)):
        for z in zenith:
            for reflectivity in albedos:
                present_zenith[j] = z
                present_albedos[j] = reflectivity
                j += 1

    col_concs = col_concs.assign({"surface_albedo": (["column"], present_albedos), "mu0": (["column"], present_zenith)})

    ### check that mults is in the correct format
    if len(mults) != 6:
        print("mults should correspond to the perturbation of the following gases: [ch4, n2o, o3, co2, cfc11, cfc12]")
        print("mults must be length 6")
        return
    
    perturbed_concs = col_concs.copy() # for safety
    perturbed_concs = perturbed_concs.assign({"ch4_mole_fraction_fl": (["column", "layer"], mults[0]*col_concs.ch4_mole_fraction_fl.data), 
                                              "ch4_mole_fraction_hl": (["column", "level"], mults[0]*col_concs.ch4_mole_fraction_hl.data)})
    perturbed_concs = perturbed_concs.assign({"n2o_mole_fraction_fl": (["column", "layer"], mults[1]*col_concs.n2o_mole_fraction_fl.data), 
                                              "n2o_mole_fraction_hl": (["column", "level"], mults[1]*col_concs.n2o_mole_fraction_hl.data)})
    perturbed_concs = perturbed_concs.assign({"o3_mole_fraction_fl": (["column", "layer"], mults[2]*col_concs.o3_mole_fraction_fl.data), 
                                              "o3_mole_fraction_hl": (["column", "level"], mults[2]*col_concs.o3_mole_fraction_hl.data)})
    perturbed_concs = perturbed_concs.assign({"co2_mole_fraction_fl": (["column", "layer"], mults[3]*col_concs.co2_mole_fraction_fl.data), 
                                              "co2_mole_fraction_hl": (["column", "level"], mults[3]*col_concs.co2_mole_fraction_hl.data)})
    perturbed_concs = perturbed_concs.assign({"cfc11_mole_fraction_fl": (["column", "layer"], mults[4]*col_concs.cfc11_mole_fraction_fl.data), 
                                              "cfc11_mole_fraction_hl": (["column", "level"], mults[4]*col_concs.cfc11_mole_fraction_hl.data)})
    perturbed_concs = perturbed_concs.assign({"cfc12_mole_fraction_fl": (["column", "layer"], mults[5]*col_concs.cfc12_mole_fraction_fl.data), 
                                              "cfc12_mole_fraction_hl": (["column", "level"], mults[5]*col_concs.cfc12_mole_fraction_hl.data)})

    return perturbed_concs

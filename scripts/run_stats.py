
import xarray as xr
import numpy as np
import generate_stats as gs


ref_data = xr.open_dataset('/data/pc2943/eval2_sw_full_sun_mu.h5', engine = "netcdf4").compute()
ref_data_order = xr.Dataset(
    data_vars = dict(
        spectral_fluxes = (["half_level", "column", "spectral_coord"], ref_data.spectral_fluxes.transpose('half_level', 'column', 'spectral_coord').data),
        reference_fluxes = (["half_level", "column"], ref_data.reference_fluxes.data.T[::-1, :].data),
        pressures = (["column", "half_level"], ref_data.pressures[:, ::-1].data),

    ),
    coords = dict(
        half_level = ref_data.half_level.data,
        column = ref_data.column.data,
        spectral_coord = ref_data.spectral_coord.data,
    ),
)

ref_data_order = ref_data_order.isel(half_level = np.arange(53))


lats = np.array([0.1, 0.3, 0.52, 0.7, 0.9])
albedos = np.array([0.15, 0.30, 0.45, 0.6, 0.75])

# ch4, n2o, o3, cfc11, cfc12, co2
n_array = np.array([1/8, 1/4, 1/2, int(2), int(4), int(8)])
col_indeces = [np.arange(40, 50), np.arange(10), np.array([10, 11, 12, 13, 15, 16, 17, 18, 19]), np.arange(30, 40), np.arange(20, 30), np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])]
present_indeces = [np.arange(39, 49), np.arange(10), np.arange(10, 19), np.arange(29, 39), np.arange(19, 29), np.array([0, 5, 10, 14, 19, 24, 29, 34, 39, 44])]

for n_idx, n in enumerate(n_array):
    mults = np.array([[n, 1, 1, 1], [1, n, 1, 1], [1, 1, n, 1], [1, 1, 1, n]])
    gas_label = ['ch4', 'n2o', 'o3', 'co2']
    idx = col_indeces[n_idx]
    present_idx = present_indeces[n_idx]

    for gas_idx in range(mults.shape[0]):

        filenames = []
        for i in idx:

            for lat in lats:
                for reflectivity in albedos:

                    if n == 8 or n == 4 or n == 2:
                        filenames.append('/data/pc2943/eval2_sw_full_sun_' + str(i) + '_mu_' + str(lat) + '_ref_' + str(reflectivity) +'_' + str(int(n)) + 'x_' + gas_label[gas_idx] + '.h5')

                    else:
                        filenames.append('/data/pc2943/eval2_sw_full_sun_' + str(i) + '_mu_' + str(lat) + '_ref_' + str(reflectivity) +'_' + str(n) + 'x_' + gas_label[gas_idx] + '.h5')


        flux_errs, heat_errs, forcing_errs = gs.stat_loop(filenames, ref_data_order, idx, present_idx)

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
        data.to_netcdf('rel_rms_' + str(n) + 'x_' + gas_label[gas_idx] + '_01.h5')
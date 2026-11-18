import pandas as pd, xarray as xr, numpy as np, matplotlib as mpl, geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import pickle

from functions import calc_z_scores, aggregate_by_country, build_country_weights, verify_population_coverage, basepath, cpath, cider_inj_lats_popweighted, lat_to_str_dict, cider_combined, baseline_start, baseline_end, W_norm_eu, countries_idx_eu, country_da, weights_2d, country_flat, weights_flat_vals, folded_non_sovereign, dropped_zero_pop, population_total, population_accounted, population_coverage_frac

n_members = 500
newpath = basepath + f'{n_members}_members/'

eval_buffer_months = 12
tau = 12
eval_len_months = 48
delta_t = 1

### file path for saving results
rpath = newpath + f'results_buffer={eval_buffer_months}m_tau={tau}m_eval={eval_len_months}m_k_2050/'
print(rpath)


## Load saved dict of dates for first f threshold crossings by country 
with open(newpath + 'f_thresh_crossing_dates_v3.pickle', 'rb') as handle:
    f_crossing_dates = pickle.load(handle)

with open(newpath + 'f_by_country_no_SAI_k=50_to_1500.pickle', 'rb') as handle:
    f_by_country = pickle.load(handle)

with open(newpath + 'country_k_for_2050_median_f_crossing.pickle', 'rb') as handle:
    country_k = pickle.load(handle)

ssp_plus_invert = xr.open_dataset(newpath + 
                    f'SSP370_plus_InVERT_{n_members}member_ensemble_TREFHT_2015_to_2065.nc')

## Select baseline period from combined dataset
baseline = ssp_plus_invert.sel(time=slice(baseline_start, baseline_end))

## Pre-compute the baseline climatology once
baseline_monthly_mean = baseline.TREFHT.mean('ensemble').groupby('time.month').mean('time')  # (month, lat, lon)
baseline_monthly_std = baseline.TREFHT.std('ensemble').groupby('time.month').mean('time')    # (month, lat, lon)

# Plain numpy arrays indexed 0..11 for month 1..12, for fast vectorized lookup later
baseline_mean_arr = baseline_monthly_mean.sel(month=np.arange(1, 13)).values   # (12, lat, lon)
baseline_std_arr = baseline_monthly_std.sel(month=np.arange(1, 13)).values     # (12, lat, lon)

# Calendar month (1-12) for every time step in the full dataset -- needed to look up
# the right baseline map for any gathered window
months_full = np.array([t.month for t in ssp_plus_invert.time.values])   # (T,)


def gather_post_window(k, country, SAI_start_buffer_months=1, window_months=60):
    
    f_crossings = f_crossing_dates[k].sel(country=country)
    crossing_idx = f_crossings.first_time_idx.values
    valid_ens = crossing_idx >= 0

    SAI_start_tidx = crossing_idx + SAI_start_buffer_months

    T = ssp_plus_invert.sizes['time']
    # Only need enough POST data now -- no more subtracting window_months for a pre-window
    in_bounds = SAI_start_tidx + window_months <= T
    usable = valid_ens & in_bounds

    n_ens = ssp_plus_invert.sizes['ensemble']
    offsets = np.arange(0, window_months)
    time_idx_matrix = SAI_start_tidx[:, None] + offsets[None, :]
    time_idx_matrix = np.clip(time_idx_matrix, 0, T - 1)

    TREFHT_vals = ssp_plus_invert.TREFHT.transpose('time', 'lat', 'lon', 'ensemble').values
    ens_idx = np.arange(n_ens)[:, None]
    gathered = TREFHT_vals[time_idx_matrix, :, :, ens_idx]      # (ensemble, window_months, lat, lon)
    months_matrix = months_full[time_idx_matrix]                # (ensemble, window_months)

    return gathered, months_matrix, usable, SAI_start_tidx

def get_f_init_at_sai_start(k, country, SAI_start_tidx, usable):
    
    f_country_full = f_by_country[k].sel(country=country).values   # (time, ensemble), from the ORIGINAL no-SAI run
    n_ens = f_country_full.shape[1]
    safe_idx = np.clip(SAI_start_tidx, 0, f_country_full.shape[0] - 1)   # only matters for unusable members
    f_init = f_country_full[safe_idx, np.arange(n_ens)]
    return f_init   # (ensemble,) -- exact f value right when SAI activates, per member

def T_to_z(T_arr, months_arr, baseline_mean_arr, baseline_std_arr):
    """
    T_arr: (ensemble, n_time, lat, lon); 
    months_arr: (ensemble, n_time), values 1-12
    """
    month_idx = months_arr - 1                          # 0-based
    mean_sel = baseline_mean_arr[month_idx]              # (ensemble, n_time, lat, lon), fancy-indexed
    std_sel = baseline_std_arr[month_idx]                # same shape
    return (T_arr - mean_sel) / std_sel

def calc_f_fast(z_scores, k, tau, delta_t, f_init=0.66, dtype=np.float32):
    '''
    Configure initial condition (set to f_threshold at crossing point for post-SAI f calcs)
    '''
    from scipy.special import erf

    z_vals = z_scores.values.astype(np.float64)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ## Ignore divide-by-zero warning output
        N_vals = 2.0 / (1.0 - erf(z_vals / np.sqrt(2.0)))
        ratio = np.where(np.isinf(N_vals), 1.0, N_vals / (k + N_vals))

    N_vals = N_vals.astype(dtype)
    a = (1.0 - delta_t * (ratio + 1.0 / tau)).astype(dtype)
    b = (delta_t * ratio).astype(dtype)

    T = z_scores.sizes['time']
    f_vals = np.empty_like(N_vals)
    
    f_vals[0] = f_init
    for t in range(1, T):
        f_vals[t] = a[t - 1] * f_vals[t - 1] + b[t - 1]

    f = xr.DataArray(f_vals, coords=z_scores.coords, dims=z_scores.dims, name='f')
    return f

def evaluate_injection_rate(inj_lat, rate, country, cider_combined, T_post_baseline, months_post, 
                            f_init, usable, k, eval_buffer_months, 
                            eval_len_months, threshold=0.66):
    """
    rate : scalar, or array of shape (ensemble,)
        Scalar -- the same injection rate is applied to every ensemble member
        (original linear-scan behavior).
        Array -- one rate PER ensemble member, pulled from cider_combined in a
        single vectorized/pointwise .sel() call along the 'ensemble' dim. This
        is what lets the binary-search version test a different candidate rate
        per member in each iteration without looping over members in Python.
    """
    if np.ndim(rate) == 0:
        CIDER_dT = cider_combined.sel(injection_lat=inj_lat, injection_rate=rate).values
        CIDER_dT = CIDER_dT[None, :, :, :]   # (1, time, lat, lon) -- broadcasts over ensemble
    else:
        rate_per_member = xr.DataArray(np.asarray(rate), dims='ensemble',
                                        coords={'ensemble': ssp_plus_invert.ensemble.values})
        CIDER_dT = cider_combined.sel(injection_lat=inj_lat, injection_rate=rate_per_member).values
        # -> (ensemble, time, lat, lon), each member's slice pulled at its own candidate rate

    T_post_with_SAI = T_post_baseline + CIDER_dT

    z_post_vals = T_to_z(T_post_with_SAI, months_post, baseline_mean_arr, baseline_std_arr)

    z_post_da = xr.DataArray(
        z_post_vals.transpose(1, 2, 3, 0),
        dims=['time', 'lat', 'lon', 'ensemble'],
        coords={'time': np.arange(z_post_vals.shape[1]),
                 'lat': ssp_plus_invert.lat.values,
                 'lon': ssp_plus_invert.lon.values,
                 'ensemble': ssp_plus_invert.ensemble.values})

    f = calc_f_fast(z_post_da, k, tau, delta_t, f_init=f_init)   # exact per-member initial condition
    ## print('f NaNs?', np.isnan(f).any())
    
    f_by_country_result = aggregate_by_country(f, W_norm_eu, countries_idx_eu)
    country_f = f_by_country_result.sel(country=country)

    eval_slice = slice(eval_buffer_months, eval_buffer_months + eval_len_months)
    eval_f = country_f.isel(time=eval_slice).values

    passes = (eval_f < threshold).all(axis=0) & usable
    return passes


def find_min_injection_rate_binary(k, country, inj_lat, cider_combined, eval_buffer_months,
                            threshold=0.66, SAI_start_buffer_months=1, eval_len_months=eval_len_months,
                            verbose=True):
    """
    Vectorized binary search for the minimum passing injection rate, per ensemble member.

    Each iteration pulls one CIDER injection-rate slice per member (via vectorized/pointwise
    .sel() along 'ensemble' in evaluate_injection_rate
    """

    rates_available = cider_combined.injection_rate.values.astype(int)
    rate_min, rate_max = int(rates_available.min()), int(rates_available.max())
    window_months = cider_combined.sizes['time']

    assert eval_buffer_months + eval_len_months <= window_months, (
        f"eval_buffer_months ({eval_buffer_months}) + eval_len_months ({eval_len_months}) "
        f"= {eval_buffer_months + eval_len_months} exceeds the {window_months} months "
        f"of CIDER data available -- reduce one of them or gather more CIDER months.")

    T_post_baseline, months_post, usable, SAI_start_tidx = gather_post_window(
        k, country, SAI_start_buffer_months, window_months)
    if verbose:
        print("usable:", usable.sum(), "/", len(usable))

    f_init = get_f_init_at_sai_start(k, country, SAI_start_tidx, usable)

    n_ens = ssp_plus_invert.sizes['ensemble']
    eval_kwargs = dict(country=country, cider_combined=cider_combined,
                        T_post_baseline=T_post_baseline, months_post=months_post,
                        f_init=f_init, usable=usable, k=k,
                        eval_buffer_months=eval_buffer_months,
                        eval_len_months=eval_len_months, threshold=threshold)

    # Members with no valid crossing / out-of-bounds window can never be resolved --
    # exclude them from the search entirely
    resolved = ~usable

    # Check the top of the range first
    passes_at_max = evaluate_injection_rate(inj_lat, rate_max, **eval_kwargs)
    never_passes = usable & ~passes_at_max
    if never_passes.any() and verbose:
        print(f"  {never_passes.sum()} usable member(s) never pass even at the top "
              f"rate ({rate_max} Tg/yr) -- leaving these as NaN.")
    resolved = resolved | never_passes

    lo = np.full(n_ens, rate_min, dtype=int)
    hi = np.full(n_ens, rate_max, dtype=int)

    max_iters = int(np.ceil(np.log2(rate_max - rate_min + 1))) + 1

    for iteration in range(max_iters):
        active = ~resolved & (lo < hi)
        if not active.any():
            break

        mid = (lo + hi) // 2
        # For inactive members, mid is harmless (already-tested, in-range value) --
        # computed for the whole ensemble each iteration to keep this vectorized.
        passes = evaluate_injection_rate(inj_lat, mid, **eval_kwargs)

        hi = np.where(active & passes, mid, hi)
        lo = np.where(active & ~passes, mid + 1, lo)

        if verbose:
            n_converged = int((~active | (lo >= hi)).sum())
            print(f'  iter {iteration}: {n_converged}/{n_ens} converged so far')

    min_rate = np.where(resolved, np.nan, lo).astype(float)
    min_rate = np.where(never_passes, np.nan, min_rate)

    return xr.DataArray(min_rate, dims=['ensemble'],
                        coords={'ensemble': ssp_plus_invert.ensemble.values},
                         name=f'min_injection_rate_{country}_k{k}')

print('Computing single lat')

countries_to_run = ['Australia', 'Bangladesh', 'Brazil', 'China', 'Egypt', 'Ethiopia', 'European Union', 'France', 'Germany', 'India', 'Indonesia', 'Japan', 'Nigeria', 'Pakistan', 'Russia', 'Saudi Arabia', 'South Africa', 'United Kingdom', 'United States of America']

       
n_ens = ssp_plus_invert.sizes['ensemble']
results = np.full((len(countries_to_run), n_ens), np.nan)

for ci, country in enumerate(countries_to_run):

    k = country_k[country]

    inj_lat = lat_to_str_dict[
        cider_inj_lats_popweighted.sel(country=country).nearest_cider_lat.values.tolist()]
    
    print(country, k, inj_lat)

    min_rate_da = find_min_injection_rate_binary(k, country, inj_lat, cider_combined, 
                                          eval_buffer_months)
    results[ci, :] = min_rate_da.values
    
## Save 
min_injection_ds = xr.DataArray(results, 
                                dims=['country', 'ensemble'],
                                coords={'country': countries_to_run, 
                                        'ensemble': ssp_plus_invert.ensemble.values},
                                name='min_injection_rate')

min_injection_ds.to_netcdf(rpath + 'single_lat_min_rates_country_k_2050.nc')
print('Single lat done')


print('Computing global uniform')

## Need to run again for global uniform 
n_ens = ssp_plus_invert.sizes['ensemble']
results = np.full((len(countries_to_run), n_ens), np.nan)

for ci, country in enumerate(countries_to_run):

    k = country_k[country]
    inj_lat = 'uniform'
    
    print(country, k, inj_lat)

    min_rate_da = find_min_injection_rate_binary(k, country, inj_lat, cider_combined, 
                                          eval_buffer_months)

    results[ci, :] = min_rate_da.values
    
## Save 
min_injection_ds_global = xr.DataArray(results, 
                                dims=['country', 'ensemble'],
                                coords={'country': countries_to_run, 
                                        'ensemble': ssp_plus_invert.ensemble.values},
                                name='min_injection_rate')

min_injection_ds_global.to_netcdf(rpath + 'global_uniform_min_rates_country_k_2050.nc')
print('Done')
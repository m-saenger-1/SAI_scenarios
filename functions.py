import pandas as pd, xarray as xr, numpy as np
from matplotlib import pyplot as plt
import warnings, nc_time_axis, seaborn, os, json
import cartopy.crs as ccrs, cartopy.feature as cfeature
import seaborn as sns
import scipy.stats
import pickle, subprocess, random
import collections, datetime
from tabulate import tabulate
collections.Iterable = collections.abc.Iterable
warnings.filterwarnings('ignore')


### Stuff to load universally

## File path shortcuts
basepath = '/home/msaenger/Perceptions_2025/cleaned_code/'
cpath = basepath + 'CIDER_output/'


######### Load background data
cider_inj_lats_popweighted = xr.open_dataset(basepath + \
                            'cider_injection_latitudes_by_country_popweighted.nc')

lat_to_str_dict = {-15.0: '15S', -30.0: '30S', -60.0: '60S', 
                   0.0: '0NS', 15.0: '15N', 30.0: '30N', 60.0: '60N'}


## CIDER gridded delta T output from stratospheric aerosol injection rates and injection latitudes
cider_combined = xr.open_dataarray(cpath + \
        'CIDER_TREFHT_output_combined_1.0_Tg_increments_first_5_years.nc')


## 2020 population data, already on the 96x192 model grid
pop = xr.open_dataarray(basepath + 'gpw-v4-population-count-rev11_totpop_96x192.nc').sel(time=2020)

countries_zip_path = basepath + 'natural_earth_data/ne_50m_admin_0_countries.zip'


## Define baseline time period
baseline_start = '2015-01'
baseline_end = '2024-12'



def dateshift_netCDF(fp):
    """
    Function to shift and save netcdf with dates at midpoint of month.
    :param fp: dataarray to be reoriented
    """
    f = fp
    if (np.unique(fp.indexes['time'].day)[0] == 1) & (len(np.unique(fp.indexes['time'].day)) == 1):
        new_time = fp.indexes['time']-datetime.timedelta(days=16)
        f = f.assign_coords({'time':new_time})
    return f

def cdo_regrid(targetgrid, file_to_regrid, regridded):
    '''
    Regrid with CDO.
    all input objects are strings: paths to files of target grid (to regrid fild_to_regrid to),
    file to be regridded, and where to save/what to name regridded file
    *** DOES NOT WORK ON MULTIPLE ENSEMBLE MEMBERS. ONLY DIMENSIONS CAN BE LAT, LON, AND TIME ***
    '''
    import subprocess
    
    targetgrid = '-remapbil,' + targetgrid
    
    subprocess.run(['cdo', targetgrid, file_to_regrid, regridded]) 

def areaweighted_mean(ds):
    '''
    Calculate area-weighted mean value of a global gridded variable 
    Input DataArray is global field of data at a single timepoint
    Adapted from https://docs.xarray.dev/en/stable/examples/area_weighted_temperature.html 
    Returns timeseries of area-weighted mean value of input variable
    over whatever spatial domain is specified in the input DataArray
    '''
    # create array of weights for each grid cell based on latitude
    
    if hasattr(ds, 'lat'): # if latitude dimension has label 'lat'
        weights = np.cos(np.deg2rad(ds.lat)) 
        weights.name = "weights"
        array_weighted = ds.weighted(weights)
        weighted_mean = array_weighted.mean(("lon", "lat"))
        
    elif hasattr(ds, 'latitude'): # if latitude dimension has label 'latitude'
        weights = np.cos(np.deg2rad(ds.latitude)) 
        weights.name = "weights"
        array_weighted = ds.weighted(weights)
        weighted_mean = array_weighted.mean(("longitude", "latitude"))
        
    return weighted_mean

def reorient_netcdf(fp):
    """
    Function to orient and save netcdf wrt -180,180 longitude.
    :param fp: dataarray to be reoriented
    
    Credit to Jessica Wan
    """
    f = fp
    if np.max(f.coords['lon'])>180:
        new_lon = [-360.00 + num if num > 180 else num for num in f.coords['lon'].values]
        f = f.assign_coords({'lon':new_lon})
        f = f.sortby(f.coords['lon'])
    return f

def calc_z_scores(ds, var_name, baseline, monthly_std=True):
    '''
    Calculate the number of standard deviations a given temperature is from the baseline mean
        - ds is the dataset you're calculating z scores for
        - var_name is the variable name for temperature
        - baseline is the monthly data over your baseline period
        - monthly_std=True means you're using month-specific baseline standard deviations
    '''
    baseline_monthly_mean = baseline[var_name].mean('ensemble').groupby(
                                            'time.month').mean('time') # map of mean T values for each month
    baseline_monthly_std = baseline[var_name].std('ensemble').groupby(
                                            'time.month').mean('time') # map of T stds for each month
    
    ## if you're not using month-specific baseline standard deviations, calc baseline std over full period
    baseline_std = baseline[var_name].std('ensemble').mean('time')
    
    ds['month'] = ds['time.month']

    z_scores = []
    for t_index in range(len(ds.time)): # Iterate over time
        T_vals = ds[var_name].isel(time=t_index)  # Current TREFHT values (dims = (lat,lon))
        month = ds.isel(time=t_index).month.values  # Find current month

        if monthly_std == True:
            # Calculate z score using baseline mean and std of T for the current month
            z_score = (T_vals - baseline_monthly_mean.sel(month=month)) / baseline_monthly_std.sel(month=month)
        else:
             z_score = (T_vals - baseline_monthly_mean.sel(month=month)) / baseline_std

        z_scores.append(z_score) # Add to list for the time index

    z_scores_da = xr.concat(z_scores, dim='time') # Concat to xarray DataArray
    
    return z_scores_da  

def calc_N_and_f_fast(z_scores, k, tau, delta_t, dtype=np.float32):
    
    from scipy.special import erf

    z_vals = z_scores.values.astype(np.float64)   # compute N in float64 to delay overflow
    N_vals = 2.0 / (1.0 - erf(z_vals / np.sqrt(2.0)))

    # Where N overflowed to inf, the true limit of N/(k+N) as N->inf is 1.0.
    # np.where avoids the inf/inf -> nan that numpy produces otherwise.
    ratio = np.where(np.isinf(N_vals), 1.0, N_vals / (k + N_vals))

    N_vals = N_vals.astype(dtype)
    a = (1.0 - delta_t * (ratio + 1.0 / tau)).astype(dtype)
    b = (delta_t * ratio).astype(dtype)

    T = z_scores.sizes['time']
    f_vals = np.empty_like(N_vals)
    ## Set initial f value
    f_vals[0] = 0.2
    for t in range(1, T):
        f_vals[t] = a[t - 1] * f_vals[t - 1] + b[t - 1]

    N = xr.DataArray(N_vals, coords=z_scores.coords, dims=z_scores.dims, name='N')
    f = xr.DataArray(f_vals, coords=z_scores.coords, dims=z_scores.dims, name='f')
    return N, f

"""
Shared helper for building a population-weighted (and optionally area-weighted) 
country-aggregation matrix from a Natural Earth country shapefile + a population grid.

This consolidates the following:
    - spatial join of grid points to country polygons
    - nearest-country fallback for unmatched grid points
    - sparse weight matrix construction + row normalization
    - dropping zero-population "countries"
    - aggregating EU member states into one "European Union" row

It also adds:
    1. A sovereignty filter option, so dependent territories (Greenland,
       French Guiana, Puerto Rico, Hong Kong, etc.) can be dropped
       BEFORE the spatial join, rather than silently ending up as
       their own "country" rows.
    2. An explicit, single `area_weight` switch controlling whether
       grid cells are weighted by population alone or by population * cos(lat) 

Usage:

from country_weights import build_country_weights, EU_MEMBERS

result = build_country_weights(
    pop=pop,                      # xr.DataArray, dims (lat, lon)
    countries_zip_path=zip_path,  # path to ne_50m_admin_0_countries.zip
    area_weight=False,            # set True to match notebook 04's cos(lat) weighting
)

    W_norm_eu       = result["W_norm_eu"]
    countries_idx_eu = result["countries_idx_eu"]
    country_da      = result["country_da"]       # per-gridcell country label, for reuse elsewhere 
    weights_2d      = result["weights_2d"]        # per-gridcell weight actually used
"""
import geopandas as gpd
import scipy.sparse as sp
from shapely.geometry import Point

# Current EU-27 member states 
EU_MEMBERS = [
    "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czechia", "Denmark",
    "Estonia", "Finland", "France", "Germany", "Greece", "Hungary", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Netherlands",
    "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden",
]

# Natural Earth's admin_0 'TYPE' field distinguishes sovereign states from
# dependencies/territories -- but it turned out to be an unreliable signal for
# that (see build_country_weights' docstring: Kazakhstan/Cuba read as
# non-sovereign under it, Greenland reads as sovereign, despite Greenland's
# own SOVEREIGNT field correctly saying Denmark). build_country_weights no
# longer uses TYPE at all -- it compares ADMIN to SOVEREIGNT directly instead.
# This list and check_sovereignty_types() are kept only as a general auditing
# tool if you want to inspect TYPE values for some other purpose.
DEFAULT_SOVEREIGN_TYPES = ["Sovereign country", "Country", "Sovereignty"]

def check_sovereignty_types(countries_zip_path):
    """
    Print the distinct 'TYPE' values in the Natural Earth countries shapefile
    and how many rows fall in each, so you can confirm DEFAULT_SOVEREIGN_TYPES
    is right for your copy of the data before relying on it.
    """
    countries = gpd.read_file(f"zip://{countries_zip_path}")
    print(countries["TYPE"].value_counts())
    return countries

def fold_territories(countries_zip_path, keep_as_sovereign=None, verbose=True):
    """
    Load the Natural Earth countries shapefile and fold every non-sovereign
    territory's polygon into its actual sovereign state's ADMIN label -- the same
    folding step build_country_weights runs internally, factored out here so that
    anything that only needs country geometries (e.g. a choropleth map) doesn't have
    to load a population grid and run the full rasterization pipeline just to get them.
    Using this instead of a separately-loaded shapefile also means country labels here
    are guaranteed to match build_country_weights' output exactly -- no hand-maintained
    name-matching fixes needed to reconcile two different shapefiles/fields.
 
    Parameters
    ----------
    countries_zip_path : str
        Path to the Natural Earth admin_0_countries shapefile zip.
    keep_as_sovereign : list[str], "all", or None, default None
        Same semantics as build_country_weights: a list of ADMIN names to exempt from
        folding, "all" to fold nothing at all, or None/[] to fold every non-sovereign
        territory into its SOVEREIGNT.
    verbose : bool, default True
        Print which territories got folded into which sovereign state.
 
    Returns
    -------
    countries : gpd.GeoDataFrame
        The full Natural Earth table, with its 'ADMIN' column already overwritten to
        each polygon's effective label (its own ADMIN name if kept sovereign, otherwise
        its SOVEREIGNT). Every original row/polygon is retained -- no dissolve. For
        plotting, just color each row by its (now effective) ADMIN value; several rows
        sharing an ADMIN value after folding render as separate polygons in the same
        color, which looks identical to a single dissolved polygon.
    folded_non_sovereign : list[str]
        "ADMIN -> SOVEREIGNT" strings describing which territories were relabeled and
        where their population/geometry now counts toward.
    """
    if isinstance(keep_as_sovereign, str):
        if keep_as_sovereign != "all":
            raise ValueError(
                "keep_as_sovereign must be a list of ADMIN names, the string 'all', "
                f"or None -- got the string {keep_as_sovereign!r}.")
        keep_all = True
    else:
        keep_all = False
        keep_as_sovereign = keep_as_sovereign or []
 
    countries = gpd.read_file(f"zip://{countries_zip_path}")
 
    if "SOVEREIGNT" not in countries.columns:
        raise ValueError("Expected a 'SOVEREIGNT' column in the Natural Earth shapefile to fold "
                        "dependencies into their actual sovereign state, but it wasn't found.")
 
    # Deliberately NOT using the 'TYPE' field here: it turns out to be an unreliable
    # signal for "is this its own country" -- e.g. in this shapefile release, Kazakhstan
    # and Cuba carry a TYPE value that reads as non-sovereign, while Greenland carries a
    # TYPE value that reads as sovereign, even though its SOVEREIGNT field correctly says
    # Denmark. Comparing ADMIN to SOVEREIGNT directly sidesteps that inconsistency: a
    # polygon folds into its SOVEREIGNT name whenever the two differ, full stop, unless
    # it's been explicitly exempted via keep_as_sovereign (for contested territories
    # where SOVEREIGNT encodes a political claim rather than settled fact), or unless
    # keep_as_sovereign="all" was passed, in which case nothing gets folded at all.
    if keep_all:
        is_kept = pd.Series(True, index=countries.index)
    else:
        is_kept = countries["ADMIN"].isin(keep_as_sovereign)
    folded_mask = ~is_kept & (countries["ADMIN"] != countries["SOVEREIGNT"])
    folded_non_sovereign = sorted(
        countries.loc[folded_mask, ["ADMIN", "SOVEREIGNT"]]
                 .drop_duplicates()
                 .apply(lambda r: f"{r['ADMIN']} -> {r['SOVEREIGNT']}", axis=1)
                 .tolist())
 
    if verbose and folded_non_sovereign:
        print(f"Folding {len(folded_non_sovereign)} non-sovereign territories into "
              f"their sovereign state, e.g.: {folded_non_sovereign[:10]}"
              f"{' ...' if len(folded_non_sovereign) > 10 else ''}")
 
    countries = countries.copy()
    countries["ADMIN"] = np.where(is_kept, countries["ADMIN"], countries["SOVEREIGNT"])
 
    return countries, folded_non_sovereign

def build_country_weights(pop,
                            countries_zip_path,
                            area_weight=False,
                            eu_members=None,
                            keep_as_sovereign=None,
                            verbose=True,
                            check_coverage=True,
                            coverage_atol=1e-6,
                            raise_on_coverage_fail=True):
    """
    Build a sparse, row-normalized (country x gridpoint) weight matrix for
    aggregating any (lat, lon, ...) field to country level, with a 'European Union' row appended.
 
    Parameters
    ----------
    pop : xr.DataArray
        Population count per grid cell, dims ('lat', 'lon'). This supplies both the weighting AND the (lat, lon) grid the 
        country mask is rasterized onto -- pass it already `.sel(time=...)`-reduced to 2D.
    countries_zip_path : str
        Path to the Natural Earth ne_50m_admin_0_countries.zip file.
    area_weight : bool, default False
        If True, weight cells by pop * cos(lat) 
        If False, weight by pop alone (matches the current notebook 02/06/07 aggregation code). 
        NOTE: population count data already scales with a cell's true area for a given
        density, so adding cos(lat) on top double-counts area to some degree -- decide which convention 
        you want and set this the same way everywhere, rather than leaving it to drift between notebooks.
    eu_members : list of str, optional
        Country ADMIN names to fold into the 'European Union' row. Defaults to EU_MEMBERS above.
    keep_as_sovereign : list of str, or the string "all", optional
        ADMIN names to always treat as their own country regardless of their Natural Earth
        SOVEREIGNT field. Use this for contested territories where SOVEREIGNT encodes a
        political claim you don't want to inherit silently -- e.g. Palestine's SOVEREIGNT field
        is "Israel", so without listing it here, Palestine's population would be folded into
        Israel's aggregate by default. Pass keep_as_sovereign=["Palestine"] to keep it separate.
        Pass keep_as_sovereign="all" to disable folding entirely and keep every territory
        (Greenland, Puerto Rico, French Guiana, etc.) as its own standalone country -- e.g.
        for a run where you want the raw, un-folded ADMIN list rather than sovereign states.
    verbose : bool, default True
        Print diagnostics (territories folded into their sovereign state, dropped zero-pop
        countries, unmatched grid points reassigned to nearest country, population coverage).
    check_coverage : bool, default True
        If True, verify that summing population weight across every "kept" country row
        (i.e. every row of W_norm_eu except the 'European Union' aggregate, which
        duplicates existing member-state rows rather than adding new population)
        accounts for 100% of the total population weight in the input `pop` grid.
        This catches grid points that silently fell through the country join and the
        nearest-neighbor fallback (e.g. because they lie outside the shapefile's extent).
    coverage_atol : float, default 1e-6
        Allowed absolute deviation of the coverage fraction from 1.0 (100%) before it's
        treated as a failure. Only used if check_coverage=True.
    raise_on_coverage_fail : bool, default True
        If True and check_coverage=True, raise a ValueError when coverage falls outside
        coverage_atol of 100%. If False, only a warning is issued, and the result dict
        still gets returned so you can inspect population_coverage_frac yourself.
 
    Returns
    -------
    dict with keys:
        W_norm_eu        : scipy.sparse.csr_matrix, shape (n_countries+1, n_points)
        countries_idx_eu : np.ndarray of country names, 'European Union' last
        country_da       : xr.DataArray, per-gridcell country label (lat, lon)
        weights_2d       : xr.DataArray, per-gridcell weight actually used
        country_flat     : flattened country label array (for reuse, e.g. centroid calcs)
        weights_flat_vals: flattened weight array (for reuse, e.g. centroid calcs)
        countries_gdf    : gpd.GeoDataFrame, the folded country polygons themselves (see
                           fold_territories) -- for plotting a choropleth keyed on the
                           exact same country labels used everywhere else above, without
                           needing a separately-loaded shapefile or manual name fixes
        folded_non_sovereign : list of "ADMIN -> SOVEREIGNT" strings describing which
                               territories were relabeled and where their population went
        dropped_zero_pop      : list of country names dropped for zero population weight
        population_total      : total population weight summed over the whole input grid
        population_accounted  : population weight summed across all kept country rows
                                 (excludes the duplicate 'European Union' row)
        population_coverage_frac : population_accounted / population_total
    """
    eu_members = eu_members or EU_MEMBERS
 
    # --- Load countries and fold non-sovereign territories into their sovereign state ---
    countries, folded_non_sovereign = fold_territories(
        countries_zip_path, keep_as_sovereign=keep_as_sovereign, verbose=verbose)
 
    # --- Build weighting grid ---
    if area_weight:
        area_weights = np.cos(np.deg2rad(pop["lat"])).broadcast_like(pop)
        weights_2d = pop * area_weights
    else:
        weights_2d = pop
 
    lat_vals = pop["lat"].values
    lon_vals = pop["lon"].values
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
 
    # --- Spatial join grid points to (now-relabeled) country polygons ---
    # Every polygon is retained here, including former dependencies -- they just
    # carry their sovereign's ADMIN label now, so the join attributes their grid
    # cells correctly without needing the nearest-neighbor fallback below.
    points = [Point(xy) for xy in zip(lon_grid.ravel(), lat_grid.ravel())]
    points_gdf = gpd.GeoDataFrame(geometry=points, crs=countries.crs)
 
    points_with_country = gpd.sjoin(
        points_gdf, countries[["geometry", "ADMIN"]], how="left", predicate="within")
 
    # --- Reassign genuinely unmatched grid points to the nearest country ---
    # (coastal cells whose center falls just offshore, small nations narrower
    # than one grid cell, etc. -- NOT for resolving territory ownership, which
    # is already handled by the relabeling above.)
    unmatched = points_with_country["index_right"].isna()
    n_unmatched = int(unmatched.sum())
 
    if n_unmatched:
        nearest_match = gpd.sjoin_nearest(
                        points_gdf.loc[unmatched], countries[["geometry", "ADMIN"]])
        
        nearest_match = nearest_match[~nearest_match.index.duplicated(keep="first")]
        points_with_country.loc[unmatched, "ADMIN"] = nearest_match["ADMIN"]
 
        if verbose:
            reassigned_pop = pop.values.ravel()[unmatched.values].sum()
            total_pop = pop.values.sum()
            print(f"Reassigned {n_unmatched} grid points with no direct polygon match "
                  f"to their nearest sovereign country ({reassigned_pop:.3e} people, "
                  f"{100 * reassigned_pop / total_pop:.2f}% of global population).")
 
    country_names = points_with_country["ADMIN"].values.reshape(len(lat_vals), len(lon_vals))
    country_da = xr.DataArray(country_names, 
                              dims=["lat", "lon"], 
                              coords={"lat": lat_vals, "lon": lon_vals})
 
    # --- Build the (country x gridpoint) weight matrix ---
    country_flat = country_da.values.ravel()
    weights_flat_vals = weights_2d.values.ravel()
 
    valid = pd.notna(country_flat)
    codes, countries_idx = pd.factorize(country_flat[valid])
    n_countries = len(countries_idx)
    n_points = country_flat.size
 
    cols = np.nonzero(valid)[0]
    vals = weights_flat_vals[valid]
 
    W_raw = sp.csr_matrix((vals, (codes, cols)), shape=(n_countries, n_points))
    row_sums = np.asarray(W_raw.sum(axis=1)).ravel()
 
    # --- Drop any remaining zero-weight "countries" ---
    zero_pop_mask = row_sums == 0
    dropped_zero_pop = list(countries_idx[zero_pop_mask])
 
    if zero_pop_mask.any():
        if verbose:
            print(f"Dropping {zero_pop_mask.sum()} countries with zero population "
                  f"weight: {dropped_zero_pop}")
        keep = ~zero_pop_mask
        W_raw = W_raw[keep, :]
        countries_idx = countries_idx[keep]
        row_sums = row_sums[keep]
        n_countries = len(countries_idx)
 
    if verbose:
        print(f"Smallest remaining population weight: {row_sums.min():.3e}")
 
    # --- Verify no population was lost off the edges of the country join ---
    # row_sums.sum() here is the population accounted for across all "kept" country
    # rows (dropped zero-pop rows don't cost anything, since they contributed zero).
    # Deliberately computed BEFORE the 'European Union' row is appended below, since
    # that row is a duplicate sum of existing member states rather than new population
    # -- including it here would make coverage read >100% instead of catching real gaps.
    population_total = float(np.nansum(weights_2d.values))
    population_accounted = float(row_sums.sum())
    population_coverage_frac = (
        population_accounted / population_total if population_total else np.nan)
 
    if verbose:
        print(f"Population coverage: {population_coverage_frac * 100:.6f}% of total "
              f"global population weight is accounted for across all kept country rows.")
 
    if check_coverage and abs(1.0 - population_coverage_frac) > coverage_atol:
        shortfall = population_total - population_accounted
        msg = (f"Population coverage check failed: only {population_coverage_frac * 100:.4f}% "
               f"of total global population weight ({population_accounted:.6e} of "
               f"{population_total:.6e}) is accounted for across the country rows -- "
               f"{shortfall:.6e} people-equivalent fell through the country join and the "
               f"nearest-neighbor fallback. This can happen if some `pop` grid points sit "
               f"outside the country shapefile's extent, or if population values are NaN "
               f"somewhere they shouldn't be. Pass check_coverage=False to skip this check, "
               f"or raise_on_coverage_fail=False to only warn instead of raising.")
        if raise_on_coverage_fail:
            raise ValueError(msg)
        else:
            warnings.warn(msg)
 
    W_norm = sp.diags(1.0 / row_sums) @ W_raw
 
    # --- Aggregate EU member states into one combined 'European Union' row ---
    present = set(countries_idx)
    eu_members_found = [c for c in eu_members if c in present]
    missing = [c for c in eu_members if c not in present]
 
    if verbose and missing:
        print(f"Dropping {len(missing)} EU countries with no grid coverage even "
              f"after nearest-country fallback: {missing}")
        print(f"EU aggregate will be based on the remaining {len(eu_members_found)} members.")
 
    eu_row_idx = np.nonzero(np.isin(countries_idx, eu_members_found))[0]
    eu_raw = sp.csr_matrix(W_raw[eu_row_idx, :].sum(axis=0))
    eu_norm = eu_raw / eu_raw.sum()
 
    W_norm_eu = sp.vstack([W_norm, eu_norm]).tocsr()
    countries_idx_eu = np.append(countries_idx, "European Union")
 
    return {
        "W_norm_eu": W_norm_eu,
        "countries_idx_eu": countries_idx_eu,
        "country_da": country_da,
        "weights_2d": weights_2d,
        "country_flat": country_flat,
        "weights_flat_vals": weights_flat_vals,
        "countries_gdf": countries,
        "folded_non_sovereign": folded_non_sovereign,
        "dropped_zero_pop": dropped_zero_pop,
        "population_total": population_total,
        "population_accounted": population_accounted,
        "population_coverage_frac": population_coverage_frac}

def verify_population_coverage(result, atol=1e-6, raise_on_fail=True):
    """
    Standalone re-check that every "kept" country row in a build_country_weights()
    result accounts for 100% of the total population weight in its input grid.

    Useful for a notebook-level sanity check on a `result` dict you already have
    lying around (e.g. loaded from a pickle), independent of whatever
    check_coverage/raise_on_coverage_fail settings it was originally built with.
    Recomputes coverage directly from `country_da` and `weights_2d` (per-gridcell
    country label and weight), rather than trusting build_country_weights' own
    bookkeeping, and -- by construction, since it sums weight per assigned grid
    cell rather than per matrix row -- naturally excludes the 'European Union'
    row, which is a duplicate sum of existing member states and would otherwise
    make coverage read above 100%.

    Parameters
    ----------
    result : dict
        Dict returned by build_country_weights.
    atol : float, default 1e-6
        Allowed absolute deviation of the coverage fraction from 1.0 (100%).
    raise_on_fail : bool, default True
        If True, raise AssertionError when coverage falls outside atol of 100%.
        If False, just print a warning and return the results for inspection.

    Returns
    -------
    dict with keys: population_total, population_accounted, coverage_frac, passed
    """
    weights_2d = result["weights_2d"]
    country_da = result["country_da"]

    population_total = float(np.nansum(weights_2d.values))
    population_accounted = float(
        np.nansum(weights_2d.values[country_da.notnull().values]))
    coverage_frac = (
        population_accounted / population_total if population_total else np.nan)
    passed = abs(1.0 - coverage_frac) <= atol

    msg = (f"Population coverage: {coverage_frac * 100:.6f}% "
           f"({population_accounted:.6e} of {population_total:.6e} people-equivalent) "
           f"accounted for across kept country rows.")

    if passed:
        print(f"PASSED -- {msg}")
    else:
        msg = f"FAILED -- {msg}"
        if raise_on_fail:
            raise AssertionError(msg)
        else:
            warnings.warn(msg)
            print(msg)

    return {
        "population_total": population_total,
        "population_accounted": population_accounted,
        "coverage_frac": coverage_frac,
        "passed": passed}


def aggregate_by_country(data, W_norm, countries_idx):
    """
    Apply a (country x gridpoint) weight matrix to any (time, lat, lon,
    ensemble) DataArray, e.g. f or N. Unchanged from notebooks 02/06/07 --
    kept here so it lives next to the matrix it's meant to be used with.
    """
    n_countries, n_points = W_norm.shape
    T, E = data.sizes["time"], data.sizes["ensemble"]
    data_2d = (
        data.transpose("lat", "lon", "time", "ensemble").values.reshape(n_points, T * E)
    )
    country_vals = (W_norm @ data_2d).reshape(n_countries, T, E)
    return xr.DataArray(
        country_vals,
        dims=["country", "time", "ensemble"],
        coords={
            "country": countries_idx,
            "time": data.time.values,
            "ensemble": data.ensemble.values},
        name=data.name)

country_weights_dict = build_country_weights(pop, countries_zip_path,
                keep_as_sovereign=['Palestine'], verbose=False)

W_norm_eu  = country_weights_dict['W_norm_eu']   
countries_idx_eu = country_weights_dict['countries_idx_eu']
country_da = country_weights_dict['country_da']
weights_2d = country_weights_dict['weights_2d']       
country_flat = country_weights_dict['country_flat']    
weights_flat_vals = country_weights_dict['weights_flat_vals']
folded_non_sovereign = country_weights_dict['folded_non_sovereign']
dropped_zero_pop = country_weights_dict['dropped_zero_pop']
population_total = country_weights_dict['population_total']
population_accounted = country_weights_dict['population_accounted']
population_coverage_frac = country_weights_dict['population_coverage_frac']


## Choose how EU member states are handled:
##   'aggregated' -> keep the EU-wide aggregate row, drop individual EU member countries
##   'individual' -> keep individual EU member countries, drop the EU-wide aggregate row

def select_eu_mode(ds, eu_mode, eu_label='European Union'):
    """
    Return ds restricted to the country rows appropriate for the chosen EU display mode. 
    Excluded rows are removed from the dataset to exclude them from vmax/colorbar computation.
    """
    eu_members_present = [c for c in ds.country.values if c in EU_MEMBERS]

    if eu_mode == 'aggregated': to_drop = eu_members_present
    elif eu_mode == 'individual':
        to_drop = [eu_label] if eu_label in ds.country.values else []
    else:
        raise ValueError("eu_mode must be 'aggregated' or 'individual'")
    if not to_drop:
        return ds

    ## Boolean masking to filter 'to_drop' countries
    keep_mask = ~ds.country.isin(to_drop)
    
    return ds.sel(country=keep_mask)
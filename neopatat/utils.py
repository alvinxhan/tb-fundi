# import libraries
import re
import os
import numpy as np
import scipy.stats as st
from scipy.optimize import minimize
import pandas as pd

import rasterio
from rasterio.enums import Resampling

from numba import jit, prange

def read_age_commute_dist(age_travel_fpath, district_mapping):
    print (district_mapping)
    age_travel_df = pd.read_excel(age_travel_fpath).set_index(["min_age", "max_age"])
    age_commute_f = np.zeros((len(district_mapping), 100), dtype = np.float32)
    for (min_age, max_age) in age_travel_df.index.unique():
        for district in list(age_travel_df):
            if district in district_mapping:
                age_commute_f[district_mapping[district]-1, min_age:max_age+1] = age_travel_df.loc[(min_age, max_age), district]/(max_age - min_age + 1)
    return age_commute_f

@jit(nopython = True, parallel = True, fastmath = True)
def compute_visitor_flux_prob(dist_matrix, popgrid, r = 1, fhome = 1, T = 7):
    # compute visitor flux between locations based on Schlapfer et al. 2021 (https://www.nature.com/articles/s41586-021-03480-9)
    # r = location area radius (assumed 1 km)
    # fhome = minimum frequency of visit (assumed to be home 1/day)
    # T = time period
    A = r**2 # area of location (square grid)
    mu = popgrid * A * fhome
    # compute visitor flux (Q) from i to j
    n = dist_matrix.shape[0]

    Q = np.zeros((n, n), dtype = np.float32)
    for i in prange(n):
        d_ij = dist_matrix[i,:] ** 2
        Q[i,:i] = A * (mu[:i] / d_ij[:i]) * (1/(1/T) - 1/fhome)
        Q[i,i+1:] = A * (mu[i+1:] / d_ij[i+1:]) * (1/(1/T) - 1/fhome)
        Q[i,i] = A * mu[i]
        Q[i,:] = Q[i,:]/Q[i,:].sum()
    return Q

@jit(nopython = True, parallel = True, fastmath = True)
def compute_rand_transmission(infectious_places, infectious_age, susceptible_places, susceptible_age, setting_contact_mat, infectious_age_commute_f, visitor_flux_prob, beta):

    n = susceptible_age.size
    exposed_boolean = np.zeros(n, dtype = np.uint8)

    # for each susceptible place
    for s in prange(n):
        sus_age = susceptible_age[s]
        sus_place = susceptible_places[s]

        poisson_mu = -beta * setting_contact_mat[sus_age][infectious_age] * visitor_flux_prob[sus_place][infectious_places] * infectious_age_commute_f
        prob = 1 - np.exp(poisson_mu.sum())

        if np.random.random() < prob:
            exposed_boolean[s] = 1

    return exposed_boolean

@jit(nopython = True, parallel = True)
def filter_susceptibles(infectious_places, infectious_age, susceptible_places):
    included_sus_mask = np.zeros(susceptible_places.size, dtype = np.uint8)
    unique_infectious_places = np.unique(infectious_places)
    for p in prange(unique_infectious_places.size):
        place = unique_infectious_places[p]
        included_sus_mask[susceptible_places == place] = 1
    return included_sus_mask

@jit(nopython = True, parallel = True, fastmath = True)
def compute_transmission(infectious_places, infectious_age, susceptible_places, susceptible_age, setting_contact_mat, place_n, beta):
    exposed_boolean = np.zeros(susceptible_places.size, dtype = np.uint8)
    n = susceptible_age.size
    for i in prange(n):
        sus_place = susceptible_places[i]
        sus_age = susceptible_age[i]
        inf_age_at_sus_place = infectious_age[infectious_places == sus_place]
        poisson_mu = -beta * setting_contact_mat[sus_age][inf_age_at_sus_place] * (1/place_n[sus_place])
        prob = 1 - np.exp(poisson_mu.sum())
        if np.random.random() < prob:
            exposed_boolean[i] = 1
    return exposed_boolean

def get_contact_matrix(datadir, country):
    contact_data = pd.read_csv(datadir + "/prem-et-al_synthetic_contacts_2020.csv").set_index(["iso3c", 'setting', 'location_contact']).sort_index()
    contact_mat = np.zeros((4, 100, 100), dtype=np.float32)
    for i, location in enumerate(['home', 'school', 'work', 'others']):
        country_contact_data = contact_data.loc[(country, 'overall', location)]
        country_contact_data = country_contact_data.reset_index()[['age_contactor', 'age_contactee', 'mean_number_of_contacts']]
        contact_matrix = country_contact_data.pivot(index='age_contactor', columns='age_contactee', values='mean_number_of_contacts').to_numpy().astype(np.float32)
        for agebin_x in np.arange(contact_matrix.shape[0]):
            min_age_x, max_age_x = agebin_x*5, (agebin_x*5)+5
            for agebin_y in np.arange(contact_matrix.shape[1]):
                min_age_y, max_age_y = agebin_y*5, (agebin_y*5)+5
                contact_mat[i, min_age_x:max_age_x, min_age_y:max_age_y] = contact_matrix[agebin_x,agebin_y]
    return contact_mat

#@jit(nopython=True, fastmath=True)
def assign_firms(wp_zipf_a, firms_n, employed_loc, popgrid, dist_matrix):

    # initialize
    employed_N = employed_loc.size
    employed_firms = np.zeros(employed_N, dtype=np.int32)
    employed_firms_location = np.zeros(employed_N, dtype=np.int32)
    employed_idx = np.arange(employed_N)
    potential_locs = np.uint32(np.arange(popgrid.size))
    popgrid_p = popgrid/popgrid.sum()

    curr_assigned_firms_n = np.int32(0)
    potential_max_firm_size = np.power(2, np.arange(10))

    curr_firm_id = 0
    for i, max_firm_size in enumerate(potential_max_firm_size):
        if i > 0:
            min_firm_size = potential_max_firm_size[i-1]
            size_firms_p = (1 / min_firm_size + 1)**wp_zipf_a - (1 / max_firm_size + 1)**wp_zipf_a
            size_firms_n = np.int32(np.around(size_firms_p * firms_n))
            curr_assigned_firms_n += size_firms_n
            if firms_n - curr_assigned_firms_n < 0:
                curr_assigned_firms_n -= size_firms_n
                max_firm_size = potential_max_firm_size[i-1]
                break
            # assign workplacs
            workplace_sizes = np.uint32(np.random.choice(np.arange(min_firm_size, max_firm_size), size_firms_n))
            #workplace_sizes_tot_n = np.uint32(fit_zipf_A(size_firms_n, wp_zipf_a, s0=min_firm_size))
            #print (workplace_sizes_tot_n, workplace_sizes.size)
            # get potential employees
            potential_employees = np.random.choice(employed_idx[employed_firms == 0], workplace_sizes.sum(), replace = False)
            # if one employee only
            if workplace_sizes[0] == 1:
                employed_firms[potential_employees] = -1
            else:
                # more than one employee
                # randomly choose firm locations
                workplace_locs = np.random.choice(potential_locs, size = size_firms_n, p = popgrid_p, replace = True)
                potential_employees_loc = employed_loc[potential_employees]
                chosen_workplace_idx = choose_workplaces(potential_employees_loc, workplace_locs, dist_matrix, min_firm_size, max_firm_size=max_firm_size)

                if chosen_workplace_idx[chosen_workplace_idx < 0].size > 0:
                    potential_employees = potential_employees[chosen_workplace_idx > 0]
                    chosen_workplace_idx = chosen_workplace_idx[chosen_workplace_idx > 0]

                employed_firms_location[potential_employees] = workplace_locs[chosen_workplace_idx - 1]

                unique_wp_idx, unique_wp_idx_count = np.unique(chosen_workplace_idx, return_counts = True)
                employed_firms[potential_employees] = chosen_workplace_idx + curr_firm_id

                print ("Setting up {:,} workplaces with mean size {:.1f}...".format(unique_wp_idx.size, unique_wp_idx_count.mean()))
                curr_firm_id += unique_wp_idx.max()

    # last i
    # assign workplacs
    size_firms_n = firms_n - curr_assigned_firms_n
    # get potential employees
    potential_employees = employed_idx[employed_firms == 0]
    # more than one employee
    # randomly choose firm locations
    workplace_locs = np.random.choice(potential_locs, size = size_firms_n, p = popgrid_p, replace = True)
    potential_employees_loc = employed_loc[potential_employees]
    chosen_workplace_idx = choose_workplaces(potential_employees_loc, workplace_locs, dist_matrix, max_firm_size, max_firm_size=max_firm_size*2)

    if chosen_workplace_idx[chosen_workplace_idx < 0].size > 0:
        potential_employees = potential_employees[chosen_workplace_idx > 0]
        chosen_workplace_idx = chosen_workplace_idx[chosen_workplace_idx > 0]

    employed_firms_location[potential_employees] = workplace_locs[chosen_workplace_idx - 1]

    unique_wp_idx, unique_wp_idx_count = np.unique(chosen_workplace_idx, return_counts = True)
    employed_firms[potential_employees] = chosen_workplace_idx + curr_firm_id
    print ("Setting up {:,} workplaces with mean size {:.1f}...".format(unique_wp_idx.size, unique_wp_idx_count.mean()))

    # unassigned individuals assumed to be their own employer
    if employed_idx[employed_firms == 0].size > 0:
        employed_firms[employed_firms == 0] = -1
    # so are singly-assiged workplaces
    unique_wp_idx, unique_wp_idx_count = np.unique(employed_firms, return_counts=True)
    employed_firms[np.isin(employed_firms, unique_wp_idx[unique_wp_idx_count == 1])] = -1

    # reset firm ids
    unique_firm_ids = np.unique(employed_firms[employed_firms>0])
    new_firm_ids = np.uint32(np.arange(unique_firm_ids.size)) + 1
    mapping_arr = np.zeros(unique_firm_ids.max() + 1, dtype=np.uint32)
    mapping_arr[unique_firm_ids] = new_firm_ids
    employed_firms[employed_firms>0] = mapping_arr[employed_firms[employed_firms>0]]

    print ('Average workplace sizes of all employed = %.2f.'%(np.unique(employed_firms, return_counts = True)[1].mean()))
    return employed_firms, employed_firms_location

@jit(nopython=True, parallel=True, fastmath=True)
def choose_workplaces(employees_loc, workplace_locs, dist_matrix, min_firm_size, max_firm_size=1000):
    n = employees_loc.size
    potential_wp_idx = np.arange(workplace_locs.size) + 1
    potential_wp_count = np.zeros(workplace_locs.size, dtype=np.uint32)
    chosen_wp_idx = np.zeros(n, dtype=np.int32) - 1

    for s in prange(n):
        sloc = employees_loc[s]
        wpdist_to_sloc = dist_matrix[sloc,:][workplace_locs]
        for max_dist in [25, 50]:
            potential_wp_to_s = potential_wp_idx[(wpdist_to_sloc<=max_dist)]
            # prefer those that havent reach minimum firm size
            wp_to_choose = potential_wp_to_s[potential_wp_count[potential_wp_to_s-1] < min_firm_size]
            if wp_to_choose.size == 0:
                wp_to_choose = potential_wp_to_s[:]
            if wp_to_choose.size > 0:
                chosen_wp = np.random.choice(wp_to_choose)
                chosen_wp_idx[s] = chosen_wp
                potential_wp_count[chosen_wp-1] += 1
                break

    return chosen_wp_idx

def minobj_zipf_fn(A, N, alpha, s0):
    n = (alpha - 1)/(alpha) * (((s0/A)**alpha - 1)/((s0/A)**alpha - (s0/A)))
    return abs(N - n)

def fit_zipf_A(N, alpha, s0=1, x0=1000):
    result = minimize(minobj_zipf_fn, x0=x0, args=(N, alpha, s0), method='Nelder-Mead', bounds=[(1e-6, 1e9)])
    print (result)
    if result.success:
        return result.x[0]
    else:
        print (result)
        raise Exception

@jit(nopython=True, parallel=True, fastmath=True)
def choose_schools(students_locations, potential_school_locations, dist_matrix):
    n = students_locations.size
    potential_school_idx = np.arange(potential_school_locations.size) + 1
    chosen_school_idx = np.zeros(n, dtype=np.int32) - 1
    for s in prange(n):
        sloc = students_locations[s]
        schdist_to_sloc = dist_matrix[sloc,:][potential_school_locations]
        for max_dist in [10, 20, 30]:
            potential_schools_to_s = potential_school_idx[schdist_to_sloc<=max_dist]
            if potential_schools_to_s.size > 0:
                chosen_school_idx[s] = np.random.choice(potential_schools_to_s)
                break
    return chosen_school_idx

def get_edu_pars(datadir, country):
    # read education dataframes
    sch_enrol_rates_df = pd.read_csv(datadir + "/UIS_school_enrollment_rates.csv")
    sch_entrance_age_df = pd.read_csv(datadir + "/UIS_school_entrance_age.csv")
    sch_duration_df = pd.read_csv(datadir + "/UIS_school_duration.csv")

    # get enrollment rate for country
    sch_enrol_rates_df = sch_enrol_rates_df[(sch_enrol_rates_df['LOCATION']==country)]
    sch_enrol_rates_df = sch_enrol_rates_df.dropna(subset=["Value"])
    sch_enrol_rates_df = sch_enrol_rates_df.set_index("Indicator")
    sch_enrol_rates = {}

    for idx in sch_enrol_rates_df.index.unique():
        row = sch_enrol_rates_df.loc[idx]
        if isinstance(row, pd.Series):
            row = pd.DataFrame(row).T
        if re.search("primary", idx, re.I):
            sch_enrol_rates[0] = np.float32(row[row['Time']==row['Time'].max()]['Value'].iloc[0]/100)
        elif re.search("lower secondary", idx, re.I):
            sch_enrol_rates[1] = np.float32(row[row['Time']==row['Time'].max()]['Value'].iloc[0]/100)
        elif re.search("upper secondary", idx, re.I):
            sch_enrol_rates[2] = np.float32(row[row['Time']==row['Time'].max()]['Value'].iloc[0]/100)

    # get enrollment age
    sch_entrance_age_df = sch_entrance_age_df[(sch_entrance_age_df['LOCATION']==country)]
    sch_entrance_age_df = sch_entrance_age_df.dropna(subset=["Value"])
    sch_entrance_age_df = sch_entrance_age_df.set_index("Indicator")
    sch_enrol_age = {}
    for idx in sch_entrance_age_df.index.unique():
        row = sch_entrance_age_df.loc[idx]
        if re.search("primary", idx, re.I):
            sch_enrol_age[0] = np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])
        elif re.search("lower secondary", idx, re.I):
            sch_enrol_age[1] = np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])
        elif re.search("upper secondary", idx, re.I):
            sch_enrol_age[2] = np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])

    # get final year age
    sch_duration_df = sch_duration_df[(sch_duration_df['LOCATION']==country)]
    sch_duration_df = sch_duration_df.dropna(subset=["Value"])
    sch_duration_df = sch_duration_df.set_index("Indicator")
    for idx in sch_duration_df.index.unique():
        row = sch_duration_df.loc[idx]
        if re.search("primary", idx, re.I):
            sch_enrol_age[0] = [sch_enrol_age[0], sch_enrol_age[0]+np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])-1]
        elif re.search("lower secondary", idx, re.I):
            sch_enrol_age[1] = [sch_enrol_age[1], sch_enrol_age[1]+np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])-1]
        elif re.search("upper secondary", idx, re.I):
            sch_enrol_age[2] = [sch_enrol_age[2], sch_enrol_age[2]+np.int32(row[row['Time']==row['Time'].max()]['Value'].iloc[0])-1]

    return sch_enrol_rates, sch_enrol_age

def distribute_randint(N, min_int, max_arr):
    # each element should minimally has min_int
    arr = np.zeros(max_arr.size, dtype=np.int32) + min_int
    # compute total of remaining values to add
    leftover = N - arr.sum()
    if leftover > 0:
        # generate idx
        idx = np.int32(np.arange(arr.size))
        repeat_dummy = np.repeat(idx, max_arr - arr)
        dummy_idx = np.int32(np.arange(repeat_dummy.size))
        # select repeated idx randomly
        selected_idx = np.random.choice(dummy_idx, leftover, replace=False)
        # add counts of repeated idx from arr
        unq_idx, unq_counts = np.unique(repeat_dummy[selected_idx], return_counts=True)
        arr[unq_idx] = arr[unq_idx] + unq_counts
    return arr

def generate_poisson_rv(mu, N, s):
    arr_idx = np.int32(np.arange(s))
    # generate s number of poisson RVs
    arr = np.int32(np.random.poisson(lam = mu, size = s))
    ones_n = arr[arr <= 1].size
    # replace all size of one with greater sizes
    while ones_n > 0:
        arr[arr <= 1] = np.random.poisson(lam = mu, size = ones_n)
        ones_n = arr[arr <= 1].size

    diff = arr.sum() - N
    if diff < 0:
        raise Exception('fuku')

    # calculate different from target N
    if diff > 0:
        more_than_two_n = arr[arr>2].size
        if more_than_two_n > diff:
            # randomly select households to minus one
            idx_to_minus_one = np.random.choice(arr_idx[arr > 2], diff, replace = False)
            arr[idx_to_minus_one] -= 1
            diff = arr.sum() - N
        else:
            raise Exception('fook')
    return arr

@jit(nopython=True)
def haversine_d(latlon1, latlon2, R=6371):
    lat1, lon1 = np.radians(latlon1)
    latlon2 = np.radians(latlon2)
    lat2 = latlon2[:,0]
    lon2 = latlon2[:,1]
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

@jit(nopython=True, parallel=True, fastmath=True)
def get_dist_matrix(latlon):
    n = latlon.shape[0]
    dist_matrix = np.zeros((n, n), dtype=np.float32)
    for i in prange(n):
        i_coords = latlon[i]
        # compute haversine distance and kernel
        d = haversine_d(i_coords, latlon)
        dist_matrix[i,:] = d
    return dist_matrix

def get_popgrid(datadir, place, year):
    # get population grid and lonlat
    popgrid, latlon = read_geotiff(datadir + "/worldpop/zaf_ppp_%i_1km_Aggregated.tif"%(year))
    # read admin id
    admin_grid = read_admin_geotiff(datadir + "/worldpop/zaf_subnational_admin_2000_2020.tif", popgrid.shape[0], popgrid.shape[1])
    # read unique admin id of place
    admin_id_arr = np.load(datadir + "/%s_admin_ids.npz"%(place))
    unique_admin_ids = []
    for district in admin_id_arr.files:
        unique_admin_ids += list(admin_id_arr[district])
    unique_admin_ids = np.asarray(unique_admin_ids)
    # filter popgrid for place
    place_mask = np.isin(admin_grid, unique_admin_ids)
    popgrid_coords = np.argwhere(place_mask)
    popgrid, latlon = popgrid[place_mask], latlon[place_mask]
    # generate district grid
    admin_grid = admin_grid[place_mask]
    district_grid = np.zeros(admin_grid.size, dtype=np.uint16)
    district_mapping = {}
    for d, district in enumerate(admin_id_arr.files):
        district_grid[np.isin(admin_grid, admin_id_arr[district])] = d + 1
        district_mapping[district] = d + 1
    return popgrid[popgrid>0], popgrid_coords[popgrid>0], latlon[popgrid>0], district_grid[popgrid>0], district_mapping, {v:k for k, v in district_mapping.items()}

def read_admin_geotiff(fpath, rescale_height, rescale_width):
    with rasterio.open(fpath) as src:
        # resample data to target shape
        data = src.read(
            out_shape=(
                src.count,
                rescale_height,
                rescale_width
            ),
            # resample by nearest value
            resampling=Resampling.nearest
        )

        # get metadata
        metadata = src.meta
        # zero no data
        mask = data==metadata['nodata']
        data[mask] = 0.
    return data[0]

def read_geotiff(fpath):
    """
    Read geotiff file and return population density grid
    """
    with rasterio.open(fpath) as src:
        # read data as numpy array (bands x rows x columns)
        data = src.read()[0]
        # get metadata
        metadata = src.meta
        # zero no data
        mask = data==metadata['nodata']
        data[mask] = 0.
        # remove any grid < 1
        data[data<1] = 0.
        # integer-ize
        data = np.around(data,0).astype(np.int32)

        # get latlon data
        latlon = np.zeros((data.shape[0], data.shape[-1], 2), dtype=np.float32)
        height = data.shape[0]
        width = data.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(src.transform, rows, cols)
        lons= np.array(xs)
        lats = np.array(ys)
        latlon[:,:,0] = lats
        latlon[:,:,1] = lons

    return data, latlon

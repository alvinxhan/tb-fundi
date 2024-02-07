# import libraries
import re
import os
import numpy as np
import sciris as sc
import pandas as pd
import time
from . import utils

# Sim object
class Sim():
    """
    Simulation object
    """
    def __init__(self, pars=None, pop=None):
        # read population object
        self.people = pop.people
        self.dist_matrix = pop.dist_matrix
        self.district_grid = pop.district_grid
        self.visitor_flux_prob = pop.visitor_flux_prob
        self.district_mapping = pop.district_mapping

        # compute number of individuals in each place
        # households
        id, id_count = np.unique(self.people.household, return_counts = True)
        self.household_n = np.zeros(id.max()+1, dtype=np.uint32)
        self.household_n[id] = id_count
        # place
        id, id_count = np.unique(self.people.swsplace, return_counts = True)
        self.swsplace_n = np.zeros(id.max()+1, dtype=np.uint32)
        self.swsplace_n[id] = id_count

        # get contact matrix (mean number of contacts per week)
        self.contact_mat = utils.get_contact_matrix(pars.datadir, "ZAF")
        # read age-based commuting fraction
        self.age_commute_f = utils.read_age_commute_dist(pars.age_travel_fpath, self.district_mapping)

        # assign HIV status
        self.assign_hiv_status(pars)
        # read state transition rates
        self.get_state_transition_rates(pars)

        # initialize
        # setup epidemic object
        self.epidemic = sc.objdict()
        # probability of infection per contact
        self.epidemic.beta = pars.beta
        # length of simulated epidemic
        self.epidemic.nweeks = pars.nweeks
        # external attack rate (per person per week)
        self.epidemic.ext_attack_rate = pars.ext_attack_rate
        # current state of each person (0 = susceptible, 1 = Infection, 2 = Minimal, 3 = Subclincal, 4 = Clinical, 5 = Recovered, 6 = Death, based on Horton et al., PNAS 2023)
        self.epidemic.curr_state = np.zeros(self.people.N, dtype=np.uint8)
        # start day of current state
        self.epidemic.curr_state_st = np.zeros(self.people.N, dtype=np.int32)
        # epidemic results
        self.epidemic.result = np.zeros((pars.nweeks+1, self.people.N), dtype=np.uint8)

        # initialize
        self.t = np.uint32(0)
        # assign initial states
        self.assign_initial_states(pars.initial_state_dist, pars.initial_state_tau, pars.district_mptb_prevalence)

        return

    def simulate(self, death_bool=1):
        """
        Simulate epidemic
        """
        print ("Starting simulation...")
        print (self.t, np.unique(self.epidemic.curr_state, return_counts = True))

        while self.t < self.epidemic.nweeks + 1:
            if self.epidemic.ext_attack_rate > 0:
                # external attack
                self.external_attack()
            # update epidemic today
            self.update_epidemic(death_bool)
            # transmission
            self.transmission("household")
            self.transmission("school")
            self.transmission("workplace")
            self.transmission("community")
            # save result
            self.epidemic.result[self.t,:] = self.epidemic.curr_state[:]

            year = np.int32(np.floor(self.t/52))
            print (year, self.t, np.unique(self.epidemic.curr_state, return_counts = True))
            # update week
            self.t += 1

    def assign_initial_states(self, initial_state_dist, initial_state_tau, district_mptb_prevalence):
        for state in range(1, 6):
            state_p = initial_state_dist[state-1]
            state_tau_mu, state_tau_std = initial_state_tau[state-1]
            state_n = np.uint32(self.people.N * state_p)

            if state == 4: # clinical
                # divide by ratio of prevalence by district
                district_prev_f = np.asarray(district_mptb_prevalence) / 1e5 * np.unique(self.people.district, return_counts=True)[-1]
                district_prev_f = np.float32(district_prev_f/district_prev_f.sum())
                for di, dif in enumerate(district_prev_f):
                    district_people_id = self.people.id[self.people.district == di+1]
                    district_state_n = np.int32(state_n * dif)

                    potential_people = district_people_id[self.epidemic.curr_state[district_people_id] == 0]
                    people_in_state = np.random.choice(potential_people, district_state_n, replace=False)
                    self.epidemic.curr_state[people_in_state] = np.uint8(state)
                    self.epidemic.curr_state_st[people_in_state] = np.around(np.absolute(np.random.normal(loc=state_tau_mu, scale=state_tau_std, size=district_state_n)))

            else:
                # infection, minimal (i.e. latent), subclinical recovered  are chosen at random across population
                potential_people = self.people.id[self.epidemic.curr_state == 0]
                people_in_state = np.random.choice(potential_people, state_n, replace=False)
                self.epidemic.curr_state[people_in_state] = np.uint8(state)
                self.epidemic.curr_state_st[people_in_state] = np.around(np.absolute(np.random.normal(loc=state_tau_mu, scale=state_tau_std, size=state_n)))

        self.epidemic.curr_state_st[self.epidemic.curr_state_st > 0] = -self.epidemic.curr_state_st[self.epidemic.curr_state_st > 0]
        return

    def transmission(self, setting):
        # get all infectious individuals, their households and places
        infectious_inds = self.people.id[(self.epidemic.curr_state>=3)&(self.epidemic.curr_state<5)]
        if setting == 'school':
            infectious_inds = infectious_inds[(self.people.swstatus[infectious_inds] >= 1)&(self.people.swstatus[infectious_inds] < 3)]
        elif setting == 'workplace':
            infectious_inds = infectious_inds[(self.people.swstatus[infectious_inds] == 3)]

        if infectious_inds.size == 0:
            return

        # get susceptibles
        susceptible_inds = self.people.id[self.epidemic.curr_state == 0]
        if setting == 'school':
            susceptible_inds = susceptible_inds[(self.people.swstatus[susceptible_inds] >= 1)&(self.people.swstatus[susceptible_inds] < 3)]
        elif setting == 'workplace':
            susceptible_inds = susceptible_inds[(self.people.swstatus[susceptible_inds] == 3)]

        # get setting contact matrix
        if setting == 'household':
            infectious_places = self.people.household[infectious_inds]
            susceptible_places = self.people.household[susceptible_inds]
            place_n = self.household_n
            setting_contact_mat = self.contact_mat[0] * 7

        elif setting == 'community':
            infectious_places = self.people.location[infectious_inds]
            susceptible_places = self.people.location[susceptible_inds]
            setting_contact_mat = self.contact_mat[3] * 7

        else:
            infectious_places = self.people.swsplace[infectious_inds]
            susceptible_places = self.people.swsplace[susceptible_inds]
            place_n = self.swsplace_n

            if setting == 'school':
                setting_contact_mat = self.contact_mat[1] * 5
            else:
                setting_contact_mat = self.contact_mat[2] * 5

        # get ages of infectious and susceptible inds
        infectious_age = self.people.age[infectious_inds]

        if setting == 'community':
            susceptible_age = self.people.age[susceptible_inds]
            # get likelihood of commuting depending on age for each district
            infectious_age_commute_f = self.age_commute_f[self.people.district[infectious_inds] - 1, infectious_age]
            # compute transmissions
            exposed_boolean = utils.compute_rand_transmission(infectious_places, infectious_age, susceptible_places, susceptible_age, setting_contact_mat, infectious_age_commute_f, self.visitor_flux_prob, self.epidemic.beta)

        else:
            # compute shortlisted susceptibles who are in the same swsplace as infectious individuals
            included_sus_mask = utils.filter_susceptibles(infectious_places, infectious_age, susceptible_places)
            susceptible_inds = susceptible_inds[included_sus_mask > 0]
            if susceptible_inds.size == 0:
                return
            susceptible_places = susceptible_places[included_sus_mask > 0]
            susceptible_age = self.people.age[susceptible_inds]

            # compute transmissions
            exposed_boolean = utils.compute_transmission(infectious_places, infectious_age, susceptible_places, susceptible_age, setting_contact_mat, place_n, self.epidemic.beta)

        exposed_persons = susceptible_inds[exposed_boolean > 0]
        self.exposed(exposed_persons)

        return

    def update_epidemic(self, death_bool):
        for state in [1, 2, 3, 4, 5]: # curr state infection, minimal, subclinical, clinical, recovered
            # get individuals in current state
            inds_in_curr_state = self.people.id[self.epidemic.curr_state == state]

            if inds_in_curr_state.size > 0:
                # get time since start of current state
                st_of_curr_state = self.epidemic.curr_state_st[inds_in_curr_state]
                state_delta_t = self.t - st_of_curr_state
                next_state_of_inds = np.zeros(inds_in_curr_state.size, dtype=np.uint8) + state

                transition_states = np.argwhere(self.state_transtion_rates[state] > 0).T[0] # get states to transition to

                for tstate in transition_states:
                    if death_bool < 1 and tstate == 6:
                        continue
                    rate = self.state_transtion_rates[state, tstate]
                    next_state_p = 1 - np.exp(-rate * state_delta_t)
                    next_state_of_inds[(np.random.random(next_state_p.size) < next_state_p)&(next_state_of_inds == state)] = tstate

                # change state
                inds_changing_state = inds_in_curr_state[next_state_of_inds != state]
                if inds_changing_state.size > 0:
                    next_states = next_state_of_inds[next_state_of_inds != state]
                    self.epidemic.curr_state[inds_changing_state] = next_states
                    self.epidemic.curr_state_st[inds_changing_state] = self.t

        return

    def exposed(self, exposed_persons):
        # change current state of exposed person to latent
        self.epidemic.curr_state[exposed_persons] = 1
        self.epidemic.curr_state_st[exposed_persons] = self.t
        return

    def external_attack(self):
        # randomly select persons infected by overseas introduction
        candidate_persons = self.people.id[self.epidemic.curr_state == 0]
        sus_n = candidate_persons.size
        if sus_n == 0:
            # no candidate person to infect
            return
        # calculate infection likelihood; overseas attack rate should be weighted by overall susceptiblity of the population
        infection_likelihood = 1 - np.exp(-self.epidemic.ext_attack_rate * (sus_n/self.people.N))
        # infect
        exposed_persons = candidate_persons[np.random.random(candidate_persons.size) < infection_likelihood]
        if exposed_persons.size == 0:
            return
        # plan infection of infected person
        self.exposed(exposed_persons)
        return

    def get_state_transition_rates(self, pars):
        df = pd.read_excel(pars.datadir + "/rate_params.xlsx").set_index(["state_i", "state_j"])
        self.state_transtion_rates = np.zeros((7, 7), dtype=np.float32)
        for (i, j) in df.index.unique():
            self.state_transtion_rates[i, j] = df.loc[(i,j), "rate_per_year"]/52.1429

    def assign_hiv_status(self, pars):
        # assign HIV status (adults)
        self.people.hiv_status = np.zeros(self.people.N, dtype=np.uint8)
        adults_id = self.people.id[self.people.age >= 15]
        adults_n = np.uint32(pars.hiv_prev_pars['adult_prev'] * adults_id.size)
        self.people.hiv_status[np.random.choice(adults_id, adults_n, replace=False)] = 1
        # assign HIV status to children
        children_id = self.people.id[self.people.age < 15]
        children_n = np.uint32(pars.hiv_prev_pars['child_adult_ratio'] * adults_n)
        print ("{:,} adults (15+ y) and {:,} children (<15y) ({:.1f}% of population) are living with HIV".format(adults_n, children_n, 100*(adults_n+children_n)/self.people.N))
        self.people.hiv_status[np.random.choice(children_id, children_n, replace=False)] = 1


# Population object
class Pop():
    """
    Population object
    """
    def __init__(self, pars=None):
        if pars.verbose > 0:
            print ("Creating population object for '{:}'.".format(pars.place.capitalize()))
        # read popgrid over the years
        popgrid, popgrid_coords, latlon, district_grid, self.district_mapping, inv_district_mapping = utils.get_popgrid(pars.datadir, pars.place, pars.year)

        # initialise people
        self.people = sc.objdict()

        # setup age of individuals
        other_demography_info = self.setup_age(pars, popgrid, district_grid)

        # setup households
        # initialize
        loc_idx_arr = np.uint32(np.arange(popgrid.size))
        self.people.household = np.zeros(self.people.N, dtype=np.uint32)
        self.people.location = np.zeros(self.people.N, dtype=np.int32) - 1
        prev_hh_id = 0

        for district_idx in np.unique(district_grid):
            # for each district
            district = inv_district_mapping[district_idx]
            # identify individuals living in district
            people_id_in_district = self.people.id[self.people.district == district_idx]
            # assign them a household
            average_hh_size = other_demography_info.loc[district, 'average_hh_size']
            self.setup_households(pars, people_id_in_district, average_hh_size)
            # ensure that household ids are unique
            district_people_hh_id = self.people.household[people_id_in_district][:]
            self.people.household[people_id_in_district] += prev_hh_id
            prev_hh_id = self.people.household[people_id_in_district].max()
            unq_hh_id, unq_hh_counts = np.unique(self.people.household[people_id_in_district], return_counts=True)
            average_hh_size = unq_hh_counts.mean()
            if pars.verbose > 0:
                print ("Setting up households (id = {:,} - {:,}; mean size = {:.1f}) in {:}...".format(unq_hh_id.min(), unq_hh_id.max(), average_hh_size, district))
            # randomly assign households to location
            # calculate likelihood of at each location (popgrid/average hh size)
            loc_hh_based_p = np.float32(popgrid[district_grid == district_idx]/average_hh_size)
            loc_hh_based_p = loc_hh_based_p/loc_hh_based_p.sum()
            district_loc_idx_arr = loc_idx_arr[district_grid == district_idx]
            # randomly choose based on likelihoods
            rand_hh_loc_arr = np.uint32(np.random.choice(district_loc_idx_arr, size = unq_hh_id.size, p = loc_hh_based_p))
            self.people.location[people_id_in_district] = rand_hh_loc_arr[district_people_hh_id - 1]

        # reset location indices to only retain populated locations
        unique_populated_locs, self.popgrid = np.unique(self.people.location, return_counts=True)
        self.popgrid_coords = popgrid_coords[unique_populated_locs]
        loc_idx_arr = loc_idx_arr[unique_populated_locs]
        self.latlon = latlon[loc_idx_arr,:]
        self.district_grid = district_grid[loc_idx_arr]
        self.loc_idx_arr = np.uint32(np.arange(unique_populated_locs.size))
        loc_idx_mapping = np.zeros(unique_populated_locs.max()+1, dtype=np.uint32)
        loc_idx_mapping[unique_populated_locs] = self.loc_idx_arr
        self.people.location = loc_idx_mapping[self.people.location]

        # compute distance-based commuting kernel between locations
        self.dist_matrix = utils.get_dist_matrix(self.latlon)

        # compute visitor flux between locations based on Schlapfer et al. 2021 (https://www.nature.com/articles/s41586-021-03480-9)
        self.visitor_flux_prob = utils.compute_visitor_flux_prob(self.dist_matrix, self.popgrid)

        # setup schooling children and working adults
        # initialize
        self.people.swstatus = np.zeros(self.people.N, dtype=np.uint8)
        self.people.swsplace = np.zeros(self.people.N, dtype=np.uint32)
        self.people.swslocation = np.zeros(self.people.N, dtype=np.uint32)
        self.setup_schools(pars, inv_district_mapping, other_demography_info)
        self.setup_workplaces(pars)

        print ("...done.")

    def setup_workplaces(self, pars):
        # get employed individuals and their locations
        employed_inds = self.people.id[self.people.swstatus == 3]
        employed_n = employed_inds.size
        employed_loc = self.people.location[employed_inds]

        # get expected number of firms
        firms_n = ((pars.wp_zipf_a - 1)/pars.wp_zipf_a) * ( ((1/employed_inds.size)**pars.wp_zipf_a - 1) / ((1/employed_inds.size)**pars.wp_zipf_a - (1/employed_inds.size)) )
        firms_n = np.uint32(firms_n)

        employed_firms, employed_firms_location = utils.assign_firms(pars.wp_zipf_a, firms_n, employed_loc, self.popgrid, self.dist_matrix)
        # remove single-employer employed individuals
        single_employee_inds = employed_inds[employed_firms<0]
        single_firm_n = single_employee_inds.size
        accounting_n = 0
        print ("\n{:,} of {:,} employed individuals are in single-person firms.".format(single_firm_n, employed_n))
        self.people.swstatus[single_employee_inds] = 4
        accounting_n += single_firm_n

        employed_inds = employed_inds[employed_firms > 0]
        employed_firms_location = employed_firms_location[employed_firms > 0]
        employed_firms = employed_firms[employed_firms > 0]

        unq_firm_ids, unq_firm_counts = np.unique(employed_firms, return_counts = True)
        for (min_n, max_n) in [(2, 10), (11, 50), (51, 250), (251, 1000)]:
            if min_n == 251:
                satisfied_ids = unq_firm_ids[(unq_firm_counts >= min_n)]
                multi_firm_n = employed_inds[np.isin(employed_firms, satisfied_ids)].size
                print ("{:,} employed individuals are in {:,} firms of size > {:,}.".format(multi_firm_n, satisfied_ids.size, min_n-1))
            else:
                satisfied_ids = unq_firm_ids[(unq_firm_counts >= min_n)&(unq_firm_counts <= max_n)]
                multi_firm_n = employed_inds[np.isin(employed_firms, satisfied_ids)].size
                print ("{:,} employed individuals are in {:,} firms of size {:,} - {:,}.".format(multi_firm_n, satisfied_ids.size, min_n, max_n))
            accounting_n += multi_firm_n

        self.people.swsplace[employed_inds] = employed_firms + self.people.swsplace.max()
        self.people.swslocation[employed_inds] = employed_firms_location

    def setup_schools(self, pars, inv_district_mapping, other_demography_info):
        school_location_count = np.zeros((self.loc_idx_arr.size, 2), dtype=np.uint32)

        for district_idx in np.unique(self.district_grid):
            # for each district
            district = inv_district_mapping[district_idx]
            # identify individuals living in district
            people_id_in_district = self.people.id[self.people.district == district_idx]
            # employment rate
            employment_rate = other_demography_info.loc[district, 'employment_rate']
            self.setup_non_hh_contact_status(pars, people_id_in_district, employment_rate)

            # setup school locations
            school_n_arr = np.asarray([other_demography_info.loc[district, "school_n_primary"], other_demography_info.loc[district, "school_n_secondary"]])
            # randomly select school locations based on population density
            district_popgrid = self.popgrid[self.district_grid == district_idx]
            district_loc_idx_arr = self.loc_idx_arr[self.district_grid == district_idx]
            for swstatus in range(2):
                selected_loc_idx = np.random.choice(district_loc_idx_arr, size=school_n_arr[swstatus], replace=True, p=district_popgrid/district_popgrid.sum())
                loc_idx, loc_counts = np.unique(selected_loc_idx, return_counts=True)
                school_location_count[loc_idx, swstatus] += np.uint32(loc_counts)

        # link students by schools based on distance
        prev_max = 0
        for swstatus in range(2):
            students_id = self.people.id[self.people.swstatus == swstatus+1]
            students_locations = self.people.location[students_id]
            potential_school_locations = np.repeat(self.loc_idx_arr[school_location_count[:,swstatus]>0], school_location_count[:,swstatus][school_location_count[:,swstatus]>0])
            chosen_sch_idx = utils.choose_schools(students_locations, potential_school_locations, self.dist_matrix)
            # reset those could not be assigned schools
            if students_id[chosen_sch_idx < 0].size > 0:
                self.people.swstatus[students_id[chosen_sch_idx < 0]] = 0

            students_id = students_id[chosen_sch_idx > 0]
            chosen_sch_idx = chosen_sch_idx[chosen_sch_idx > 0]
            self.people.swslocation[students_id] = potential_school_locations[chosen_sch_idx-1]

            chosen_sch_idx = chosen_sch_idx + prev_max
            if pars.verbose > 0:
                sch_sizes = np.unique(chosen_sch_idx, return_counts=True)[1]
                print ("Setting up {:,} {:} schools (average size = {:,} learners)".format(sch_sizes.size, "secondary" if swstatus+1 > 1 else "primary", np.int32(sch_sizes.mean())))
            self.people.swsplace[students_id] = chosen_sch_idx

            prev_max = chosen_sch_idx.max()

    def setup_non_hh_contact_status(self, pars, people_id_in_district, employment_rate):
        """
        Setup non-household contact statuses
        """
        # setup schooling kids
        # get school information arrays
        sch_enrol_rates, sch_enrol_age = utils.get_edu_pars(pars.datadir, country="ZAF")

        # setup schooling statuses
        for sch_status, sch_rate in sch_enrol_rates.items():
            min_age, max_age = sch_enrol_age[sch_status]
            # randomly select children for schooling
            children_idx = people_id_in_district[(self.people.age[people_id_in_district]>=min_age)&(self.people.age[people_id_in_district]<=max_age)]
            # schooling kids
            n = np.uint32(np.around(children_idx.size * sch_rate))
            schooling_children_idx = np.random.choice(children_idx, size=n, replace=False)
            # setup school status
            if sch_status > 0:
                self.people.swstatus[schooling_children_idx] = 2 # secondary
            else:
                self.people.swstatus[schooling_children_idx] = sch_status + 1 # primary

        # setup employed individuals
        employable_idx = people_id_in_district[(self.people.swstatus[people_id_in_district] == 0)&(self.people.age[people_id_in_district]>=15)&(self.people.age[people_id_in_district]<=65)]
        n = np.uint32(np.around(employable_idx.size * employment_rate))
        self.people.swstatus[np.random.choice(employable_idx, n, replace=False)] = 3

    def setup_households(self, pars, people_id_in_district, average_hh_size):

        # read household composition data from UN
        hh_df = pd.read_excel(pars.datadir + '/undesa_pd_2022_hh-size-composition.xlsx', na_values='..')
        hh_df = hh_df[hh_df['ISO Code'] == 710] # fixed to south africa
        hh_df = hh_df[hh_df['Reference date (dd/mm/yyyy)'] == hh_df['Reference date (dd/mm/yyyy)'].max()]

        # get household parameters
        one_member_hh_p, head_18_19y_p, head_20_64y_p, head_over_64y_p, member_under_15y_p, member_over_64y_p = np.float32(hh_df[["1 member", "head_18-19y", "head_20-64y", "head_over_64y", "member_under_15y", "member_over_64y"]].to_numpy()[0])

        # probability of household head based on age
        head_age_p = np.zeros(self.people.age[people_id_in_district].max()+1, dtype=np.float32)
        if np.isnan(head_18_19y_p) or np.isnan(head_20_64y_p) or np.isnan(head_over_64y_p):
            # uniform probability across all ages
            head_age_p[18:] = 1.
        else:
            # data-based probabilities
            head_age_p[18:20] = head_18_19y_p
            head_age_p[20:65] = head_20_64y_p
            head_age_p[65:] = head_over_64y_p

        # likelihood of household with at least one x years member
        member_under_15y_p = np.float32(member_under_15y_p/100)
        member_over_64y_p = np.float32(member_over_64y_p/100)

        # calculate average number of household members for those with > 1 member in household
        one_member_hh_p = np.float32(one_member_hh_p/100)
        average_multimem_hh_n = (average_hh_size - one_member_hh_p)/(1 - one_member_hh_p)

        # calculate expected number of households
        expected_hh_n = np.int32(people_id_in_district.size/average_hh_size)
        # choose household head for expected_hh_n households
        potential_hh_head = people_id_in_district[self.people.age[people_id_in_district] >= 18]
        potential_hh_age = self.people.age[potential_hh_head]
        p_arr = head_age_p[potential_hh_age]
        p_arr /= p_arr.sum()
        chosen_hh_heads = np.random.choice(potential_hh_head, expected_hh_n, p=p_arr, replace=False)
        # assign household ids to chosen_hh_heads
        hh_id_arr = np.int32(np.arange(1, expected_hh_n+1))
        self.people.household[chosen_hh_heads] = hh_id_arr

        # get household ids with multiple members
        multimem_hh_mask = np.random.random(expected_hh_n) >= one_member_hh_p
        multimem_hh_id_arr = hh_id_arr[multimem_hh_mask]
        idx_arr = np.int32(np.arange(multimem_hh_id_arr.size))

        # calculate number of homeless people left
        n_homeless = people_id_in_district.size - expected_hh_n
        multimem_hh_size_arr = utils.generate_poisson_rv(average_multimem_hh_n, n_homeless + multimem_hh_id_arr.size, multimem_hh_id_arr.size)
        multimem_hh_size_arr -= 1 # remove the household head count

        if not pd.isna(member_under_15y_p):
            """if pars.verbose > 0:
                print ("Assigning households to children < 15 years...")"""
            # choose household idx with at least one children < 15y
            potential_people_mask = self.people.age[people_id_in_district] < 15
            potential_people = people_id_in_district[potential_people_mask]
            potential_people_n = potential_people.size

            selected_idx = idx_arr[np.random.random(multimem_hh_id_arr.size) < member_under_15y_p]
            # number of households should be more than potential_people_n
            if selected_idx.size > potential_people_n:
                selected_idx = np.random.choice(selected_idx, potential_people_n, replace=False)
            # number of people in multimember households should not be less than potential persons to fill those households
            if multimem_hh_size_arr[selected_idx].sum() < potential_people_n:
                raise Exception("total members in multi-member households less than potential persons that reside in these households")
            # randomly distribute potential_people_n to selected households
            n_in_selected_idx = utils.distribute_randint(N=potential_people_n, min_int=1, max_arr=multimem_hh_size_arr[selected_idx])
            # assign household id
            hh_id_for_potential_people = multimem_hh_id_arr[np.repeat(selected_idx, n_in_selected_idx)]
            np.random.shuffle(hh_id_for_potential_people)
            self.people.household[potential_people] = hh_id_for_potential_people

            # remove completed multimember households
            multimem_hh_size_arr[selected_idx] = multimem_hh_size_arr[selected_idx] - n_in_selected_idx
            multimem_hh_id_arr = multimem_hh_id_arr[multimem_hh_size_arr > 0]
            idx_arr = np.int32(np.arange(multimem_hh_id_arr.size))
            multimem_hh_size_arr = multimem_hh_size_arr[multimem_hh_size_arr > 0]

        if not pd.isna(member_over_64y_p):
            """if pars.verbose > 0:
                print ("Assigning households to children > 64 years...")"""
            # choose household idx with at least one adult > 64y
            potential_people_mask = (self.people.age[people_id_in_district]>64)&(self.people.household[people_id_in_district]==0)
            potential_people = people_id_in_district[potential_people_mask]
            potential_people_n = potential_people.size

            selected_idx = idx_arr[np.random.random(multimem_hh_id_arr.size) < member_over_64y_p]
            # number of households should be more than potential_people_n
            if selected_idx.size > potential_people_n:
                selected_idx = np.random.choice(selected_idx, potential_people_n, replace=False)
            # number of people in multimember households should not be less than potential persons to fill those households
            if multimem_hh_size_arr[selected_idx].sum() < potential_people_n:
                raise Exception("total members in multi-member households less than potential persons that reside in these households")
            # randomly distribute potential_people_n to selected households
            n_in_selected_idx = utils.distribute_randint(N=potential_people_n, min_int=1, max_arr=multimem_hh_size_arr[selected_idx])
            # assign household id
            hh_id_for_potential_people = multimem_hh_id_arr[np.repeat(selected_idx, n_in_selected_idx)]
            np.random.shuffle(hh_id_for_potential_people)
            self.people.household[potential_people] = hh_id_for_potential_people
            # remove completed multimember households
            multimem_hh_size_arr[selected_idx] = multimem_hh_size_arr[selected_idx] - n_in_selected_idx
            multimem_hh_id_arr = multimem_hh_id_arr[multimem_hh_size_arr > 0]
            idx_arr = np.int32(np.arange(multimem_hh_id_arr.size))
            multimem_hh_size_arr = multimem_hh_size_arr[multimem_hh_size_arr > 0]

        # Assigning households for everyone else
        hh_id_for_potential_people = np.repeat(multimem_hh_id_arr, multimem_hh_size_arr)
        np.random.shuffle(hh_id_for_potential_people)
        potential_people_mask = self.people.household[people_id_in_district]==0
        potential_people = people_id_in_district[potential_people_mask]
        self.people.household[potential_people] = hh_id_for_potential_people

        return

    def setup_age(self, pars, popgrid, district_grid):

        age_headers = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29', '30-34', '35-39', '40-44', '45-49', '50-54', '55-59', '60-64', '65-69', '70-74', '75-79', '80-84', '85+']

        # get age demography
        wpp = pd.read_excel(pars.datadir + "/%s_census_2011_demography.xlsx"%(pars.place))
        wpp = wpp[wpp['district'].isin(list(self.district_mapping.keys()))]
        other_demography_info = wpp[list(set(wpp)-set(age_headers))].set_index("district")

        # initialize
        self.people.N = np.uint32(wpp[age_headers].to_numpy().sum())
        if pars.verbose > 0:
            print ("Population size = {:,}".format(self.people.N))
        self.people.id = np.arange(self.people.N).astype(np.uint32)
        self.people.age = np.zeros(self.people.N, dtype=np.uint16)
        self.people.district = np.zeros(self.people.N, dtype=np.uint16)

        prev_id = 0
        for district, district_idx in self.district_mapping.items():
            district_wpp = wpp[wpp['district'] == district]
            age_demography = district_wpp[age_headers].to_numpy()[0]
            #age_demography = np.int32(np.around(age_demography/age_demography.sum() * popgrid.sum()))

            for agebin, N in enumerate(age_demography):
                curr_id = prev_id + N
                # get age range of agebin
                min_age = agebin * 5
                if min_age < 85:
                    max_age = min_age + 5
                else:
                    max_age = 100
                age_range = np.arange(min_age, max_age).astype(np.int32)
                # sample age uniformly
                self.people.age[prev_id:curr_id] = np.random.choice(age_range, N, replace=True)
                # update district
                self.people.district[prev_id:curr_id] = district_idx
                # update prev_id
                prev_id += N

        return other_demography_info

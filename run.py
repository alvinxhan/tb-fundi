import os
import sciris as sc
import neopatat
import numpy as np

pop_pars = sc.objdict(
    datadir = "./data", # data directory
    year = 2005,
    place = "gauteng",
    # https://www.statssa.gov.za/publications/P0211/P02114thQuarter2011.pdf (ref A)
    # https://www.statssa.gov.za/publications/P0021/P00212011.pdf (ref B)
    wp_zipf_a = 0.983, # calculated based on 6,307,971 employed in 357,537 enterprises (ref B)
    verbose = 1,
)

if not os.path.isdir("./gauteng"):
    os.mkdir("./gauteng")

try:
    print ("Loading population object...")
    popobj = sc.load(filename = "./gauteng/pop.obj")
except:
    popobj = neopatat.Pop(pop_pars)
    sc.save(filename = "./gauteng/pop.obj", obj=popobj)

#raise Exception
sim_pars = sc.objdict(
    datadir = "./data", # data directory
    age_travel_fpath = "./data/gauteng_travel_data.xlsx",
    hiv_prev_pars = {"adult_prev":0.155, "child_adult_ratio":280000/4.2e6}, # prevalence in adults (15+), ratio of children/adults LHIV from UNAIDS (https://www.unaids.org/en/resources/documents/2023/HIV_estimates_with_uncertainty_bounds_1990-present)
    nweeks = np.uint32(3 * 52), #np.uint32(3 * 52), # period in weeks
    initial_state_dist = [0.2519, 0.1037, 0.0437, 0.0249, 0.378], # infection, minimal, subclinical, clinical, recovered
    initial_state_tau = [(3.61, 3.09), (8.39, 6.71), (3.31, 2.86), (7.13, 5.76), (5.68, 4.66)],
    district_mptb_prevalence = [149, 734, 522, 501, 603], # (per 100,000 people) tshwane, west rand, ekurhuleni, johannesburg, sedibeng
    beta = 1e-2, # per contact per week (to calibrate)
    year = 2005, # starting year
    ext_attack_rate = 0., # external attack rate per person per week
)

for sim_i in range(10):
    print ('Starting epidemic simulation %02d...'%(sim_i))
    simobj = neopatat.Sim(sim_pars, pop = popobj)
    simobj.simulate(death_bool=1)
    sc.save(filename = "./gauteng/sim_%02d.obj"%(sim_i), obj=simobj)
    break 

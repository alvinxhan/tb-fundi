# TB-FUNDI: an agent-based tuberculosis transmission model
Alvin X. Han and Colin A. Russell 

## Summary 
TB-Fundi is an agent-based model that simulates the spread of Mycobacterium tuberculosis (Mtb) through a spatially- and age-structured population. Fundi means an expert or knowledgeable person, from the isiZulu and isiXhosa word for teacher, umFundisi. TB-Fundi follows a programmatic flow:   
1. A population of simulated individuals is created based on demographic and location data. Contact networks between individuals are then set up based on their age and known population data (i.e. household composition data, school enrollment, employment rates, and employment firm size distribution) and human mobility properties.   
2. After setting up the population, the model iterates over a period of discrete time-steps (e.g. one-week), simulating the spread of the Mtb within the population created in step 1. During each time step, the simulation first updates the disease progression of each infected individual. Transmissions are then computed within households, schools, workplaces and through community contacts.  


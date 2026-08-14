"""
Section 1: E.coli Agent and Project Setup
TODO(owner: Dalila): status - in progress

Goal:
- Store default simulation parameters here (step length, dt, etc)
  so other sections import from a single source instead of HARDCODING!!!

"""

default_stepLength = 1.0 #run step, arbitrary value
default_dt = 0.01 #simulation timestep (s)
def_tumbles_perCycle = 4 #for section 3(Emma) gradient-check

#Numpy random seed generator
#can change seed for different results later on
default_seed = 42 #reproducable steps/exp's
rng = np.random.default_rng(default_seed) #local random generator instance

#E.coli movement
speed = 20 #(um/s) units , average speed of E.coli swimming

#Model constants
initial = [0,0] #cells start at the origin
ligand_center = [1000,1000] #position for highest concentration #can change later
center_exp, start_exp = 8,2 #exponent for concetration at [1000, 1000] and [0,0]

origin_to_center = 0 #distance from start to center initialized here
saturation_conc = 10**8 #change based on other sections?






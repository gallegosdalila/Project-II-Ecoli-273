"""
Section 1: E.coli Agent and Project Setup
TODO(owner: Dalila): status - in progress

Goal:
-Write tests that confirm :
    - direction vectors have length 1
    - step lengths are correct
    - fixed random seeds are repoducible
    - position history updates properly

"""

import numpy as np
#import our Bacterium class made in agents 
from ecoli_walking.agents import Bacterium 

def test_direction_l():
    rng = np.random.default_rng(0) #0 is the starting point for randomness
    #starts at origin [0,0] then pick the angle with rng
    #since no constructor is passed, the helper method created earlier is called
    b = Bacterium(position = [0,0], rng = rng)

    #check direction vector is a proper unit vector of length = 1
    #direction attribute on a Bacterium object (defined earlier)
    #np.linalg.norm will compute the magnitude(length) of the vector 
    #np.isclose will wheck if the normalized direction is close to 1.0 (estimation)
    #assert will give an error if the test/check failed to be close to 1.0
    #check line saying: 
    assert np.isclose(np.linalg.norm(b.direction), 1.0)







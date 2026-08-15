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

def test_direction_length():
    rng = np.random.default_rng(0) #0 is the starting point for randomness
    #starts at origin [0,0] then pick the angle with rng
    #since no constructor is passed, the helper method created earlier is called
    b = Bacterium(position = [0,0], rng = rng)

    #check direction vector is a proper unit vector of length = 1
    #direction attribute on a Bacterium object (defined earlier)
    #np.linalg.norm will compute the magnitude(length) of the vector 
    #np.isclose will wheck if the normalized direction is close to 1.0 (estimation)
    #assert will give an error if the test/check failed to be close to 1.0
    #what this checks: take the ecoli current direction vector, measure length & give error if not approx. 1 
    assert np.isclose(np.linalg.norm(b.direction), 1.0)

def test_tumble_direction():
    rng = np.random.default_rng(1)
    b = Bacterium(position = [0,0], rng = rng)
    #call the tumble method defined in the Bacterium class in agents.py onto b
    #wherever it says 'self' in the method, you apply b in its place
    b.tumble()
    assert np.isclose(np.linalg.norm(b.direction), 1.0)

def test_runStep_length():
    #create bacterium object by calling the class at position [0,0] & stored in variable b 
    b = Bacterium(position = [0,0])
    start = b = b.position.copy() 
    b.run(direction = [0,0], length = 2.5) #direction/length for the run
    assert np.isclose(np.linalg.norm(b.position - start), 2.5)

#check if using the same random seed will give the same result
def test_fixedSeed_reproducible():
    #use same seed for each rng
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    #b1 abd b1 creates a bacterium using the class 
    b1 = Bacterium(position = [0,0], rng = rng1)
    b2 = Bacterium(position = [0,0], rng = rng2)
    #compare two arrays x to x and y to y positions and confirms every pair is close enough 
    assert np.allclose(b1.direction, b2.direction)

def test_history_updates_run_tumble():
    b = Bacterium(position = [0,0])
    b.run(direction = [1,0], length = 1.0)
    b.tumble()
    assert len(b.position_history) == 3 #init + run + tumble









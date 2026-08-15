"""
Section 1: E.coli Agent and Project Setup
TODO(owner: Dalila): status - in progress

Represents an individual E.coli and controls its basic movement.
It does NOT calculate or interpret concentration! 
Only moves when told to by the simulaiton layer (Section 3 - Tracy)

Goal: 
- Create the Bacterium class
- Store position, direction, state and position history
- Implement tumble(rng) for one random direction step
- Implement run(direction, length) for one directed step
"""

import numpy as np

#E.coli is alternating between running in a stright line vs tumpling (stop & pick new direction)
class Bacterium:
    #position: where it starts, direction: which way it faces, state: run or tumble, rng: random num gen
    #by setting a max amount of steps, we can pre-allocate space in an array for running the simulation
    #this will be mor efficient since we know the amount of space we have instead of figuring that out as it runs
    def __init__(self, position, direction = None, state = "run", rng = None, max_steps = 1000):
        self.position = np.array(position, dtype = float) #convert [0,0] array into an array 
        self.state = state 
        #conditonal statement below: evaluates the condition once & assigns 1 of 2 possible values 
        #this is an if/else statement that produce a value
        #if statement assigns value WE passed in & else statement assigns a random value
        if rng is not None: #check if variable has a real value or if its empty 'None"
            self.rng = rng 
        else:
            self.rng = np.random.default_rng()

        #if no direction was given (on our end) will pick a random one by using the helper method 
        #'else' will converrt to whatever was passed into a float array
        if direction is None:
            self.direction = self._random_unit_vector()
        else:
            self.direction = np.array(direction, dtype = float)

        #create a list that tracks every position E.coli visits over time(for plotting trajectory)
        #the .copy() avoid reference to position and instead get a history of each point
        self.position_history = np.zeros((max_steps + 1, 2)) #preallocated space
        self.position_history[0] = self.position
        self.step_count = 0


    #internal private helper method needed for direction
    #pick a random angle between 0 and 2pi (unit circle) 
    #will give a random direction(angle) into a 2D vector normalized to 1
    def _random_unit_vector(self):
        angle = self.rng.uniform(0,2 * np.pi)
        return np.array([np.cos(angle), np.sin(angle)])

    #tumble method
    #here we are getting a tumble event that happened at a position (eg: X)
    def tumble(self, rng = None):
        #re-orient direction NOT position
        if rng is not None:
            r = rng
        else:
            r = self.rng

        angle = r.uniform(0, 2*np.pi)

        self.direction = np.array([np.cos(angle), np.sin(angle)])
        self.state = "tumble"

        self.step_count += 1 #adds 1 to each step count
        #get the current position into the next slot
        self.position_history[self.step_count] = self.position #indexed

        return self.direction
    #move 'length' along the 'direction
    #straight line movement based on biological movement
    def run(self, direction, length):
        direction = np.array(direction, dtype = float)
        norm = np.linalg.norm(direction) #compute vectors magnitude

        #check if directional vector is 0, thus you cant normalize!
        if norm == 0:
            raise ValueError("direction vector must be a nonzero")

        #normalize the vector to length 1 
        unit_direction = direction / norm

        self.position = self.position + (unit_direction * length)

        self.direction = unit_direction
        self.state = "run"
        #preallocate numpy array space since we set a # of steps ahead of time in constructor
        self.step_count += 1
        self.position_history[self.step_count] = self.position





        

        

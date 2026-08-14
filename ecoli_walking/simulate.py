"""
Section 3: Chemotaxis and Core Simulation
TODO(owner: Tracy) status - in progress

Implement the main biased random walk algorithm.
This section decides how the bacterium uses recent concentration info to choose a directed run.

It calls Section 1(Dalila) for movement and Section (Emma) 
for concentration, but does not own either.

Goal:
- Make each bacterium take four random tumble steps.
- Track concentration history over the sampling window so the value
    from four time steps earlier is available.
- Compare concentration at the current position with concentration
    four time steps earlier.
- Use the change in concentration and the change in position to
    estimate the directional gradient (estimate_gradient()) along the
    sampled displacement.
- Perform the directed run step.
- Repeat for the required number of iterations (chemotaxis_cycle(),
    run_simulation(), or a ChemotaxisSimulator class).
- Handle special cases such as very small displacement when the four
    tumbles nearly cancel.
- Fill in and return a SimulationResult, whose definition lives in
    contracts.py.


"""
import numpy as np


def estimate_gradient (old_position, new_position, old_concentration, new_concentration, tolerance= 1e-12,):
    """
    Goal: Estimate concentration gradient information using the displacement over  4 tumble steps. 

    Parameters: 

    old_position: array- like position at t - 4*dt
    new_position: array-like position at t
    old_concentration: float; concentration at t - 4*dt 
    new_concentration: float; concentration at t
    tolerance: float; to prevent division by a very small displacement 

    """
    old_position = np.array(old_position,dtype=float)
    new_position = np.array(new_position, dtype=float)

    displacement = new_position - old_position

    displacement_squared = (displacement[0]**2 + displacement[1]**2)

    if displacement_squared < tolerance:
        return None

    delta_concentration = (new_concentration - old_concentration)

    gradient = (delta_concentration/displacement_squared)*displacement

    return gradient

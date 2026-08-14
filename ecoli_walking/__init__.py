#This file turns the folder into a package

from .agents import Bacterium 
from .config import default_stepLength, default_dt, default_seed

__all__ = ["Bacterium, default_stepLength, default_dt, default_seed"]


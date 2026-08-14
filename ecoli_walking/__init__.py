#This file turns the folder into a package

from .agents import Bacterium 
from .config import def_stepLength, def_dt, def_seed

__all__ = ["Bacterium, def_stepLength, def_dt, deef_seed"]


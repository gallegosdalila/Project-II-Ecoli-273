#This file makes the our pckage installable.
#run pip install -e . in the terminal to install the package in editable mode.
#this will avoid path issues when importing

#pip install setuptools
from setuptools import setup, find_packages

setup( 
    name = "ecoli_sim",
    version = "0.1",
    description = "Biased Random Walk E.Coli as an Optimization",
    packages = find_packages(exclude = ["tests", "tests.*"]),
    install_requires = 
    [
        "numpy",
        "pandas",
        "matplotlib",
        "scipy"
    ],

    python_requires = ">=3.9", #Python 3.13.5 #check your version with "python --version" in your terminal to match!!



)
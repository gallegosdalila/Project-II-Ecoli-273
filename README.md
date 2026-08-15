#install steps, run instructions, package layout overview 

# ecoli_walk

Biased random walk simulation of E. coli chemotaxis.
CHEM 273 | Project 2
Aleyna Nur Celebi, Nisa Nur Celebi, Tracy Doumit, Dalila Zamantha Gallegos, Emma Patrichi 

## Status

Under active development. See project table for section breakdown and owners.

## Install

```bash
git clone <repo-url>
cd ecoli_walk
pip install -r requirements.txt
pip install -e .
```

## Run tests

```bash
pytest
```

## Usage

_TODO (Section 5): add quickstart example once main.py / experiments.py are complete._

## Package layout

- `agents.py` — Bacterium class (position, direction, state, history)
- `config.py` — default simulation parameters
- `fields.py` — concentration field profiles
- `contracts.py` — shared data structures (e.g. SimulationResult)
- `simulate.py` — core chemotaxis simulation loop
- `stats.py`, `experiments.py`, `main.py` — population experiments and CLI entry point
- `plotting.py`, `figures.py` — figure generation

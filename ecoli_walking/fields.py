"""
Section 2: Concentration Fields
TODO(owner: Emma) status - in progress

Creates the differential chemical concentration enviornments that the E.coli move through

Goal:
- Implement the 3 required concentration profiles:
    - shallow/linear gradient, single source, and competing sources
- Write concentration(x,y) so it can accept one position or an array of many positions
    and return the corresponding concentration values
- Implement the analytical gradient where possible so it can be used to validate Section 3(Tracy)
- Keep source locations and other field parameters configurable instead of HARDCODED!
- Add meshgrid helper that Section 5(Nisa) can use for contour plots

"""
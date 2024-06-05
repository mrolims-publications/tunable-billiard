"""
basin_entropy_from_data.py

This script calculates the basin entropy from the escape basin data.
It saves the Sb and Sbb in a new file as a function of h.

Usage: python basin_entropy_from_data.py

Author: Matheus Rolim Sales
Email: matheusrolim95@gmail.com
Last updated: 05/06/2024
"""

import numpy as np
import pandas as pd
from functions import *
import sys

# The path variable defines the path to where the data will be stored.
path = "Data"

gamma = 3
eps_ini = 0.2
eps_end = 0.90
deps = 0.01
epss = np.arange(eps_ini, eps_end + deps, deps)
h_ini = 0.01
h_end = 0.20
dh = 0.005
hs = np.arange(h_ini, h_end + dh, dh)
ns = [5]
N = int(1e6)
exponent = int(np.log10(N))
base = int(N/10**(exponent))
grid = 540
dalpha = 0.5
num_exits = int(sys.argv[1])
Sb = np.zeros((len(epss), len(hs), len(ns)))
Sbb = np.zeros_like(Sb)
for i, eps in enumerate(epss):
    print()
    for j, h in enumerate(hs):
        df = "%s/escape_basin_gamma=%i_eps=%.5f_h=%.3f_N=%ie%i_grid=%i_numexits=%i.dat" % (path, gamma, eps, h, base, exponent, grid, num_exits)
        print("Extracting data from %s..." % df)
        df = pd.read_csv(df, header=None, delim_whitespace=True)
        basin = np.array(df[2])
        basin = basin.reshape((grid, grid))
        print("Calculating entropy...")
        for k, n in enumerate(ns):
            Sb[i, j, k], Sbb[i, j, k], _, _ = boundary_entropy(basin, n)
    for k, n in enumerate(ns):
        df2 = "%s/basin_entropy_vs_h_n=%i_gamma=%i_eps=%.5f_N=%ie%i_grid=%i_numexits=%i.dat" % (path, n, gamma, eps, base, exponent, grid, num_exits)
        print("Saving data in %s..." % df2)
        with open(df2, "w") as df2:
            for j, h in enumerate(hs):
                df2.write("%.16f %.16f %.16f\n" % (h, Sb[i, j, k], Sbb[i, j, k]))
    
                
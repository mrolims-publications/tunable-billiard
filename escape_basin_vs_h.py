"""
escape_basin_vs_h_vs_xi.py

This script calculates the escape basins for changing h and xi.

Usage: python escape_basin_vs_h_vs_xi.py num_exits

where num_exits = 2 or 3.

Author: Matheus Rolim Sales
Email: matheusrolim95@gmail.com
Last updated: 05/05/2024
"""

import numpy as np
from functions import *
from datetime import datetime
import sys
import getpass


def escape_basin_vs_h(args):

    xi, num_exits = args
    # Define gamma
    gamma = 3
    # ---------------- #
    # --- Define h --- #
    # ---------------- #
    hs = np.array([0.01, 0.05, 0.08, 0.12, 0.15, 0.2])
    hs = hs[::-1]
    # -------------------------------- #
    # --- Define the escape region --- #
    # -------------------------------- #
    theta_range = np.array([0.0, np.pi/gamma])
    theta = np.linspace(theta_range[0], theta_range[1], grid, endpoint=True)
    dalpha = 0.5
    alpha_range = np.array([np.pi/2 - dalpha/2, np.pi/2 + dalpha/2])
    alpha = np.linspace(alpha_range[0], alpha_range[1], grid, endpoint=True)
    theta, alpha = np.meshgrid(theta, alpha)
    # ------------------------------- #
    # --- Define other parameters --- #
    # ------------------------------- #
    N = int(1e6) # Number of collisions
    exponent = int(np.log10(N))
    base = int(N/10**(exponent))
    grid = 1080 # Grid size
    # ----------------- #
    # --- ATTENTION --- #
    # ----------------- #
    # The path variable defines the path to where the data will be stored.
    # Please update the following lines accordingly
    user = getpass.getuser()
    if user == "matheus" or user == "rolim":
        path = "/home/%s/Pesquisa/TunableBilliard/Data" % user
    elif user == "jdanilo" or user == "jdsjunior":
        path = "/home/%s/Matheus/Pesquisa/TunableBilliard/Data" % user
    else:
        print("Cannot assign path!")
        sys.exit()

    # Calculates the escape_basin for different h
    for h in hs:

        eb = escape_basin(theta, alpha, gamma, xi, h, N, num_exits)

        df = "%s/escape_basin_gamma=%i_xi=%.5f_h=%.3f_N=%ie%i_grid=%i_numexits=%i.dat" % (path, gamma, xi, h, base, exponent, grid, num_exits)
        with open(df, "w") as df:
            for i in range(grid):
                for j in range(grid):
                    df.write("%.16f %.16f %i\n" % (theta[i, j], alpha[i, j], eb[i, j]))
                df.write("\n")

if __name__ == "__main__":

    # Define the number of exits
    num_exists = 2
    # ----------------- #
    # --- Define xi --- #
    # ----------------- #
    xi = 0.45
    args = [xi, num_exists]
    escape_basin_vs_h(args)
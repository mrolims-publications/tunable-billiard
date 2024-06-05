"""
survival_probability.py

This script calculates the survival probability for changing xi and h.

Usage: python survival_probability.py

Author: Matheus Rolim Sales
Email: matheusrolim95@gmail.com
Last updated: 24/04/2024
"""

import numpy as np
import time
from functions import *
from getpass import getuser

# --- ATTENTION ---#
# The path variable defines the path to where the data will be stored.
# Please update the following lines accordingly
user = getuser()
if user == "rolim" or user == "matheus":
    path = "/home/%s/Pesquisa/TunableBilliard/Data" % user
elif user == "jdsjunior" or user == "jdanilo":
    path = "/home/%s/Matheus/Pesquisa/TunableBilliard/Data" % user
else:
    print("Unable to assign path!")
    import sys
    sys.exit()

gamma = 3
# ----------------- #
# --- Define xi --- #
# ----------------- #
xis = [0.45]
# ---------------- #
# --- Define h --- #
# ---------------- #
h_ini = 0.01
h_end = 0.20
dh = 0.01
hs = np.arange(h_ini, h_end + dh, dh)
hs = hs[::-1]
#hs = np.array([0.01, 0.05, 0.08, 0.12, 0.15, 0.2])
#hs = [0.2]
# -------------------- #
# --- Define exits --- #
# -------------------- #
eh1 = 2*np.pi/3
eh2 = 5*np.pi/6
esc_exits = np.array([eh1, eh2])
# -------------------------------- #
# --- Define the escape region --- #
# -------------------------------- #
theta_range = np.array([0.0, np.pi/gamma])
dalpha = 0.5
alpha_range = np.array([np.pi/2 - dalpha/2, np.pi/2 + dalpha/2])

N = int(1e6) # Number of collisions
t = np.arange(1, N + 1) # Time array
eN = int(np.log10(N)) # exponent of N
bN = int(N/10**(eN)) # base of N
n_ic = int(1e6) # Number of initial conditions
enic = int(np.log10(n_ic)) # exponent of n_ic
bnic = int(n_ic/10**(enic)) # base of n_ic
# Array to store the survival probability of each exit
surv_prob = np.zeros((N, esc_exits.shape[0]))
for h in hs: # Change h
    for xi in xis: # Change xi
        print("Starting execution of h = %.3f and xi = %.3f" % (h, xi))
        # Define the random initial conditions
        theta = np.random.uniform(theta_range[0], theta_range[1], n_ic)
        alpha = np.random.uniform(alpha_range[0], alpha_range[1], n_ic)
        # Iterates through the exits
        for i in range(len(esc_exits)):
            print("Calculating survival probability for exit #%i..." % (i + 1))
            start_time = time.time()
            escape_times = escape_time(theta, alpha, gamma, xi, esc_exits[i], h, N)
            surv_prob[:, i] = survival_prob(escape_times, N)
            end_time = time.time()
            print("Finished exit #%i in %.2f min." % (i + 1, (end_time - start_time)/60))
        # File to store the data
        df = "%s/survival_prob_gamma=%i_xi=%.5f_h=%.3f_N=%ie%i_nic=%ie%i.dat" % (path, gamma, xi, h, bN, eN, bnic, enic)
        print("Writing data into file %s" % df)
        data = np.zeros((t.shape[0], esc_exits.shape[0] + 1))
        data[:, 0] = t
        for i in range(1, esc_exits.shape[0] + 1):
            data[:, i] = surv_prob[:, i - 1]
        np.savetxt(df, data, fmt="%.16f", delimiter=" ")
        print("Done with h = %.3f and xi = %.3f\n" % (h, xi))

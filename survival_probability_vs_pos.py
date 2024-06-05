"""
survival_probability_vs_pos.py

This script calculates the survival probability and the escaping histogram for 
changing xi and h for different exit positions.

Usage: python survival_probability_vs_pos.py

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
# ----------------------- #
# --- Parameter gamma --- #
# ----------------------- #
gamma = 3
# ----------------- #
# --- Define xi --- #
# ----------------- #
xi = 0.45
# ---------------- #
# --- Define h --- #
# ---------------- #
hs = [0.20]
# -------------------- #
# --- Define exits --- #
# -------------------- #
ehs = np.array([2*np.pi/3, 5*np.pi/7, 47*np.pi/63, 7*np.pi/9, 5*np.pi/6, 12*np.pi/13, 17*np.pi/18, 29*np.pi/30, 89*np.pi/90, np.pi]) #np.linspace(ehs_ini, ehs_end, 100)
# -------------------------------- #
# --- Define the escape region --- #
# -------------------------------- #
theta_range = np.array([0.0, np.pi/gamma])
dalpha = 0.5
alpha_range = np.array([np.pi/2 - dalpha/2, np.pi/2 + dalpha/2])
# ------------------------------- #
# --- Define other parameters --- #
# ------------------------------- #
N = int(1e6) # Number of collisions
eN = int(np.log10(N))
bN = int(N/10**(eN))
n_ic = int(5e6) # Number of initial_conditions
enic = int(np.log10(n_ic))
bnic = int(n_ic/10**(enic))
t_sp = np.arange(1, N + 1)
t_eh = np.arange(0, N + 1)

for h in hs: # Change h
    for eh in ehs: # Change exit position
        print("Starting execution of h = %.3f, eh = %.10f and xi = %.3f" % (h, eh, xi))
        # Define the random initial conditions
        theta = np.random.uniform(theta_range[0], theta_range[1], n_ic)
        alpha = np.random.uniform(alpha_range[0], alpha_range[1], n_ic)
        print("\tCalculating escape times...")
        start_time = time.time()
        escape_times = escape_time(theta, alpha, gamma, xi, eh, h, N)
        print("\tCalculating survival probability...")
        surv_prob = survival_prob(escape_times[np.where(escape_times != -1)], N)
        print("\tCalculating escape histogram...")
        has_not_escaped = np.where(escape_times == -1)
        escape_times[has_not_escaped] = N
        esc_hist = escape_hist(escape_times, N)
        end_time = time.time()
        print("Finished in %.2f min." % ((end_time - start_time)/60))

        df = "%s/survival_prob_vs_pos_gamma=%i_xi=%.5f_h=%.3f_N=%ie%i_nic=%ie%i_eh=%.10f.dat" % (path, gamma, xi, h, bN, eN, bnic, enic, eh)
        print("\tWriting survival probability data into file %s" % df)
        data = np.zeros((surv_prob.shape[0], 2))
        data[:, 0] = t_sp
        data[:, 1] = surv_prob
        np.savetxt(df, data, fmt="%.16f", delimiter=" ")

        df = "%s/escape_histogram_vs_pos_gamma=%i_xi=%.5f_h=%.3f_N=%ie%i_nic=%ie%i_eh=%.10f.dat" % (path, gamma, xi, h, bN, eN, bnic, enic, eh)
        print("\tWriting escape histogram data into file %s" % df)
        data = np.zeros((esc_hist.shape[0], 2))
        data[:, 0] = t_eh
        data[:, 1] = esc_hist
        np.savetxt(df, data, fmt="%.16f", delimiter=" ")
        print("Done with h = %.3f, eh = %.10f and xi = %.3f\n" % (h, eh, xi))

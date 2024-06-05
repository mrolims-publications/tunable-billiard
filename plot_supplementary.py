import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from string import ascii_lowercase
import pandas as pd
from joblib import Parallel, delayed
from functions import *

def main(args):

    iii, eps, num_exits = args
    if iii < 10:
        iii = "00%i" % iii
    elif iii >= 10:
        iii = "0%i" % iii
    elif iii >= 100:
        iii = "%i" % iii
    gamma = 3
    N = int(1e6)
    eN = int(np.log10(N))
    bN = int(N/10**eN)
    grid = 1080
    # The path_data variable defines the path to where the data will be stored.
    path_data = "Data"
    # The path_figures variable defines the path to where the figures will be stored.
    path_figures = "Figures"

    plot_params(fontsize=18)
    fig, ax = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(11, 5.8))
    if num_exits == 2:
        colors = ["white", "black", "red"]
        values = [-1.5, -0.5, 0.5, 1.5]
        ticks = [-1, 0, 1]
        tick_labels = ["$e_\\infty$", "$e_1$", "$e_2$"]
    elif num_exits == 3:
        colors = ["white", "black", "red", "gold"]
        values = [-1.5, -0.5, 0.5, 1.5, 2.5]
        ticks = [-1, 0, 1, 2]
        tick_labels = ["$e_\\infty$", "$e_1$", "$e_2$", "$e_3$"]
    cmap = mpl.colors.ListedColormap(colors)
    norm = mpl.colors.BoundaryNorm(values, cmap.N)
    xbox = 0.009
    ybox = 0.9162
    bbox = {'linewidth': 0.0, 'facecolor': 'white', 'alpha': 0.75, 'pad': 1}
    dalpha = 0.5
    hs = [0.01, 0.05, 0.08, 0.12, 0.15, 0.20]
    for i, h in enumerate(hs):
        df = "%s/escape_basin_gamma=%i_eps=%.5f_h=%.3f_N=%ie%i_grid=%i_numexits=%i.dat" % (path_data, gamma, eps, h, bN, eN, grid, num_exits)
        print("Extracting data from %s..." % df)
        df = pd.read_csv(df, header=None, delim_whitespace=True)
        x = np.array(df[0])
        y = np.array(df[1])
        z = np.array(df[2])
        x = x.reshape((grid, grid))
        y = y.reshape((grid, grid))
        z = z.reshape((grid, grid))
        print("Plotting the data...")
        hm = ax[int(i / 3), i % 3].pcolor(x, y, z, cmap=cmap, norm=norm)
        ax[int(i / 3), i % 3].text(xbox, ybox, "(%s)" % (ascii_lowercase[i]), transform=ax[int(i / 3), i % 3].transAxes, bbox=bbox)
    ax[0, 0].set_xticks([0, np.pi/6, np.pi/3])
    ax[0, 0].set_xticklabels(["$0$", "$\\pi/6$", "$\\pi/3$"])
    ax[0, 0].set_yticks([np.pi/2 - dalpha/2, np.pi/2, np.pi/2 + dalpha/2])
    ax[0, 0].set_yticklabels(["$\\pi/2 - %g$" % (dalpha/2), "$\\pi/2$", "$\\pi/2 + %g$" % (dalpha/2)])
    [ax[-1, i].set_xlabel("$\\theta$") for i in range(ax.shape[1])]
    [ax[i, 0].set_ylabel("$\\alpha$") for i in range(ax.shape[0])]

    ax[0, 1].set_title("$\\xi = %.2f$" % eps, fontsize=18)

    cbar_ax = fig.add_axes([0.961, 0.0925, 0.01, 0.95-0.0925])
    cbar = fig.colorbar(hm, cax=cbar_ax, ticks=ticks)
    cbar.ax.set_yticklabels(tick_labels)

    plt.subplots_adjust(left=0.107, bottom=0.0925, right=0.955, top=0.95, hspace=0.1, wspace=0.08)
    figname = "%s/%iexits/%s.png" % (path_figures, num_exits, iii)
    print("Saving in %s..." % figname)
    plt.savefig(figname, dpi=300)
    plt.close()

if __name__ == "__main__":

    eps_ini = 0.2
    eps_end = 0.9
    deps = 0.01
    epss = np.arange(eps_ini, eps_end + deps, deps)
    if abs(epss[-1] - eps_end) > 1e-10:
        epss = epss[:-1]

    num_exits = [2]

    for num_exit in num_exits:
        Parallel(n_jobs=2)(delayed(main)([i, epss[i], num_exit]) for i in range(len(epss)))
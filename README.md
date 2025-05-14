# tunable-billiard-escape

Code repository accompanying the publication entitled "[An investigation of escape and scaling properties of a billiard system](https://doi.org/10.1063/5.0222215)".

This project contains the code to generate and plot the data from all figures.

## Requirements

The required packages are listed in ``` requirements.txt ```. To install them please execute ``` pip install -r requirements.txt ```.

## Figure 1

To generate Figure 1, run all cells within the heading named ``` Fig. 1 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

## Figure 2

To generate Figure 2, run all cells within the heading named ``` Fig. 2 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

## Figure 3

To generate Figure 3, run all cells within the heading named ``` Fig. 3 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

## Figure 4

To generate Figure 4, run all cells within the heading named ``` Fig. 4 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

## Figures 5, 6, and 7

To generate the data used in these Figures, run ``` python survival_probability.py ```. It will generate the survival probability for ``` xi = 0.45 ``` with different hole sizes.

### Figure 5

To generate Figure 5, run all cells within the heading named ``` Fig. 5 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

### Figure 6

To generate Figure 6, run all cells within the heading named ``` Fig. 6 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

### Figure 7

To generate Figure 7, run all cells within the heading named ``` Fig. 7 ``` in the ``` Plots.ipynb ``` Jupyter notebook. Make sure to run the cells within the heading named ``` Fig. 6 ``` before running these cells.

## Figure 8

To generate the data in Figure 8(a), run ``` python survival_probability_vs_pos.py ```. It generates the survival probability for a fixed hole size with different hole positions. To generate Figure 8, run all cells within the heading named ``` Fig. 8 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

## Figure 9

To generate the data of Figure 9, run ``` python escape_basin_vs_h.py ```. It generates the escape basin for a fixed value of the control parameter with different hole sizes.  To generate Figure 9, run all cells within the heading named ``` Fig. 9 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

## Figure 10 and 11

To generate the data of Figure 10 and 11, run ``` python escape_basin_vs_h_vs_xi.py ```. It generates the escape basin for different values of the control parameter and different hole sizes. Then, run ``` python basin_entropy_from_data.py ```. It calculates the basin entropy and the basin boundary entropy given the escape basin data.

### Figure 10

To generate Figure 10, run all cells within the heading named ``` Fig. 10 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

### Figure 11

To generate Figure 10, run all cells within the heading named ``` Fig. 11 ``` in the ``` Plots.ipynb ``` Jupyter notebook.

## Supplementary video

To generate the data used in the Supplementary Video, run ``` python escape_basin_vs_h_vs_xi.py ```. It generates the escape basin for different values of the control parameter and different hole sizes. Then, run ``` python basin_entropy_from_data.py ```. It calculates the basin entropy and the basin boundary entropy given the escape basin data. To generate the Figures used in the Video, run  ``` python plot_supplementary.py ```. It plots the escape basins for changing parameter control with 6 different hole sizes.

## Citation

If you use this repository or parts of it in your work, please cite:

M. Rolim Sales, D. Borin, D. R. da Costa, J. D. Szezech Jr., and E. D. Leonel, **An investigation of escape and scaling properties of a billiard system**, [*Chaos: An Interdisciplinary Journal of Nonlinear Science 34, 113122 (2024)*](https://doi.org/10.1063/5.0222215).

## Contact

[matheusrolim95@gmail.com](mailto:matheusrolim95@gmail.com)

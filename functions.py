"""
functions.py

This module contains the main functions for the analysis reported on the paper "An investigation of the escaping properties of a billiard system".

List of Functions:
-------------------
- Geometry:
    1. R: Calculate the radius of the billiard.
    2. dRdtheta: Calculate the first derivative of R(theta) with respect to theta.
    3. d2Rdtheta2: Calculate the second derivative of R(theta) with respect to theta.
    4. F_R: Calculate the value of F(R).
    5. Fl_R: Calculate the first derivative of F(R) with respect to R.
    6. Fll_R: Calculate the second derivative of F(R) with respect to R.

- Collision and trajectories:
    7. collision_point: Calculate the next collision point given the previous collision.
    8. time_series: Calculate the time series of variables x and y.
    9. phase_space: Calculate the time series of variables theta and alpha.

- Escape properties:
    10. escape_time: Calculate the escape time of a particle from a given exit position.
    11. survival_prob: Calculate the survival probability given the escape times.
    12. escape_hist: Calculate the escape histogram given the escape times.
    13. get_exits: Define the positions of the exits.
    14. escape_basin: Calculate the escape basin of a given initial condition.
    15. escape_basin_and_time: Calculate the escape basin and escape time for a given initial condition.
    
- Entropy and information:
    16. boundary_entropy: Calculate the basin entropy and basin boundary entropy of a given escape basin.

- Plotting:
    17. plot_params: Define the parameters for making plots.

Author: Matheus Rolim Sales
Email: matheusrolim95@gmail.com
Last updated: 13/04/2024
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys
import collections
from numba import vectorize, njit

@vectorize(['f8(i8, f8, f8)'],
            nopython=True)
def R(gamma: np.int32, xi: np.float64, theta: np.float64) -> float:
    """
    Calculate the radius of the billiard given theta element-wise.
    
    Parameters
    ------------
    gamma : np.int32
        Defines the shape of the billiard.
    xi   : np.float64
        Modifies the shape of the billiard.
    theta : np.float64 or np.ndarray
        Polar angle measured counter-clockwise from the x-axis.
                  
    Returns
    ------------
    R : np.float64 or np.ndarray
        The radius of the billiard for a given angle theta. This is a scalar if theta is a scalar.
    """
    limit = 1e9
    b = 2*np.sqrt(3*xi)*np.cos(gamma*theta)/9

    if abs(b) < 1e-11:
        xn = 3.0
    else:
        xs = -2/(3*b)
        xi = xs/2.0
        if xs <= 0:
            xn = 3.0
        else:
            if xi > 3.0:
                xn = 3.0
            else:
                xn = xi
    
    e = 1/limit
    i = 1
    while True:
        if i > limit:
            print('No convergence for theta!!!')
        FR = xn**2 + b*xn**3 - 1
        dFdR = xn*(3*b*xn + 2)
        xn_new = xn - FR/dFdR
        d = xn_new - xn
        if abs(d) < e:
            return xn_new
        xn = xn_new
        i = i + 1
    
    return -1

def F_R(R: np.float64, b: np.float64) -> float:
    """
    Calculate the value of F(R) = R² + bR³ - 1.

    Parameters
    ----------
    R : np.float64
        The value of R.
    b : np.float64
        b = 2*np.sqrt(3*xi)/9 * cos(theta)
    
    Returns
    -------
    float : the derivative of R.
    """

    return R**2 + b*R**3 - 1

def Fl_R(R : np.float64, b : np.float64) -> float:
    """
    Calculate the value of F'(R) = R(3bR + 2).

    Parameters
    ----------
    R : np.float64
        The value of R.
    b : np.float64
        b = 2*np.sqrt(3*xi)/9 * cos(theta)
    
    Returns
    -------
    float : the derivative of R.
    """
    return R*(3*b*R + 2)

@vectorize(['f8(f8, i8, f8, f8)'],
          nopython=True)
def dRdtheta(R: np.float64, gamma: np.int32, xi: np.float64, theta: np.float64) -> float:
    """
    Calculate the first derivative of R(theta), dR / d theta element-wise.

    Parameters
    ------------
    R : np.float64 or np.ndarray
        The radius of the billiard for a given angle theta. Must be of same size as theta.
    gamma : np.int32
        Defines the shape of the billiard.
    xi : np.float64
        Modifies the shape of the billiard.
    theta : np.float64 or np.ndarray
        Polar angle measured counter-clockwise from the x-axis. Must be of same size as R.
    
    Returns
    ------------
    dRdtheta : np.float64 or np.ndarray
        The first derivative of R(theta). This is a scalar if R and theta are scalars.
    """
    a = 2*np.sqrt(3*xi)/9

    cima = a*gamma*np.sin(gamma*theta)*R**2
    baixo = 2 + 3*a*np.cos(gamma*theta)*R

    return cima/baixo

@vectorize(['f8(f8, f8, i8, f8, f8)'],
          nopython=True)
def d2Rdtheta2(dRdtheta: np.float64, R: np.float64, gamma: np.int32, xi: np.float64, theta: np.float64) -> float:
    """
    Calculate the second derivative of R(theta), d2(R) / d theta2 element-wise.

    Parameters
    ------------
    dR/dtheta : np.float64 or np.ndarray
        The first derivative of R for a given angle theta. Must be of same size as theta and R.
    R : np.float64 or np.ndarray
        The radius of the billiard for a given angle theta. Must be of same size as theta and dRdtheta.
    gamma : np.int32
        Defines the shape of the billiard.
    xi : np.float64
        Modifies the shape of the billiard.
    theta : np.float64 or np.ndarray
        Polar angle measured counter-clockwise from the x-axis. Must be of same size as R and dR/dtheta.
    
    Returns
    ------------
    dR2dtheta2 : np.float64 or np.ndarray
        The second derivative of R(theta). This is a scalar if R, dRdtheta and theta are scalars.
    """

    a = 2*np.sqrt(3*xi)/9

    AA = (2*R*dRdtheta*a*gamma*np.sin(gamma*theta) + a*gamma*gamma*R*R*np.cos(gamma*theta))*(2 + 3*a*R*np.cos(gamma*theta))
    BB = (a*gamma*R*R*np.sin(gamma*theta))*(3*a*dRdtheta*np.cos(gamma*theta) - 3*a*gamma*R*np.sin(gamma*theta))
    CC = (2 + 3*a*R*np.cos(gamma*theta))**2

    return (AA - BB)/CC

@njit
def collision_point(x0: np.float64, y0: np.float64, Rmax: np.float64, gamma: np.int32, xi: np.float64, mu: np.float64) -> float:
    """
    Calculate the next coliision point given the initial point (x0, y0) and the angle mu.

    This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ------------
    x0 : np.float64
        Initial x-position on the boundary.
    y0 : np.float64
        Initial y-position on the boundary.
    Rmax : np.float64
        Maximum radius of the billiard. It is the value of R for theta = pi/gamma.
    gamma : np.int32
        Defines the shape of the billiard.
    xi : np.float64
        Modifies the shape of the billiard.
    mu : np.float64
        Direction of the particle's velocity measured counter-clockwise from the x-axis.

    Returns
    -----------
    theta : np.float64
        Polar angle measured counter-clockwise from the x-axis.
           
    """
    tol = 1e-11
    b = 2*(x0*np.cos(mu) + y0*np.sin(mu))
    c = x0**2 + y0**2 - Rmax**2
    dte = (-b + np.sqrt(b**2 - 4*c))/2
    j = 1
    while True:
        if(j > 200):
            print('No solution for theta!!')
            break
        xe = x0 + np.cos(mu)*dte
        ye = y0 + np.sin(mu)*dte
        re = np.sqrt(xe**2 + ye**2)
        thetaa = np.arctan2(ye, xe) % (2*np.pi)
        Ra = R(gamma, xi, thetaa)
        xa = Ra*np.cos(thetaa)
        ya = Ra*np.sin(thetaa)
        if abs(re - Ra) < tol and abs(xe - xa) < tol and abs(ye - ya) < tol:
            return thetaa
        # Update the positions for a new test
        Rla = dRdtheta(Ra, gamma, xi, thetaa)
        xla = Rla*np.cos(thetaa) - ya
        yla = Rla*np.sin(thetaa) + xa
        dte = (ya - y0 + (yla/xla)*(x0 - xa))/(np.sin(mu) - (yla/xla)*np.cos(mu))
        j += 1
    
    return -1

@njit
def time_series(theta0: np.float64, alpha0: np.float64, gamma: np.int32, xi: np.float64, num_coll: np.int32) -> np.ndarray:
    """
    Calculate the (x, y) time series given an initial condition.

    This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ------------
    theta0   : np.float64
        Initial polar angle on the boundary measured counter-clockwise from the x-axis.
    alpha0   : np.float64
        Initial velocity's direction measured from the tangent line on theta0.
    gamma    : np.int32
        Defines the shape of the billiard.
    xi      : np.float64
        Modifies the shape of the billiard.
    num_coll : np.int32
        Number of collisions.

    Returns
    ------------
    ts : np.ndarray
        The x and y time series. t[:, 0] = x and t[:, 1] = y.
    """
    # Return array
    ts = np.zeros((num_coll + 1, 2))
    # Initial quantities
    Rmax = R(gamma, xi, (np.pi/gamma))
    R0 = R(gamma, xi, theta0)
    Rl = dRdtheta(R0, gamma, xi, theta0)
    x0 = R0*np.cos(theta0)
    y0 = R0*np.sin(theta0)
    xl = Rl*np.cos(theta0) - y0
    yl = Rl*np.sin(theta0) + x0
    phi = np.arctan2(yl, xl) % (2*np.pi)
    mu = (alpha0 + phi) % (2*np.pi)
    # Initial condition
    ts[0, 0] = x0
    ts[0, 1] = y0
    
    for i in range(num_coll):
        theta = collision_point(ts[i, 0], ts[i, 1], Rmax, gamma, xi, mu)
        R0 = R(gamma, xi, theta)
        Rl = dRdtheta(R0, gamma, xi, theta)
        ts[i + 1, 0] = R0*np.cos(theta)
        ts[i + 1, 1] = R0*np.sin(theta)
        xl = Rl*np.cos(theta) - ts[i + 1, 1]
        yl = Rl*np.sin(theta) + ts[i + 1, 0]
        phi = np.arctan2(yl, xl) % (2*np.pi)
        alpha = (phi - mu) % (np.pi)
        mu = (alpha + phi) % (2*np.pi)
        
    return ts

@njit
def phase_space(theta: np.float64, alpha: np.float64, gamma: np.int32, xi: np.float64, num_coll: np.int32) -> np.ndarray:
    """
    Calculate the (theta, alpha) time series given an initial condition.

    This function uses Numba's `njit` decorator for performance optimization.

    Parameters
    ------------
    theta0 : np.float64
        Initial polar angle on the boundary measured counter-clockwise from the x-axis.
    alpha0 : np.float64
        Initial velocity's direction measured from the tangent line on theta0.
    gamma : np.int32
        Defines the shape of the billiard.
    xi : np.float64
        Modifies the shape of the billiard.
    num_coll : np.int32
        Number of collisions.

    Returns
    ------------
    ts : np.ndarray
        The theta and alpha time series. t[:, 0] = theta and t[:, 1] = alpha.
    """
    ps = np.zeros((num_coll + 1, 2))
    ps[0, 0] = theta
    ps[0, 1] = alpha
    # Initial quantities
    Rmax = R(gamma, xi, (np.pi/gamma))
    Ra = R(gamma, xi, theta)
    Rl = dRdtheta(Ra, gamma, xi, theta)
    x = Ra*np.cos(theta)
    y = Ra*np.sin(theta)
    xl = Rl*np.cos(theta) - Ra*np.sin(theta)
    yl = Rl*np.sin(theta) + Ra*np.cos(theta)
    phi = np.arctan2(yl, xl) % (2*np.pi)
    mu = (alpha + phi) % (2*np.pi)
    for i in range(1, num_coll + 1):
        theta = collision_point(x, y, Rmax, gamma, xi, mu)
        Ra = R(gamma, xi, theta)
        Rl = dRdtheta(Ra, gamma, xi, theta)
        x = Ra*np.cos(theta)
        y = Ra*np.sin(theta)
        xl = Rl*np.cos(theta) - Ra*np.sin(theta)
        yl = Rl*np.sin(theta) + Ra*np.cos(theta)
        phi = np.arctan2(yl, xl) % (2*np.pi)
        alpha = (phi - mu) % np.pi
        mu = (alpha + phi) % (2*np.pi)
        #
        ps[i, 0] = theta
        ps[i, 1] = alpha

    return ps

@vectorize(["i8(f8, f8, i8, f8, f8, f8, i8)"],
            nopython=True,
            target="parallel")
def escape_time(theta0: np.float64, alpha0: np.float64, gamma: np.int32, xi: np.float64, theta_c: np.float64, h: np.float64, N: np.int32) -> int:
    """

    Calculates the escape time given the initial condition, element-wise, for the exit centered at `theta_c`.

    Parameters
    ----------
    theta0 : np.float64
        Initial angle of the particle.
    alpha0 : np.float64
        Initial angle parameter.
    gamma : np.int32
        Integer parameter for the basin.
    xi : np.float64
        Energy parameter.
    theta_c : np.float64
        The angular position of the center of the exit
    h : np.float64
        Size of the exit.
    N : np.int32
        Maximum number of collisions.
    """
    esctime = -1
    theta = theta0
    alpha = alpha0
    Rmax = R(gamma, xi, (np.pi/gamma))
    Ra = R(gamma, xi, theta)
    Rl = dRdtheta(Ra, gamma, xi, theta)
    x = Ra*np.cos(theta)
    y = Ra*np.sin(theta)
    xl = Rl*np.cos(theta) - Ra*np.sin(theta)
    yl = Rl*np.sin(theta) + Ra*np.cos(theta)
    phi = np.arctan2(yl, xl) % (2*np.pi)
    mu = (alpha + phi) % (2*np.pi)
    for i in range(1, N + 1):
        theta = collision_point(x, y, Rmax, gamma, xi, mu)
        if theta > theta_c - h/2 and theta < theta_c + h/2 and i > 1:
            esctime = i
            break
        Ra = R(gamma, xi, theta)
        Rl = dRdtheta(Ra, gamma, xi, theta)
        x = Ra*np.cos(theta)
        y = Ra*np.sin(theta)
        xl = Rl*np.cos(theta) - Ra*np.sin(theta)
        yl = Rl*np.sin(theta) + Ra*np.cos(theta)
        phi = np.arctan2(yl, xl) % (2*np.pi)
        alpha = (phi - mu) % np.pi
        mu = (alpha + phi) % (2*np.pi)

    return esctime

@njit
def survival_prob(escape_times: np.ndarray, N: np.int32):
    """
    Calculate the survival probability for a given set of escape times.

    Parameters
    ----------
    escape_times : numpy.ndarray
        An array containing the escape times.
    N : numpy.int32
        The maximum time or observation point for which to calculate the survival probability.

    Returns
    -------
    numpy.ndarray
        An array of survival probabilities at each time point up to N.

    Notes
    -----
    The survival probability at a given time point i is defined as the number of observations 
    that escape beyond time i divided by the total number of observations.

    Examples
    --------
    >>> escape_times = np.array([1, 2, 3, 4, 5])
    >>> N = 5
    >>> survival_prob(escape_times, N)
    array([5/5, 4/5, 3/5, 2/5, 1/5])
    """
    sp = np.zeros(N)
    n_ic = len(escape_times)

    escape_times.sort()

    for i in range(N):
        idx = np.searchsorted(escape_times, i + 1, side='right')
        sp[i] = n_ic - idx

    return sp / n_ic

def escape_hist(escape_times, N):
    """
    Calculate the normalized histogram of escape times up to N.

    Parameters:
        escape_times (list or array): List of escape times.
        N (int): Maximum escape time.

    Returns:
        ndarray: Normalized histogram of escape times.
    """
    # Convert escape_times to a numpy array
    escape_times = np.array(escape_times)
    
    # Initialize an array to hold the histogram values
    histogram = np.bincount(escape_times, minlength=N + 1)
    
    # Normalize the histogram by the total number of initial conditions
    n_ic = len(escape_times)
    normalized_histogram = histogram / n_ic
    
    return normalized_histogram

@njit
def get_exits(num_exits):
    """
    Generate escape exits angles based on the number of exits requested.

    Parameters
    ----------
    num_exits : int
        The number of escape holes to generate.

    Returns
    -------
    numpy.ndarray
        An array containing the escape hole angles in radians.

    Notes
    -----
    The function generates escape exti angles based on pre-defined values. 
    The escape exits are selected based on the number of exits requested.

    Examples
    --------
    >>> get_exits(2)
    array([2*np.pi/3, 5*np.pi/6])
    >>> get_exits(3)
    array([2*np.pi/3, 5*np.pi/6, np.pi])
    """

    ee1 = 2*np.pi/3
    ee2 = 5*np.pi/6
    ee3 = np.pi
    esc_exits = np.array([ee1, ee2, ee3])
    esc_exits = esc_exits[:num_exits]

    return esc_exits

@vectorize(["i8(f8, f8, i8, f8, f8, i8, i8)"],
            nopython=True,
            target="parallel")
def escape_basin(theta0: np.float64, alpha0: np.float64, gamma: np.int32, xi: np.float64, h: np.float64, N: np.int32, num_exits: np.int32):
    """
    Calculates the escape basin of a given initial condition given the set of exits.

    Parameters
    ----------
    theta0 : np.float64
        Initial angle of the particle.
    alpha0 : np.float64
        Initial angle parameter.
    gamma : np.int32
        Integer parameter for the basin.
    xi : np.float64
        Energy parameter.
    h : np.float64
        Size of the exit holes.
    N : np.int32
        Maximum number of collisions.
    num_exits : np.int32
        Number of exits in the basin.

    Returns
    -------
    np.int32
        Index of the exit hole through which the particle escaped.
        Returns -1 if the particle did not escape within N collisions.

    Notes
    -----
    The function simulates the escape of a particle from the billiard
    using a series of collision events. The escape is determined by checking if
    the particle's angle of escape matches the position of an exit hole within
    a certain tolerance (h/2). The function returns the index of the exit hole
    through which the particle escaped or -1 if the particle did not escape
    within N collisions.
    """
    esc_exits = get_exits(num_exits)
    theta = theta0
    alpha = alpha0
    Rmax = R(gamma, xi, (np.pi/gamma))
    Ra = R(gamma, xi, theta)
    Rl = dRdtheta(Ra, gamma, xi, theta)
    x = Ra*np.cos(theta)
    y = Ra*np.sin(theta)
    xl = Rl*np.cos(theta) - Ra*np.sin(theta)
    yl = Rl*np.sin(theta) + Ra*np.cos(theta)
    phi = np.arctan2(yl, xl) % (2*np.pi)
    mu = (alpha + phi) % (2*np.pi)
    for i in range(1, N + 1):
        theta = collision_point(x, y, Rmax, gamma, xi, mu)
        for j in range(len(esc_exits)):
            if theta > esc_exits[j] - h/2 and theta < esc_exits[j] + h/2 and i > 2:
                return j
        Ra = R(gamma, xi, theta)
        Rl = dRdtheta(Ra, gamma, xi, theta)
        x = Ra*np.cos(theta)
        y = Ra*np.sin(theta)
        xl = Rl*np.cos(theta) - Ra*np.sin(theta)
        yl = Rl*np.sin(theta) + Ra*np.cos(theta)
        phi = np.arctan2(yl, xl) % (2*np.pi)
        alpha = (phi - mu) % np.pi
        mu = (alpha + phi) % (2*np.pi)

    return -1

@njit
def escape_basin_and_time(theta0: np.float64, alpha0: np.float64, gamma: np.int32, xi: np.float64, h: np.float64, N: np.int32, esc_exits: np.ndarray):
    """
    Calculates the escape basin and escape time of a given initial condition given the set of exits.

    Parameters
    ----------
    theta0 : np.float64
        Initial angle of the particle.
    alpha0 : np.float64
        Initial angle of deviation.
    gamma : np.int32
        Deformation parameter for the system.
    xi : np.float64
        Deformation parameter for the system.
    h : np.float64
        Width of the escape exits.
    N : np.int32
        Maximum number of collisions for the particle.
    esc_exits : np.ndarray
        Array containing the escape exit positions.
    
    Returns
    -------
    np.ndarray
        Array containing two values:
        1. Index of the escape hole.
        2. Time step at which the particle escaped.
    
    Notes
    -----
    The function simulates the motion of a particle in the billiard and checks for escape 
    through one of the specified escape exits. It returns the index of the escape exit 
    and the time step at which the particle escaped.
    """
    theta = theta0
    alpha = alpha0
    Rmax = R(gamma, xi, (np.pi/gamma))
    Ra = R(gamma, xi, theta)
    Rl = dRdtheta(Ra, gamma, xi, theta)
    x = Ra*np.cos(theta)
    y = Ra*np.sin(theta)
    xl = Rl*np.cos(theta) - Ra*np.sin(theta)
    yl = Rl*np.sin(theta) + Ra*np.cos(theta)
    phi = np.arctan2(yl, xl) % (2*np.pi)
    mu = (alpha + phi) % (2*np.pi)
    for i in range(1, N + 1):
        theta = collision_point(x, y, Rmax, gamma, xi, mu)
        for j in range(len(esc_exits)):
            if theta > esc_exits[j] - h/2 and theta < esc_exits[j] + h/2 and i > 2:
                return np.array([j, i])
        Ra = R(gamma, xi, theta)
        Rl = dRdtheta(Ra, gamma, xi, theta)
        x = Ra*np.cos(theta)
        y = Ra*np.sin(theta)
        xl = Rl*np.cos(theta) - Ra*np.sin(theta)
        yl = Rl*np.sin(theta) + Ra*np.cos(theta)
        phi = np.arctan2(yl, xl) % (2*np.pi)
        alpha = (phi - mu) % np.pi
        mu = (alpha + phi) % (2*np.pi)

    return np.array([-1, i])

def boundary_entropy(basin: np.ndarray, n: np.int32) -> np.ndarray:
    """
    Calculate boundary entropy for a given 2-dimensional basin.

    Parameters
    ----------
    basin : np.ndarray
        2-dimensional numpy array representing the basin of attraction.
    
    n : np.int32
        Integer representing the size of each sub-box for entropy calculation.
    
    Returns
    -------
    tuple of np.ndarray
        Tuple containing four entropy values:
        Sb: Average entropy per box.
        Sbb: Average entropy per boundary.
        Sb_max: Maximum entropy per box (equiprobable states).
        Sbb_max: Maximum entropy per boundary (equiprobable states).
    
    Notes
    -----
    The function calculates the entropy of a given 2-dimensional basin by dividing
    it into n x n sub-boxes. It calculates the entropy for each sub-box and boundary,
    and returns the average entropy values as well as the maximum entropy values
    assuming equiprobable states.

    Raises
    ------
    ValueError
        If basin is not 2-dimensional.
        If grid size is invalid (not divisible by n).
    
    Examples
    --------
    >>> basin = np.array([[0, 0, 1, 1],
    ...                   [0, 0, 1, 1],
    ...                   [1, 1, 0, 0],
    ...                   [1, 1, 0, 0]])
    >>> n = 2
    >>> boundary_entropy(basin, n)
    (0.6931471805599453, 1.3862943611198906, 0.6931471805599453, 0.6931471805599453)
    """
    if len(basin.shape) == 2:
        N = basin.shape[0]
    else:
        print("basin must be 2-dimensional!")
        sys.exit()
    if N % n == 0:
        log = np.log2
        M = int(N/n)
        S = np.zeros((M, M))
        Smax = 0
        Nb = 0
        for i in range(M): # x loop
            x_ini = int(i*n)
            x_end = int((i + 1)*n)
            for j in range(M): # y loop
                y_ini = int(j*n)
                y_end = int((j + 1)*n)
                # goes through each box of size n x n
                states = []
                for ii in range(x_ini, x_end):
                    for jj in range(y_ini, y_end):
                        states.append(basin[ii, jj])
                # Get the number of unique states
                unique_states = list(set(states))
                num_unique_states = len(unique_states)
                if num_unique_states > 1:
                    Nb += 1
                # Count the number of occurences of each state
                occur_states = collections.Counter(states)
                # Count the total number of states
                num_states = len(states)
                # Calculate the maximum entropy (equiprobable states)
                Smax += log(num_unique_states)
                #
                for us in unique_states:
                    p_uc = occur_states[us]
                    p = p_uc/num_states
                    S[i, j] -= p*log(p)
        Sb = np.sum(S)/M**2
        Sbb = np.sum(S)/Nb
        Sb_max = Smax/M**2
        Sbb_max = Smax/Nb
    else:
        print("Invalid grid size!")
        sys.exit()

    return Sb, Sbb, Sb_max, Sbb_max

def plot_params(fontsize=20, legend_fontsize=14, axes_linewidth=1.3):
    """
    Update the parameters of the plot.

    Returns
    -------
    cmap : string
        The color map used in the colored plots.
    """
    tick_labelsize = fontsize - 3
    plt.clf()
    plt.rc('font', size=fontsize)
    plt.rc('xtick', labelsize=tick_labelsize)
    plt.rc('ytick', labelsize=tick_labelsize)
    plt.rc('legend', fontsize=legend_fontsize)
    font = {'family' : 'stix'}
    plt.rc('font', **font)
    plt.rcParams["mathtext.fontset"] = "stix"
    mpl.rcParams['axes.linewidth'] = axes_linewidth #set the value globally

    return fontsize, legend_fontsize, axes_linewidth
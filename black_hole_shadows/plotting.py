"""Functionality for plotting black hole shadows."""
import numpy as np
import matplotlib.pyplot as plt
from .orbits import Orbit
from .utils import Metric

def plot_orbits(orbits, xlim=(-5,5), ylim=(-5,5), add_box_size=False, box_size=0):
    """Plots the orbits of photons around a black hole.

    Parameters
    ----------
    orbits : list of Orbit objects
        List of Orbit objects to plot.
    xlim : tuple, optional
        Limits for the x-axis. The default is (-5,5).
    ylim : tuple, optional
        Limits for the y-axis. The default is (-5,5).   
    add_box_size : bool, optional
        Whether to add a box representing the box size used for the source.
        The default is False.
    box_size : float, optional
        Size of the box to add if add_box_size is True. The default is 0
    """
    fig, ax = plt.subplots()
    # Plotting all the photon trajectories
    for orbit in orbits:
        ax.plot(orbit.y[:,0]*np.cos(orbit.phi),orbit.y[:,0]*np.sin(orbit.phi),
                color='black', lw=0.5, alpha=1)
    # Adding the black hole and photon sphere
    black_hole = plt.Circle((0, 0), 1, color='black')
    photon_sphere = plt.Circle((0, 0), 1.5, color='grey', alpha=0.3, ec='none')
    ax.add_patch(photon_sphere)
    ax.add_patch(black_hole)
    if add_box_size:
        box = plt.Rectangle((-box_size/2, -box_size/2), box_size, box_size, 
                            color='cyan', alpha=1, fc='none')
        ax.add_patch(box)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal')
    plt.show()
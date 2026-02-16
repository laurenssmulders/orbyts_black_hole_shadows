import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from black_hole_shadows.orbits import generate_orbit, Orbit
from black_hole_shadows.utils import Metric
from black_hole_shadows.plotting import plot_orbits

# Everything in units of r_s
D = 100
n_points = 100000

def A(r):
    return 1-1/r
def B(r):
    return 1-1/r
def dAdr(r):
    return 1/r**2
def dBdr(r):
    return 1/r**2

metric = Metric(A, B, dAdr, dBdr)

# Generating 100 orbits
n_orbits = 100
orbits = []
b = np.linspace(1e-5,5,n_orbits)
for i in range(n_orbits):
    orbit = Orbit(b[i], D, metric, n_points)
    orbits.append(orbit)

plot_orbits(orbits)


import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from black_hole_shadows.orbits import generate_orbit, Orbit
from black_hole_shadows.utils import Metric
from black_hole_shadows.plotting import plot_orbits
from black_hole_shadows.sources import Source, Box

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

box = Box(box_size=10, box_points=10)
box.calculate_photon_rays(n_b=10, metric=metric, D=D, b_max=10, n_points=n_points)
source = Source(condition=lambda x: x[0] < 5, coordinates="Spherical")
box.add_source(source)
box.plot_box_orbits()
box.calculate_pixel_brightness(n_alpha=10)
box.plot_image()


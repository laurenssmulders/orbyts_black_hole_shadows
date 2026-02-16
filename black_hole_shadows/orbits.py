""" Generates photon orbits around a black hole."""
import numpy as np
from scipy.optimize import root
import matplotlib.pyplot as plt
from .utils import Metric

def generate_orbit(b, D, metric, n_points=100000):
    """Generates a photon orbit around a black hole with given metric functions 
    A and B.

    Parameters
    ----------
    b : float
        Impact parameter of the photon.
    D : float
        Distance of the observer from the black hole.
    A : function
        Metric function A(r).
    B : function
        Metric function B(r).
    dAdr : function
        Derivative of A with respect to r.
    dBdr : function
        Derivative of B with respect to r.
    n_points : int, optional
        Number of points to generate in the orbit. The default is 100000.   

    Returns
    -------
    phi : array
        Array of phi values along the orbit.
    y : array
        Array of r and dr/dphi values along the orbit.
    """
    if b >= 0:
        direction = +1
    else:
        direction = -1
    # Initial conditions
    r0 = np.sqrt(D**2 + b**2)
    phi0 = np.arctan(b/D)
    kh2 = metric.A(r0)/(r0**2*metric.B(r0))*(metric.B(r0)+D**2/b**2) # The (k/h)^2 constant to obtain the initial dr/dphi

    # Residue equation
    def res(yi):
        return np.array([
            yi[1],
            -yi[0]*metric.B(yi[0])-0.5*yi[0]**2*metric.dBdr(yi[0]) 
            + yi[0]**4*metric.B(yi[0])/(2*metric.A(yi[0]))*kh2*(
                4/yi[0]+metric.dBdr(yi[0])/metric.B(yi[0])-metric.dAdr(yi[0])/metric.A(yi[0]))
        ])

    phi = np.zeros(np.array([n_points]))
    y = np.zeros(np.array([n_points,2]))

    # Initial conditions
    phi[0] = phi0
    y[0] = np.array([r0, -r0*D/b]) # dr/dphi = -r0*D/b from the geometry of the problem
    for n in range(0,n_points-1):
        # Adapting the stepsize if the derivative becomes too large for b < 1, 
        # to avoid numerical instabilities.
        if abs(y[n,1]) > y[n,0]:
            dphi = direction*4*np.pi/n_points*y[n,0]/abs(y[n,1])
        else:
            dphi = direction*4*np.pi/n_points
        k1 = res(y[n])
        k2 = res(y[n]+dphi/2*k1)
        k3 = res(y[n]+dphi/2*k2)
        k4 = res(y[n]+dphi*k3)
        phi[n+1] = phi[n] + dphi
        y[n+1] = y[n] + dphi*(k1 + 2*k2 + 2*k3 + k4)/6
        if y[n+1,0] < 0: # If we get a negative radius, stop the integration
            print("Negative radius! After", n+1, "steps.")
            phi = phi[:n+2]
            y = y[:n+2]
            break
        if y[n+1,0] < 1: # If we get inside the event horizon, stop the integration
            print("Photon captured by the black hole after ", n+1, "steps.")
            phi = phi[:n+2]
            y = y[:n+2]
            break
        if y[n+1,0] > 1000: # If we get too far away, stop the integration
            print("Photon escaped to infinity after ", n+1, "steps.")
            phi = phi[:n+2]
            y = y[:n+2]
            break
    return phi, y

class Orbit:
    """Class to represent a photon orbit around a black hole."""
    def __init__(self, b, D, metric, n_points=100000):
        self.b = b
        self.D = D
        self.metric = metric
        self.n_points = n_points
        self.phi, self.y = generate_orbit(b, D, metric, n_points)

    




"""Generates photon orbits around a Schwarzschild black hole."""

import numpy as np
from scipy.integrate import solve_ivp
    
def calc_photon_orbit(A, B, dAdr, dBdr, r0, phi0, b, n_loops, n_points_per_loop):
    """
    Calculates photon orbits around a Schwarzschild black hole.

    Parameters
    ----------
    A : function
        Metric function A(r).
    B : function
        Metric function B(r).
    dAdr : function
        Derivative of A with respect to r.
    dBdr : function
        Derivative of B with respect to r.
    r0 : float
        Initial radius.
    phi0 : float
        Initial angle.
    b : float
        Impact parameter.
    n_loops : int
        Number of loops to calculate.
    n_points_per_loop : int
        Number of points per loop.

    Returns
    -------
    r_sol_plus : array
        Radial coordinates for the "plus" solution.
    r_sol_minus : array
        Radial coordinates for the "minus" solution.
    phi_sol_plus : array
        Angular coordinates for the "plus" solution.
    phi_sol_minus : array
        Angular coordinates for the "minus" solution.
    """


    y0_plus = r0*np.array([1, +np.sqrt(B(r0)*(r0**2/(b**2*A(r0))-1))]) # y = [r, dr/dphi]
    y0_minus = r0*np.array([1, -np.sqrt(B(r0)*(r0**2/(b**2*A(r0))-1))])

    def dydphi(phi, y):
        r = y[0]
        drdphi = y[1]
        return np.array([drdphi,-r*B(r)-0.5*r**2*dBdr(r)+r**4*B(r)/(2*b**2*A(r))*(4/r+dBdr(r)/B(r)-dAdr(r)/A(r))])

    def event_y0_less_than_1(t, y):
        return y[0] - 1

    event_y0_less_than_1.terminal = True     # stop integration
    event_y0_less_than_1.direction = -1      # only trigger when decreasing


    sol_plus_fwd = solve_ivp(dydphi, (phi0,phi0+2*n_loops*np.pi), y0_plus, events=event_y0_less_than_1, dense_output=True, t_eval=np.linspace(phi0, phi0+2*n_loops*np.pi, n_points_per_loop*n_loops))
    sol_plus_bwd = solve_ivp(dydphi, (phi0,phi0-2*n_loops*np.pi), y0_plus, events=event_y0_less_than_1, dense_output=True, t_eval=np.linspace(phi0, phi0-2*n_loops*np.pi, n_points_per_loop*n_loops))
    sol_minus_fwd = solve_ivp(dydphi, (phi0,phi0+2*n_loops*np.pi), y0_minus, events=event_y0_less_than_1, dense_output=True, t_eval=np.linspace(phi0, phi0+2*n_loops*np.pi, n_points_per_loop*n_loops))
    sol_minus_bwd = solve_ivp(dydphi, (phi0,phi0-2*n_loops*np.pi), y0_minus, events=event_y0_less_than_1, dense_output=True, t_eval=np.linspace(phi0, phi0-2*n_loops*np.pi, n_points_per_loop*n_loops))

    y_sol_plus = np.concatenate((sol_plus_bwd.y, sol_plus_fwd.y), axis=1)
    y_sol_minus = np.concatenate((sol_minus_bwd.y, sol_minus_fwd.y), axis=1)
    phi_sol_plus = np.concatenate((sol_plus_bwd.t, sol_plus_fwd.t))
    phi_sol_minus = np.concatenate((sol_minus_bwd.t, sol_minus_fwd.t))

    r_sol_plus = y_sol_plus[0]
    r_sol_minus = y_sol_minus[0]
    # Giving r and phi in both directions from the initial point
    return r_sol_plus, r_sol_minus, phi_sol_plus, phi_sol_minus

class PhotonOrbit:
    """Class to calculate photon orbits around a Schwarzschild black hole.
    
    Attributes
    ----------
    r0 : float
        Initial radius.
    phi0 : float
        Initial angle.
    b : float
        Impact parameter.
    sol : tuple
        Solution containing radial and angular coordinates.
    """
    def __init__(self, r0, phi0, b):
        self.r0 = r0
        self.phi0 = phi0
        self.b = b

    def calculate(self, A, B, dAdr, dBdr, n_loops, n_points_per_loop):
        self.sol = calc_photon_orbit(A, B, dAdr, dBdr, self.r0, self.phi0, 
                                     self.b, n_loops, n_points_per_loop)
        
class PointSource:
    """Class representing a point source of photons.
    
    Attributes
    ----------
    r : float
        Radial coordinate of the source.
    phi : float
        Angular coordinate of the source.
    """
    def __init__(self, r, phi):
        self.r = r
        self.phi = phi

    def calculate_ps_trajectories(self, A, B, dAdr, dBdr, n_trajectories, 
                                  n_loops, n_points_per_loop):
        """Calculates a set of photon trajectories from the point source."""
        self.trajectories = []
        b_values = np.linspace(0, self.r/np.sqrt(A(self.r)), n_trajectories)
        for b in b_values:
            orbit = PhotonOrbit(self.r, self.phi, b)
            orbit.calculate(A, B, dAdr, dBdr, n_loops, n_points_per_loop)
            self.trajectories.append(orbit)

    def find_matching_impact_parameter(self, target_r, target_phi):
        # Need to update this with the fact that there might be two possible matching impact parameters
        """Finds the impact parameter that leads to a trajectory passing through
        the specified (target_r, target_phi) point.
        """
        for orbit in self.trajectories:
            r_sol_plus, r_sol_minus, phi_sol_plus, phi_sol_minus = orbit.sol
            for r_sol, phi_sol in [(r_sol_plus, phi_sol_plus), (r_sol_minus, 
                                                                phi_sol_minus)]:
                for r, phi in zip(r_sol, phi_sol):
                    if np.isclose(r, target_r) and np.isclose(phi, target_phi):
                        return orbit.b
        return None
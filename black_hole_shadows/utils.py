"""General Utilities for the black hole shadow code."""

import numpy as np

class Metric:
    """Class to represent a metric."""
    def __init__(self, A, B, dAdr, dBdr):
        self.A = A
        self.B = B
        self.dAdr = dAdr
        self.dBdr = dBdr

def cartesian_to_spherical(x):
    """Converts Cartesian coordinates to spherical coordinates.

    Parameters
    ----------
    x : array-like
        Cartesian coordinates (x,y,z).

    Returns
    -------
    array
        Spherical coordinates (r, theta, phi).
    """
    r = np.sqrt(x[0]**2 + x[1]**2 + x[2]**2)
    theta = np.arccos(x[2]/r)
    alpha = np.arctan2(x[1], x[0])
    return np.array([r, theta, alpha])

def cartesian_to_cylindrical(x):
    """Converts Cartesian coordinates to cylindrical coordinates.

    Parameters
    ----------
    x : array-like
        Cartesian coordinates (x,y,z).

    Returns
    -------
    array
        Cylindrical coordinates (rho, phi, z).
    """
    rho = np.sqrt(x[0]**2 + x[1]**2)
    alpha = np.arctan2(x[1], x[0])
    z = x[2]
    return np.array([rho, alpha, z])

def spherical_to_cartesian(s):
    """Converts spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    s : array-like
        Spherical coordinates (r, theta, phi).

    Returns
    -------
    array
        Cartesian coordinates (x,y,z).
    """
    r, theta, phi = s
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.array([x, y, z])

def cylindrical_to_cartesian(c):
    """Converts cylindrical coordinates to Cartesian coordinates.

    Parameters
    ----------
    c : array-like
        Cylindrical coordinates (rho, phi, z).

    Returns
    -------
    array
        Cartesian coordinates (x,y,z).
    """
    rho, phi, z = c
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.array([x, y, z])

def spherical_to_cylindrical(s):
    """Converts spherical coordinates to cylindrical coordinates.

    Parameters
    ----------
    s : array-like
        Spherical coordinates (r, theta, phi).

    Returns
    -------
    array
        Cylindrical coordinates (rho, phi, z).
    """
    r, theta, phi = s
    rho = r * np.sin(theta)
    z = r * np.cos(theta)
    return np.array([rho, phi, z])

def cylindrical_to_spherical(c):
    """Converts cylindrical coordinates to spherical coordinates.

    Parameters
    ----------
    c : array-like
        Cylindrical coordinates (rho, phi, z).

    Returns
    -------
    array
        Spherical coordinates (r, theta, phi).
    """
    rho, alpha, z = c
    r = np.sqrt(rho**2 + z**2)
    theta = np.arctan2(rho, z)
    return np.array([r, theta, alpha])
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RBFInterpolator
from .utils import cartesian_to_spherical, spherical_to_cartesian, cylindrical_to_cartesian, spherical_to_cylindrical, cylindrical_to_spherical, cartesian_to_cylindrical
from .orbits import Orbit
from.plotting import plot_orbits

class Source:
    """Class to represent a source."""
    def __init__(self, condition, coordinates):
        self.condition = condition # Condition will be a function of a point in space returning a boolean.
        self.coordinates = coordinates
        if coordinates not in ["Cartesian", "Spherical", "Cylindrical"]:
            raise ValueError("Coordinates must be 'Cartesian', 'Spherical' or 'Cylindrical'.")
        
class Box:
    """Class to represent a box."""
    def __init__(self, box_size, box_points, viewing_angles=(0,0)):
        self.box_size = box_size
        self.box_points = box_points
        self.viewing_angles = viewing_angles # Viewing angles will be a tuple of two angles (theta, phi) representing the viewing angle of the observer in spherical coordinates. Theta is the angle from the z-axis and phi is the angle from the x-axis in the xy-plane.
        self.cartesians = np.transpose(
            np.array(
                np.meshgrid(
                    np.linspace(-box_size/2, box_size/2, box_points),
                    np.linspace(-box_size/2, box_size/2, box_points),
                    np.linspace(-box_size/2, box_size/2, box_points),
                    indexing="ij"
                    )
                ),
            [1,2,3,0]
            )
        self.sphericals = np.array(
            [cartesian_to_spherical(cartesian) 
             for cartesian in self.cartesians.reshape(-1,3)]
            ).reshape(box_points,box_points,box_points,3)
        self.cylindricals = np.array(
            [cartesian_to_cylindrical(cartesian) 
             for cartesian in self.cartesians.reshape(-1,3)]
            ).reshape(box_points,box_points,box_points,3)
        self.source = np.zeros(self.cartesians.shape[:3], dtype=bool)
        # Depending on the viewing angle, the source frame will be rotated
        self.cartesians_source_frame = (self.cartesians 
                                        @ np.array(
                                            [
                                                [np.cos(self.viewing_angles[1]), 0, np.sin(self.viewing_angles[1])], 
                                                [0, 1, 0], 
                                                [-np.sin(self.viewing_angles[1]), 0, np.cos(self.viewing_angles[1])]
                                                ]
                                                )
                                            )
        self.cartesians_source_frame = (self.cartesians_source_frame 
                                        @ np.array(
                                            [
                                                [1, 0, 0], 
                                                [0, np.cos(self.viewing_angles[0]), -np.sin(self.viewing_angles[0])], 
                                                [0, np.sin(self.viewing_angles[0]), np.cos(self.viewing_angles[0])]
                                                ]
                                                )
                                            )
        self.sphericals_source_frame = np.array(
            [cartesian_to_spherical(cartesian) 
             for cartesian in self.cartesians_source_frame.reshape(-1,3)]
            ).reshape(box_points,box_points,box_points,3)
        self.cylindricals_source_frame = np.array(
            [cartesian_to_cylindrical(cartesian) 
             for cartesian in self.cartesians_source_frame.reshape(-1,3)]
            ).reshape(box_points,box_points,box_points,3)

    def add_source(self, source):
        """Adds a source to the box."""
        if source.coordinates == "Cartesian":
            for x in range(self.cartesians_source_frame.shape[0]):
                for y in range(self.cartesians_source_frame.shape[1]):
                    for z in range(self.cartesians_source_frame.shape[2]):
                        if source.condition(self.cartesians_source_frame[x,y,z]):
                            self.source[x,y,z] = True
        elif source.coordinates == "Spherical":
            for x in range(self.sphericals_source_frame.shape[0]):
                for y in range(self.sphericals_source_frame.shape[1]):
                    for z in range(self.sphericals_source_frame.shape[2]):
                        if source.condition(self.sphericals_source_frame[x,y,z]):
                            self.source[x,y,z] = True
        elif source.coordinates == "Cylindrical":
            for x in range(self.cylindricals_source_frame.shape[0]):
                for y in range(self.cylindricals_source_frame.shape[1]):
                    for z in range(self.cylindricals_source_frame.shape[2]):
                        if source.condition(self.cylindricals_source_frame[x,y,z]):
                            self.source[x,y,z] = True
        # Finally ensuring there are no sources behind the horizon
        for x in range(self.sphericals_source_frame.shape[0]):
            for y in range(self.sphericals_source_frame.shape[1]):
                for z in range(self.sphericals_source_frame.shape[2]):
                    if self.sphericals_source_frame[x,y,z,0] < 1:
                        self.source[x,y,z] = False

    def clear_sources(self):
        """Clears the sources from the box."""
        self.source = np.zeros(self.cartesians.shape[:3], dtype=bool)

    def plot_source(self):
        """Plots the source in Cartesian coordinates."""
        x = self.cartesians[self.source][:,0]
        y = self.cartesians[self.source][:,1]
        z = self.cartesians[self.source][:,2]
        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(x,y,z, s=1)
        # Adding the black hole and photon sphere
        theta = np.linspace(0, np.pi, 100)
        phi = np.linspace(0, 2*np.pi, 100)
        black_hole_x = np.sin(theta)[:,None] * np.cos(phi)
        black_hole_y = np.sin(theta)[:,None] * np.sin(phi)
        black_hole_z = np.cos(theta)[:,None] * np.ones_like(phi)
        ax.plot(black_hole_x, black_hole_y, black_hole_z, color='black', lw=2)
        photon_sphere_x = 1.5 * np.sin(theta)[:,None] * np.cos(phi)
        photon_sphere_y = 1.5 * np.sin(theta)[:,None] * np.sin(phi)
        photon_sphere_z = 1.5 * np.cos(theta)[:,None] * np.ones_like(phi)
        ax.plot(photon_sphere_x, photon_sphere_y, photon_sphere_z, color='grey', lw=2, alpha=0.5)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")
        ax.set_box_aspect([1,1,1])
        ax.set_xlim(-self.box_size/2, self.box_size/2)
        ax.set_ylim(-self.box_size/2, self.box_size/2)
        ax.set_zlim(-self.box_size/2, self.box_size/2)
        plt.show()
            
    def calculate_photon_rays(self, n_b, metric, D, b_max, n_points=100000, verbose=False):
        """Calculates the photon rays through the box ending up on the screen.
        
        Parameters
        ----------
        n_b: int
            Number of impact parameters to consider for the rays.
        metric : Metric object
            Metric object to use for the calculations.
        D : float
            Distance of the observer from the black hole. 
            (should just be large enough to be in the asymptotically flat region)
        b_max : float
            Maximum impact parameter to consider for the rays.
            (should be a bit larger than the box size to capture all rays that 
            could potentially hit the box)
        n_points : int, optional
            Number of points to generate in the orbit. The default is 100000.
        """
        self.orbits = []
        self.b_values = np.linspace(1e-5, b_max, n_b)
        for i in range(n_b):
            orbit = Orbit(self.b_values[i], D, metric, n_points, verbose=verbose)
            self.orbits.append(orbit)
    
    def plot_box_orbits(self):
        """
        Plots the orbits of photons through the box ending up on the screen.
        """
        plot_orbits(self.orbits, 
                    xlim=(-max(self.b_values[-1],1.2*self.box_size/2), 
                          max(self.b_values[-1],1.2*self.box_size/2)), 
                    ylim=(-max(self.b_values[-1],1.2*self.box_size/2), 
                          max(self.b_values[-1],1.2*self.box_size/2)), 
                    add_box_size=True, 
                    box_size=self.box_size)
        
    def calculate_pixel_brightness(self, n_alpha):
        """Calculates the brightness of each pixel on the screen based on the 
        number of photons that hit the box and their impact parameters.
        
        Parameters
        ----------
        n_alpha : int
            Number of alpha values to consider. The number of pixels will be 
            n_alpha x n_b.
        """
        self.alpha_values = np.linspace(-np.pi, np.pi, n_alpha)
        self.pixels = np.transpose(
            np.array(
                np.meshgrid(
                    (self.b_values[:-1]+self.b_values[1:])/2,
                    (self.alpha_values[:-1]+self.alpha_values[1:])/2,
                    indexing="ij"
                    )
                ),
            (1,2,0)
            ) # Array of the pixel midpoints in (b, alpha) coordinates
        self.pixel_brightness = np.zeros(self.pixels.shape[:2])
        # Iterating over pixels
        for b_index in range(len(self.b_values)-1):
            for alpha_index in range(len(self.alpha_values)-1):
                # And iterating over points in the box to determine if they hit 
                # the pixel, i.e. whether they are enclosed by the corresponding
                # "ray volume" of the pixel. If so, add 1 to the pixel 
                # brightness.
                for x in range(self.source.shape[0]):
                    for y in range(self.source.shape[1]):
                        for z in range(self.source.shape[2]):
                            if self.source[x,y,z]: 
                                # First checking if it lies in the correct alpha
                                # slice of the box.
                                alpha = self.cylindricals[x,y,z,1]
                                if np.abs(alpha - self.pixels[b_index, alpha_index, 1]) < np.pi/n_alpha or np.abs(alpha - self.pixels[b_index, alpha_index, 1] + np.pi) < np.pi/n_alpha or np.abs(alpha - self.pixels[b_index, alpha_index, 1] - np.pi) < np.pi/n_alpha:
                                    # The checking if it lies in between the 
                                    # two corresponding orbits of the pixel.
                                    # First calculating the r and phi 
                                    # coordinates of the point in the box.
                                    r = self.sphericals[x,y,z,0]
                                    if np.abs(alpha - self.pixels[b_index, alpha_index, 1]) < np.pi/n_alpha: # i.e. the point lies in the upper half plane
                                        phi = self.sphericals[x,y,z,1]
                                    else: # i.e. the point lies in the lower half plane
                                        phi = 2*np.pi - self.sphericals[x,y,z,1]
                                    # Finally checking if the point lies in between the two orbits
                                    if ((r > np.interp(phi, self.orbits[b_index].phi, self.orbits[b_index].y[:,0]) and 
                                        r < np.interp(phi, self.orbits[b_index+1].phi, self.orbits[b_index+1].y[:,0])) or
                                        ((r < np.interp(phi+2*np.pi, self.orbits[b_index].phi, self.orbits[b_index].y[:,0]) and
                                        r > np.interp(phi+2*np.pi, self.orbits[b_index+1].phi, self.orbits[b_index+1].y[:,0])))):
                                        self.pixel_brightness[b_index, alpha_index] += 1
                # Finally normalizing the pixel brightness by the area if the 
                # pixel in the (b, alpha) plane, which is approximately 
                # (b_{i+1}^2 - b_i^2) * (alpha_{j+1} - alpha_j) / 2
                area = ((self.b_values[b_index+1]**2 - self.b_values[b_index]**2) 
                * (self.alpha_values[alpha_index+1] 
                   - self.alpha_values[alpha_index])) / 2
                self.pixel_brightness[b_index, alpha_index] /= area

    def plot_image(self, n_plot_points=1000, smoothing=0, cmap="afmhot", kernel="linear", log=True):
        """Plots the image of the box on the screen based on the pixel brightness."""
        # Flattening the arrays
        pixels_flattened = self.pixels.reshape(-1,2)
        # Recasting the angles to deal with periodicity
        pixels_recast = np.zeros([pixels_flattened.shape[0],3])
        pixels_recast[:,0] = pixels_flattened[:,0]
        pixels_recast[:,1] = np.cos(pixels_flattened[:,1])
        pixels_recast[:,2] = np.sin(pixels_flattened[:,1])
        brightness_flattened = self.pixel_brightness.flatten()
        # Interpolating the brightness values for a smooth plot
        if log:
            # Taking a log to ensure all features are visible
            rbf = RBFInterpolator(pixels_recast, np.log(1+brightness_flattened), smoothing=smoothing, kernel=kernel)
        else:
            rbf = RBFInterpolator(pixels_recast, brightness_flattened, smoothing=smoothing, kernel=kernel)
        # Creating an x,y grid for plotting
        x = self.pixels[:,:,0] * np.cos(self.pixels[:,:,1])
        y = self.pixels[:,:,0] * np.sin(self.pixels[:,:,1])
        xi = np.linspace(-np.max(self.b_values), np.max(self.b_values), n_plot_points)
        yi = np.linspace(-np.max(self.b_values), np.max(self.b_values), n_plot_points)
        xi, yi = np.meshgrid(xi, yi, indexing="ij")
        brightness_interpolated = rbf(np.transpose(np.array([np.sqrt(xi.flatten()**2 + yi.flatten()**2), xi.flatten()/np.sqrt(xi.flatten()**2 + yi.flatten()**2), yi.flatten()/np.sqrt(xi.flatten()**2 + yi.flatten()**2)])))
        # Plotting the image
        plt.figure(figsize=(8,8))
        # Taking a log to ensure all features are shown
        plt.imshow(brightness_interpolated.reshape(xi.shape).T, extent=(xi.min(), xi.max(), yi.min(), yi.max()), origin="lower", cmap=cmap)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()


        
    

        



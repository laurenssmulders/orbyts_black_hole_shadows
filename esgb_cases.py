from orbyts_black_hole_shadows.utils import Metric
from orbyts_black_hole_shadows.sources import Source, Box
import numpy as np

a_values = [1e-1]

for a in a_values:
    a_string = str(a).replace(".","p")
    def A2(r):
        return -49/(5*r)-49/(5*r**2)-137/(15*r**3)-7/(15*r**4)+26/(15*r**5)+10/(3*r**6)

    def B2(r):
        return 49/(5*r)+29/(5*r**2)+19/(5*r**3)-203/(15*r**4)-218/(15*r**5)-46/(3*r**6)

    def dA2dr(r):
        return 49/(5*r)+98/(5*r**3)+137/(5*r**4)+28/(15*r**5)-26/(3*r**6)-20/r**7

    def dB2dr(r):
        return -49/(5*r**2)-58/(5*r**3)-57/(5*r**4)+812/(15*r**5)+218/(3*r**6)+92/r**7

    def A(r):
        return (1-1/r)*(1+A2(r)*a**2)**2

    def B(r):
        return (1-1/r)*(1+B2(r)*a**2)**(-2)

    def dAdr(r):
        return 1/r**2*(1+A2(r)*a**2)**2 + 2*(1-1/r)*(1+A2(r)*a**2)*dA2dr(r)*a**2

    def dBdr(r):
        return 1/r**2*(1+B2(r)*a**2)**2 - 2*(1-1/r)*(1+B2(r)*a**2)**(-3)*dB2dr(r)*a**2

    metric = Metric(A,B,dAdr,dBdr)

    sphere = Source(condition=lambda x: x[0] < 5, coordinates="Spherical")

    # Initialising the box with a certain size. Make sure the source you have 
    # fits inside this box.
    box = Box(box_size=10, box_points=100, viewing_angles=(0,0))

    # We then calculate the photon rays for different impact parameters.
    box.calculate_photon_rays(n_b=20, metric=metric, D=1000, b_max=6.8)
    # And plotting these calculated photon rays
    box.plot_box_orbits(imsave='figures/esgb/photon_orbits_a_{a}'.format(a=a_string), show=False)

    # We add our source to the box, change "sphere" to a different variable if you
    # have defined a different source
    box.add_source(sphere)
    box.calculate_pixel_brightness(n_alpha=20)
    box.plot_image(smoothing=1, log=False, imsave='figures/esgb/image_a_{a}'.format(a=a_string)) 

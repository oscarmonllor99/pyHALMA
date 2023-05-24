# masclet_with_pyfof

This is a halo finder package for the MASCLET code (hydrodynamical N-body code designed for cosmological applications, https://doi.org/10.1111/j.1365-2966.2004.08040.x). It finds (star) particle clusters using the friends-of-friends algorithm pyfof (available at https://github.com/simongibbons/pyfof#pyfof) which uses an optimized R*-tree implemented in C++. 

Given this clusters, the code calculates the physical quantities defining the stellar haloes and does a phase-space cleaning to avoid non-desired structures (unbound clusters, philaments, etc.)
Then, it calculates for the physical stellar haloes (galaxies) many quantities of interest as: mass, star formation, gas mass, cold and bound gas mass fraction, radius, velocity dispersion, angular momentum,
center of mass, bulk velocity, progenitors, age, metallicity, fast/slow rotator quantities, shape, Sérsic index, luminosity in SDSS filters and
surface brightness in SDSS filters. On top of that, files containing the images (surface brightness and luminosity per pixel in each band) are produced.

In order to be able to "give" light to the stellar particles, we use the E-MILES public library (available at: http://research.iac.es/proyecto/miles/pages/spectral-energy-distributions-seds/e-miles.php)
which assumes some type of IMF, and according to metallicity, mass and age, produces a given spectral distribution. Since this part is computationally expensive, it is implemented in Fortran with OpenOMP
parallel directives.

The Sérsic index fitting for the light image of each galaxy is done by means of https://github.com/astropy/photutils

Just-in-time (JIT) compilation of the code and automatic parallelization is done thanks to https://numba.pydata.org/, allowing speeds comparable to Fortran or C.
Nevertheless, heavy parts of the code (for instance the gas unbinding, which has O(N^2) complexity) are implemented in Fortran with OpenOMP directives.

Tools and readers developed in https://github.com/dvallesp/masclet_framework are necessary.

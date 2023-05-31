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

Just-in-time (JIT) compilation and automatic parallelization of the Python code is done thanks to https://numba.pydata.org/, allowing high computational speeds for less computationally expensive parts. Nevertheless, the heaviest parts (for instance the gas unbinding, which has O(N^2) complexity) are implemented in Fortran with OpenOMP directives. 

Fortran is called within Python by means of F2PY: https://numpy.org/doc/stable/f2py/

Tools and readers developed in https://github.com/dvallesp/masclet_framework are necessary.

USAGE:

* CREATE/LINK NECESSARY FOLDERS:
  - simu_masclet --> directory of MASCLET simulation results
  - E-MILES --> directory of E-MILES SSP models
  - calipso_output --> where to save calipso outputs
  - halo_particles --> where to save particles indices of each halo found by the halo finder.
 
* COMPILE "compile_f2py" with ./compile_f2py

* Call ./run.sh with your desired options.


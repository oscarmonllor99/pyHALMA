# pyHALMA

This is a halo finder package for the MASCLET code (hydrodynamical N-body code designed for cosmological applications, https://doi.org/10.1111/j.1365-2966.2004.08040.x). It finds (star) particle clusters (haloes) using the friends-of-friends algorithm pyfof (available at https://github.com/simongibbons/pyfof#pyfof) which uses an optimized R*-tree implemented in C++ or either using an internal friends-of-friends algorithm fully written in python which uses scipy.spatial.KDTree. It also tries to find substructures within the FoF groups in order to split bridges between different bound stellar structures. For every stellar halo found, the main progenitors are traced back in time in parallel.

Given this clusters, the code calculates the physical quantities defining the stellar haloes and does a phase-space cleaning to avoid non-desired structures (unbound clusters, philaments, etc.)
Then, it calculates for the physical stellar haloes (galaxies) many quantities of interest as: mass, star formation, gas mass, cold and bound gas mass fraction, radius, velocity dispersion, angular momentum, center of mass, bulk velocity, progenitors, age, metallicity, fast/slow rotator quantities, shape, Sérsic index, luminosity in SDSS filters and
surface brightness in SDSS filters. On top of that, files containing the images (surface brightness and luminosity per pixel in each band) are produced.

In order to be able to "give" light to the stellar particles, we use the E-MILES public library (available at: http://research.iac.es/proyecto/miles/pages/spectral-energy-distributions-seds/e-miles.php)
which assumes some type of IMF, and according to metallicity, mass and age, produces a given spectral distribution. Since this part is computationally expensive, it is implemented in Fortran with OpenMP
parallel directives. We use the AB system for magnitudes and the SDSS passbands, as it is usual in galactic astronomy.

It is always interesting to know the properties of the dark matter haloes where the stellar haloes are living. Thus, by means of an ASOHF catalogue (see https://github.com/dvallesp/ASOHF), we search the corresponding dark matter halo for every galaxy. 

Just-in-time (JIT) compilation and automatic parallelization for the Python code is carried out thanks to https://numba.pydata.org/, allowing high computational speeds for less computationally expensive parts. Nevertheless, the heaviest parts (for instance the gas unbinding, which has O(N^2) complexity) are implemented in Fortran with OpenMP directives.

Fortran is called within Python by means of F2PY: https://numpy.org/doc/stable/f2py/

Tools and readers developed in https://github.com/dvallesp/masclet_framework are necessary in order to load MASCLET data. There are also readers for the pyHALMA outputs.

USAGE:

* CREATE/LINK NECESSARY FOLDERS:
  - simu_masclet --> directory of MASCLET simulation results
  - E-MILES --> directory of E-MILES SSP models
  - asohf_results --> where to find the ASOHF catalogue
 
* COMPILE "compile_f2py" with ./compile_f2py to create callable Fortran modules inside "fortran_modules"

* Specify the most suitable options in pyHALMA.dat and other necessary parameters, like the path to "masclet_framework".

* Call ./run.sh with your desired options.
  
* Output is written in a "pyHALMA_output" folder which is created inside your "simu_masclet" directory.


WARNINGS:

* Code should not be executed on Windows, since Python multiprocessing using the "fork" method is not available (only "spawn").

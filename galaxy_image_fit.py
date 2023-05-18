import numpy as np
import matplotlib.pyplot as plt
import pyimfit
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model
from photutils.aperture import EllipticalAperture
from scipy.optimize import curve_fit

def photutils_fit(R05, R_fit_min, R_fit_max, res, flux_2D):
    ###################################################################################
    #PHOTUTILS IS A PACKAGE THAT ALLOWS TO FIT ISOPHOTES TO A 2D IMAGE
    #IT IS BASED ON THE ALGORITHM OF JEDRZEJEWSKI (1987) AND BENDINELLI ET AL. (1990)
    #PROBLEM: CALCULATIONS DONE IN PYTHON, SO IT IS SLOW (BUT IT WORKS)
    ###################################################################################

    #FIRST, BUILD AN ELLIPSE MODEL
    #x0, y0 --> CENTRE OF THE GALAXY in pixels
    #sma --> semi-major axis in pixels
    #eps --> ellipticity
    #pa --> position angle of sma (in radians) relative to the x axis

    sma = R05/res #FROM mpc to pixels

    sma_fit_min = R_fit_min/res
    sma_fit_max = R_fit_max/res
    sma_fit_0 = (sma_fit_min + sma_fit_max)/2.

    argmax = np.argmax(flux_2D)
    x0 = np.unravel_index(argmax, flux_2D.shape)[0]
    y0 = np.unravel_index(argmax, flux_2D.shape)[1]
    geometry = EllipseGeometry(x0 = x0, y0 = y0, sma = sma, eps = 0.3, pa = 45 / 180 * np.pi)
    
    #PLOT TO CHECK
    # aper = EllipticalAperture((geometry.x0, geometry.y0), geometry.sma,
    #                         geometry.sma * (1 - geometry.eps),
    #                         geometry.pa)
    # plt.imshow(flux_2D.T, origin='lower', cmap='viridis')
    # aper.plot(color='white')
    # plt.show()

    #NOW FIT ---> MOST EXPENSIVE PART, THE MORE PIXELS, THE MORE TIME SPENT
    #SERSIC PROFILE
    def sersic(R, Re, Ie, n):
        bn = 2*n - 1/3 + 4/(405*n)
        return Ie*np.exp( -bn*( (R/Re)**(1/n) - 1 ) )
    
    minpoints = 10 # minimum number of points to fit

    try:
        ellipse = Ellipse(flux_2D, geometry)
        isolist = ellipse.fit_image(minsma=sma_fit_min, maxsma=sma_fit_max, sma0 = sma_fit_0)
        ellipticity_list = isolist.eps
        intens_list = isolist.intens
        sma_list = isolist.sma

        #PLOT TO CHECK
        # model_image = build_ellipse_model(star_density_2D_xy.shape, isolist_xy)
        # plt.imshow(model_image.T)
        # plt.show

        if len(intens_list) >= minpoints:
            #FIT THE SERSIC INDEX
            sma_list = np.array(sma_list)
            intens_list = np.array(intens_list)
            ellipticity_list = np.array(ellipticity_list)
            #NOW, FIT
            guess = [sma, np.max(intens_list), 1.]
            param, _ = curve_fit(sersic, sma_list, intens_list, p0 = guess)
            n = param[2]
            #WE SELECT THE ELLIPTICITY OF THE CLOSEST ISOPHOTE TO THE HALF LIGHT RADIUS
            eps = ellipticity_list[np.argmin(np.abs(sma_list - sma))]

            # CHECK FIT
            # plt.plot(sma_list_xy, intens_list_xy, 'o')
            # plt.plot(sma_list_xy, sersic(sma_list_xy, param_xy[0], param_xy[1], param_xy[2]))
            # plt.show()

        else:
            n = np.nan
            eps = np.nan

    except:
        n = np.nan
        eps = np.nan


    return n, eps


def pyimfit_fit(R05x, R05y, R05z, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz):
    ########################################################################################################
    # PyImfit is a Python wrapper around the (C++-based) image-fitting program Imfit (Erwin 2015)
    #
    # Calculations are done in C++, thus it is much faster than photutils, specially for 
    # images with a low number of pixels like the ones from cosmological simulations (e.g. ~128x128 at most)
    ########################################################################################################

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      WARNING         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Imfit was written to follow the standard 2D array indexing conventions of FITS, IRAF, and (e.g.) 
    # SAOimage DS9, which are 1-based and column-major. This means that the center of the first pixel (in the lower-left of the image) 
    # has coordinates (x,y) = (1.0,1.0); the lower-left corner of that pixel has coordinates (0.5,0.5), 
    # and the upper-right corner of the same pixel is at (1.5,1.5). The first coordinate (“x”) is the column number; 
    # the second (“y”) is the row number.

    # To allow one to use Imfit configuration files with PyImfit, PyImfit adopts the same column-major, 1-based indexing standard. 
    # The most obvious way this shows up is in the X0,Y0 coordinates for the centers of function sets.

    #Python (and in particular NumPy), on the other hand, is 0-based and row-major. This means that the first pixel 
    # in the image is at (0,0); it also means that the first index is the row number.

    #To translate coordinate systems, remember that a pixel with Imfit/PyImfit coordinates 
    # x,y would be found in a NumPy array at array[y0 - 1, x0 - 1].
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    #CREATE A GALAXY MODEL FOR EACH PLANE

    #x0, y0 --> CENTRE OF THE GALAXY in pixels
    #sma --> semi-major axis in pixels
    #eps --> ellipticity
    #pa --> position angle of sma (in radians) relative to the x axis

    def galaxy_model(x0, y0, pa, eps, n, I_e, r_e, Imax):
        model = pyimfit.SimpleModelDescription()
        model.x0.setValue(x0, [x0-10, x0+10]) #x0 and y0 are the centre of the galaxy
        model.y0.setValue(y0, [y0-10, y0+10]) #It is important not to give too much freedom to the centre of the galaxy !!!!
        galaxy_profile = pyimfit.make_imfit_function('Sersic') 
        galaxy_profile.PA.setValue(pa, [0, 180]) #position angle of the semi-major axis
        galaxy_profile.ell.setValue(eps, [0, 1]) #ellipticity
        galaxy_profile.n.setValue(n, [0.5, 5]) #sersic index
        galaxy_profile.I_e.setValue(I_e, [0., Imax]) #surface brightness at the effective radius
        galaxy_profile.r_e.setValue(r_e, [0.5*r_e, 1.5*r_e]) #effective radius
        model.addFunction(galaxy_profile)
        return model

    sma_xy = int(R05z/res)
    sma_xz = int(R05y/res)
    sma_yz = int(R05x/res)

    #xy plane
    argmax_xy = np.argmax(star_density_2D_xy)
    x0_xy = np.unravel_index(argmax_xy, star_density_2D_xy.shape)[1] + 1
    y0_xy = np.unravel_index(argmax_xy, star_density_2D_xy.shape)[0] + 1
    I_e_xy = 0.01*np.max(star_density_2D_xy)
    model_desc_xy = galaxy_model(x0 = x0_xy, y0 = y0_xy, pa = 95., eps = 0.3, 
                                    n = 1., I_e = I_e_xy, r_e = sma_xy,
                                    Imax = np.max(star_density_2D_xy))
    
    #xz plane
    argmax_xz = np.argmax(star_density_2D_xz)
    x0_xz = np.unravel_index(argmax_xz, star_density_2D_xz.shape)[1] + 1
    y0_xz = np.unravel_index(argmax_xz, star_density_2D_xz.shape)[0] + 1
    I_e_xz = 0.01*np.max(star_density_2D_xz)
    model_desc_xz = galaxy_model(x0 = x0_xz, y0 = y0_xz, pa = 95., eps = 0.3,
                                    n = 1., I_e = I_e_xz, r_e = sma_xz,
                                    Imax = np.max(star_density_2D_xz))    
    #yz plane
    argmax_yz = np.argmax(star_density_2D_yz)
    x0_yz = np.unravel_index(argmax_yz, star_density_2D_yz.shape)[1] + 1
    y0_yz = np.unravel_index(argmax_yz, star_density_2D_yz.shape)[0] + 1
    I_e_yz = 0.01*np.max(star_density_2D_yz)
    model_desc_yz = galaxy_model(x0 = x0_yz, y0 = y0_yz, pa = 95., eps = 0.3,
                                    n = 1., I_e = I_e_yz, r_e = sma_yz,
                                    Imax = np.max(star_density_2D_yz))
    
    #CREATE THE FITTER OBJECTS
    imfitter_xy = pyimfit.Imfit(model_desc_xy)
    imfitter_xz = pyimfit.Imfit(model_desc_xz)
    imfitter_yz = pyimfit.Imfit(model_desc_yz)

    #ELIMINATE ZEROES (NOT ALLOWED BY IMFIT), REPLACE THEM WITH NANs, WHICH ARE IGNORED BY IMFIT
    star_density_2D_xy_flat = star_density_2D_xy.flatten()
    star_density_2D_xy_flat[star_density_2D_xy_flat == 0] = np.nan
    star_density_2D_xy = star_density_2D_xy_flat.reshape(star_density_2D_xy.shape)

    star_density_2D_xz_flat = star_density_2D_xz.flatten()
    star_density_2D_xz_flat[star_density_2D_xz_flat == 0] = np.nan
    star_density_2D_xz = star_density_2D_xz_flat.reshape(star_density_2D_xz.shape)

    star_density_2D_yz_flat = star_density_2D_yz.flatten()
    star_density_2D_yz_flat[star_density_2D_yz_flat == 0] = np.nan
    star_density_2D_yz = star_density_2D_yz_flat.reshape(star_density_2D_yz.shape)

    #LOAD THE DATA
    imfitter_xy.loadData(star_density_2D_xy)
    imfitter_xz.loadData(star_density_2D_xz)
    imfitter_yz.loadData(star_density_2D_yz)

    #FIT
    solver = 'LM' #'LM' (faster but can be trapped in local minimum), 'NM' (Slower, but more robust)
    
    try:
        result_xy = imfitter_xy.doFit(solver = solver)
        converged_xy = result_xy.fitConverged
        bestfit_parameters_xy = result_xy.params
    except:
        converged_xy = False
        bestfit_parameters_xy = np.nan

    try:
        result_xz = imfitter_xz.doFit(solver = solver)
        converged_xz = result_xz.fitConverged
        bestfit_parameters_xz = result_xz.params
    except:
        converged_xz = False
        bestfit_parameters_xz = np.nan

    try:
        result_yz = imfitter_yz.doFit(solver = solver)
        converged_yz = result_yz.fitConverged
        bestfit_parameters_yz = result_yz.params
    except:
        converged_yz = False
        bestfit_parameters_yz = np.nan

    # GET SERSIC INDEX AND ELLIPTICITY
    # --> ORDER OF PARAMETERS (Sérsic profile): x0, y0, PA, ell, n, I_e, r_e
    if converged_xy:
        n_xy = bestfit_parameters_xy[4]
        eps_xy = bestfit_parameters_xy[3]
    else:
        n_xy = np.nan
        eps_xy = np.nan

    if converged_xz:
        n_xz = bestfit_parameters_xz[4]
        eps_xz = bestfit_parameters_xz[3]
    else:
        n_xz = np.nan
        eps_xz = np.nan

    if converged_yz:
        n_yz = bestfit_parameters_yz[4]
        eps_yz = bestfit_parameters_yz[3]
    else:
        n_yz = np.nan
        eps_yz = np.nan

    # PLOT
    # bestfit_model_im_xy = imfitter_xy.getModelImage() #2D array
    # bestfit_model_im_xz = imfitter_xz.getModelImage()
    # bestfit_model_im_yz = imfitter_yz.getModelImage()
    # Imshow
    # fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    # ax[0].imshow(bestfit_model_im_xy.T)
    # ax[1].imshow(bestfit_model_im_xz.T)
    # ax[2].imshow(bestfit_model_im_yz.T)
    # plt.show()

    #NOW, WE COMPUTE THE AVERAGE SERSIC INDEX AND ELLIPTICITY
    n = np.nanmean([n_xy, n_xz, n_yz]) #average ignoring nans
    eps = np.nanmean([eps_xy, eps_xz, eps_yz]) #average ignoring nans

    return n, eps


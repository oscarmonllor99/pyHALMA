import numpy as np
import matplotlib.pyplot as plt
import photutils
import pyimfit
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model
from photutils.aperture import EllipticalAperture
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit, least_squares
import sph3D



def photutils_fit(R05, R05x, R05y, R05z, R_fit_min, R_fit_max, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz):
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
    sma_xy = int(R05z/res)
    sma_xz = int(R05y/res)
    sma_yz = int(R05x/res)

    sma_fit_min = R_fit_min/res
    sma_fit_max = R_fit_max/res
    sma_fit_0 = (sma_fit_min + sma_fit_max)/2.

    argmax_xy = np.argmax(star_density_2D_xy)
    x0_xy = np.unravel_index(argmax_xy, star_density_2D_xy.shape)[0]
    y0_xy = np.unravel_index(argmax_xy, star_density_2D_xy.shape)[1]
    geometry_xy = EllipseGeometry(x0 = x0_xy, y0 = y0_xy, sma = sma, eps = 0.3, pa = 45 / 180 * np.pi)

    argmax_xz = np.argmax(star_density_2D_xz)
    x0_xz = np.unravel_index(argmax_xz, star_density_2D_xz.shape)[0]
    y0_xz = np.unravel_index(argmax_xz, star_density_2D_xz.shape)[1]
    geometry_xz = EllipseGeometry(x0 = x0_xz, y0 = y0_xz, sma = sma, eps = 0.3, pa = 45 / 180 * np.pi)

    argmax_yz = np.argmax(star_density_2D_yz)
    x0_yz = np.unravel_index(argmax_yz, star_density_2D_yz.shape)[0]
    y0_yz = np.unravel_index(argmax_yz, star_density_2D_yz.shape)[1]
    geometry_yz = EllipseGeometry(x0 = x0_yz, y0 = y0_yz, sma = sma, eps = 0.3, pa = 45 / 180 * np.pi)
    
    # PLOT TO CHECK
    # aper = EllipticalAperture((geometry_xy.x0, geometry_xy.y0), geometry_xy.sma,
    #                         geometry_xy.sma * (1 - geometry_xy.eps),
    #                         geometry_xy.pa)
    # plt.imshow(star_density_2D_xy.T, origin='lower', cmap='viridis')
    # aper.plot(color='white')
    # plt.show()

    #NOW FIT ---> MOST EXPENSIVE PART, THE MORE PIXELS, THE MORE TIME SPENT
    #SERSIC PROFILE
    def sersic(R, Re, Ie, n):
        bn = 2*n - 1/3 + 4/(405*n)
        return Ie*np.exp( -bn*( (R/Re)**(1/n) - 1 ) )
    
    minpoints = 10 # minimum number of points to fit
    #XY PLANE
    try:
        ellipse_xy = Ellipse(star_density_2D_xy, geometry_xy)
        isolist_xy = ellipse_xy.fit_image(minsma=sma_fit_min, maxsma=sma_fit_max, sma0 = sma_fit_0)
        ellipticity_list_xy = isolist_xy.eps
        intens_list_xy = isolist_xy.intens
        sma_list_xy = isolist_xy.sma

        #PLOT TO CHECK
        # model_image = build_ellipse_model(star_density_2D_xy.shape, isolist_xy)
        # plt.imshow(model_image.T)
        # plt.show

        if len(intens_list_xy) >= minpoints:
            #FIT THE SERSIC INDEX
            sma_list_xy = np.array(sma_list_xy)
            intens_list_xy = np.array(intens_list_xy)
            ellipticity_list_xy = np.array(ellipticity_list_xy)
            #NOW, FIT
            guess_xy = [sma, np.max(intens_list_xy), 1.]
            param_xy, _ = curve_fit(sersic, sma_list_xy, intens_list_xy, p0 = guess_xy)
            n_xy = param_xy[2]
            #WE SELECT THE ELLIPTICITY OF THE CLOSEST ISOPHOTE TO THE HALF LIGHT RADIUS
            eps_xy = ellipticity_list_xy[np.argmin(np.abs(sma_list_xy - sma_xy))]

            # CHECK FIT
            # plt.plot(sma_list_xy, intens_list_xy, 'o')
            # plt.plot(sma_list_xy, sersic(sma_list_xy, param_xy[0], param_xy[1], param_xy[2]))
            # plt.show()
        else:
            n_xy = np.nan
            eps_xy = np.nan

    except:
        n_xy = np.nan
        eps_xy = np.nan

    #XZ PLANE
    try:
        ellipse_xz = Ellipse(star_density_2D_xz, geometry_xz)
        isolist_xz = ellipse_xz.fit_image(minsma=sma_fit_min, maxsma=sma_fit_max, sma0 = sma_fit_0)
        ellipticity_list_xz = isolist_xz.eps
        intens_list_xz = isolist_xz.intens
        sma_list_xz = isolist_xz.sma

        #PLOT TO CHECK
        # model_image = build_ellipse_model(star_density_2D_xz.shape, isolist_xz)
        # plt.imshow(model_image.T)
        # plt.show

        if len(intens_list_xz) >= minpoints:
            #FIT THE SERSIC INDEX
            sma_list_xz = np.array(sma_list_xz)
            intens_list_xz = np.array(intens_list_xz)
            ellipticity_list_xz = np.array(ellipticity_list_xz)
            #NOW, FIT
            guess_xz = [sma, np.max(intens_list_xz), 1.]
            param_xz, _ = curve_fit(sersic, sma_list_xz, intens_list_xz, p0 = guess_xz)
            n_xz = param_xz[2]
            #WE SELECT THE ELLIPTICITY OF THE CLOSEST ISOPHOTE TO THE HALF LIGHT RADIUS
            eps_xz = ellipticity_list_xz[np.argmin(np.abs(sma_list_xz - sma_xz))]

        else:
            n_xz = np.nan
            eps_xz = np.nan

    except:
        n_xz = np.nan
        eps_xz = np.nan

    #YZ PLANE
    try:
        ellipse_yz = Ellipse(star_density_2D_yz, geometry_yz)
        isolist_yz = ellipse_yz.fit_image(minsma=sma_fit_min, maxsma=sma_fit_max, sma0 = sma_fit_0)
        ellipticity_list_yz = isolist_yz.eps
        intens_list_yz = isolist_yz.intens
        sma_list_yz = isolist_yz.sma
        
        #PLOT TO CHECK
        # model_image = build_ellipse_model(star_density_2D_yz.shape, isolist_yz)
        # plt.imshow(model_image.T)
        # plt.show

        if len(intens_list_yz) >= minpoints:
            #FIT THE SERSIC INDEX
            sma_list_yz = np.array(sma_list_yz)
            intens_list_yz = np.array(intens_list_yz)
            ellipticity_list_yz = np.array(ellipticity_list_yz)
            #NOW, FIT
            guess_yz = [sma, np.max(intens_list_yz), 1.]
            param_yz, _ = curve_fit(sersic, sma_list_yz, intens_list_yz, p0 = guess_yz)
            n_yz = param_yz[2]
            #WE SELECT THE ELLIPTICITY OF THE CLOSEST ISOPHOTE TO THE HALF LIGHT RADIUS
            eps_yz = ellipticity_list_yz[np.argmin(np.abs(sma_list_yz - sma_yz))]

        else:
            n_yz = np.nan
            eps_yz = np.nan

    except:
        n_yz = np.nan
        eps_yz = np.nan

    #NOW, WE COMPUTE THE AVERAGE SERSIC INDEX AND ELLIPTICITY
    n = np.nanmean([n_xy, n_xz, n_yz]) #average ignoring nans
    eps = np.nanmean([eps_xy, eps_xz, eps_yz]) #average ignoring nans

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



def sersic_fit(R05, R05x, R05y, R05z, R_fit_min, R_fit_max, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz, fit_mode = 'photutils'):
    #ERROR CONTROL    
    fit_modes = ['photutils', 'pyimfit']
    if fit_mode not in fit_modes:
        raise ValueError('mode must be one of %r.' % fit_modes)
    ###########################################################

    #FIT
    if fit_mode == 'photutils':
        n, eps = photutils_fit(R05, R05x, R05y, R05z, R_fit_min, R_fit_max, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz)

    # NOT WORKING YET
    if fit_mode == 'pyimfit':
        n, eps = pyimfit_fit(R05x, R05y, R05z, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz)
        
    return n, eps


def sersic_index(part_list, st_x, st_y, st_z, cx, cy, cz, R05, R05x, R05y, R05z):
    # In order to capture correctyle the slope of the profile, we need to fit between
    # the right radius. This is because the profile s not a perfect sersic, and it 
    # has a bump (or flattening) at the center and in the outer regions the
    # profile is not well defined.

    R_fit_min = 0.
    R_fit_max = 1.5*R05

    #ONLY CONSIDER PARTICLES WITHIN R_sersic
    x_pos = st_x[part_list]
    y_pos = st_y[part_list]
    z_pos = st_z[part_list]
    R_pos = ((x_pos-cx)**2 + (y_pos-cy)**2 + (z_pos-cz)**2)**0.5

    part_list = part_list[R_pos < R_fit_max]
    x_pos = x_pos[R_pos < R_fit_max]
    y_pos = y_pos[R_pos < R_fit_max]
    z_pos = z_pos[R_pos < R_fit_max]

    # NOW CONVERT TO POSITIONS BETWEEN 0, 2*R_sersic, f4(float32)
    x_pos = np.float32(x_pos - cx + R_fit_max) # kpc
    y_pos = np.float32(y_pos - cy + R_fit_max)
    z_pos = np.float32(z_pos - cz + R_fit_max)

    #DEFINING THE GRID
    partNum = np.int32(len(part_list))
    L_box = np.float32(2*R_fit_max) #kpc
    res = np.float32(      ll     )  # IMPORTANT!!!! RESOLUTION OF THE GRID FOR THE SERSIC INDEX
                                       # IT SHOULD BE -->LL<--, BUT SMALL GALAXIES ARE A PROBLEM
                                       # BIGGEST GALAXIES ARE NOT A PROBLEM, SINCE THERE IS A MAXIMUM NUMBER OF CELLS
                                       # SEE BELOW
    ncell = np.int32(min( max(L_box/res, 32), 64) )
    res = np.float32(L_box/ncell)   # RECALCULATE RES IN CASE I CHANGED IT
    kneigh = np.int32(16) # h distance in SPH kernel is calculated as the distance to the "kneigh" nearest neighbour
                          # the higher the kneigh value, the more time it will take
    # CALL FORTRAN 
    field = np.ones(partNum, dtype = np.float32) #CONSIDER ALL PARTICLES AS EQUALLY MASSIVE
    star_density_3D, _ = sph3D.sph.main(x_pos, y_pos, z_pos, L_box, L_box, L_box, field, kneigh, ncell, ncell, ncell, partNum)
    
    #NOW, SURFACE DENSITY IN EACH PLANE
    star_density_2D_xy = np.mean(star_density_3D, axis = 2)
    star_density_2D_xz = np.mean(star_density_3D, axis = 1)
    star_density_2D_yz = np.mean(star_density_3D, axis = 0)

    ##################################################################################################
    #EXTRAPOLATION TO GET A BETTER RESOLUTION, with scipy.regular_grid_interpolator
    # grid_faces = np.linspace(0, L_box, ncell+1)
    # grid_centers = (grid_faces[1:] + grid_faces[:-1])/2.

    # interp_xy = RegularGridInterpolator((grid_centers, grid_centers), star_density_2D_xy, bounds_error=False, fill_value=None, method = 'linear')
    # interp_xz = RegularGridInterpolator((grid_centers, grid_centers), star_density_2D_xz, bounds_error=False, fill_value=None, method = 'linear')
    # interp_yz = RegularGridInterpolator((grid_centers, grid_centers), star_density_2D_yz, bounds_error=False, fill_value=None, method = 'linear')

    # n_extrapolate = 256
    # res = L_box/n_extrapolate
    # grid_faces_finner = np.linspace(0, L_box, n_extrapolate+1)
    # grid_centers_finner = (grid_faces_finner[1:] + grid_faces_finner[:-1])/2.
    # X, Y = np.meshgrid(grid_centers_finner, grid_centers_finner)

    # star_density_2D_xy = interp_xy((X, Y))
    # star_density_2D_xz = interp_xz((X, Y))
    # star_density_2D_yz = interp_yz((X, Y))

    # APPLYING A GAUSSIAN FILTER TO SMOOTH THE SURFACE DENSITY
    # sfilter = 2.
    # star_density_2D_xy = gaussian_filter(star_density_2D_xy, sigma = sfilter)
    # star_density_2D_xz = gaussian_filter(star_density_2D_xz, sigma = sfilter)
    # star_density_2D_yz = gaussian_filter(star_density_2D_yz, sigma = sfilter)

    # PLOT
    # fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    # ax[0].imshow(star_density_2D_xy.T, origin = 'lower')
    # ax[1].imshow(star_density_2D_xz.T, origin = 'lower')
    # ax[2].imshow(star_density_2D_yz.T, origin = 'lower')
    # plt.show()

    #FITTING THE SERSIC PROFILE
    n, eps = sersic_fit(R05, R05x, R05y, R05z, R_fit_min, R_fit_max, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz, fit_mode = 'photutils')

    return n, eps
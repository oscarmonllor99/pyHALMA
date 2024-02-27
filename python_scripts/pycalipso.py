#  Created on Tuesday Oct 4 11:00:10 2022
#  @author: ÓSCAR MONLLOR BERBEGAL
from numba import njit, prange, get_num_threads
import numpy as np
from astropy.io import fits
from math import log10
from masclet_framework import units
############################################
from fortran_modules import calipso


### AVISOS PARA ENTENDER EL CÓDIGO:
# mag no es una magnitud. Al sumarle cfact: fmag = mag + cfact lo convertimos en una magnitud
# cfact CONVIERTE sum_s DENTRO DE mag EN UN FLUJO (dividiendo por 4pidlum^2), pasa dlum de Mpc a cm y ademas pasa la luminsodidad de Lo a erg/s
# fmag puede ser APARENTE o ABSOLUTA. si dlum = 1 parsec, entonces es ABSOLUTA


##################################################################################################
# READERS: SSP, FILTERS AND VEGA/AB CALIBRATION #
##################################################################################################
def readSSPfiles(dirssp, lstart, lend):
    print('1 - Reading SSP files')
    age_span = np.array([0.0631, 0.0708, 0.0794, 0.0891, 0.10, 0.1122, 0.1259, 0.1413, 
                        0.1585, 0.1778, 0.1995, 0.2239, 0.2512, 0.2818, 0.3162, 0.3548, 0.3981, 
                        0.4467, 0.5012, 0.5623, 0.6310, 0.7079, 0.7943, 0.8913, 1.00, 1.1220, 
                        1.2589, 1.4125, 1.5849, 1.7783, 1.9953, 2.2387, 2.5119, 2.8184, 3.1623, 
                        3.5481, 3.9811, 4.4668, 5.0119, 5.6234, 6.3096, 7.0795, 7.9433, 8.9125, 
                        10.00, 11.2202, 12.5893, 14.1254, 15.8489, 17.7828])
    
    nages = len(age_span)
    MH_span = np.array([-2.32, -1.71, -1.31, -0.71, -0.40, 0.00, 0.22])
    Z_span = np.array([0.0001, 0.0004, 0.001, 0.004, 0.008, 0.019, 0.03])
    nZ = len(Z_span)
    nw = 53689
    wavelenghts = np.zeros(nw) #MILES COVERING
    SSP = np.zeros((nages, nZ, nw))
    first_reading = True
    for bin_age in range(nages):
        for bin_Z in range(nZ):
            #CONSTRUCTING THE FILENAME
            age_chain = 'T'
            if age_span[bin_age] < 10:
                age_chain += '0'+str(age_span[bin_age])
            else:
                age_chain += str(age_span[bin_age])
            dot_at = age_chain.index('.')
            while len(age_chain[dot_at:])<5:
                age_chain += '0'
            
            Z_chain = 'Z'
            if MH_span[bin_Z] < 0:
                Z_chain += 'm'
                Z_chain += str(MH_span[bin_Z])[1:] #para evitar el signo - en la cadena
            else:
                Z_chain += 'p'
                Z_chain += str(MH_span[bin_Z])
            
            dot_at = Z_chain.index('.')
            if len(Z_chain[dot_at:])<3:
                Z_chain += '0'

            filename = dirssp+'/Eun1.30'+Z_chain+age_chain+'_iPp0.00_baseFe.fits'
            filename_w = 'calipso_files/e-miles_spectral_resolution.dat'
            data = fits.open(filename)
            if first_reading:
                data_w = open(filename_w, 'r')
                data_w.readline() #skip header
            SSP_data = data[0].data
            
            for iw in range(nw): #DATOS DE LONGITUD DE ONDA Y FLUJO
                SSP[bin_age, bin_Z, iw] = SSP_data[iw]
                if first_reading:
                    wavelenghts[iw] = float(data_w.readline().split()[0])

            data.close()
            if first_reading:
                data_w.close()
                first_reading = False
                
    w_good_index_1 = lstart < wavelenghts
    w_good_index_2 = wavelenghts < lend
    w_good_index = w_good_index_1*w_good_index_2
    wavelenghts = wavelenghts[w_good_index]
    nw = len(wavelenghts) #se actualiza nw
    new_SSP = np.zeros((nages, nZ, nw))
    new_SSP[:,:,:] = SSP[:,:, w_good_index]
    SSP = new_SSP
    print('1 -----> done')
    print('w limits: ',wavelenghts[0], wavelenghts[-1])
    return wavelenghts, SSP, age_span, Z_span, MH_span, nages, nZ, nw


def readFilters():
    print('2 - Reading filters file')
    file_name = 'calipso_files/SLOAN_SDSS.'
    flist = ['u','g','r','i'] #SDSS FILTERS TO BE READ
    nf = len(flist) #number of filters
    nlf = np.zeros(nf, dtype = int) #number of lines for each filter
    for i_f in range(nf):
        file_name2 = file_name+flist[i_f]+'.dat'
        filters_file = open(file_name2, 'r')
        line = '0'
        line_counter = 0
        while line != '':
            line = filters_file.readline()
            line_counter += 1
        filters_file.close()
        line_counter -= 1
        nlf[i_f] = line_counter

    #OBTAINING FILTER RESPONSE
    nmax = np.max(nlf)
    wf = np.zeros((nmax,nf))
    rf = np.zeros((nmax,nf))
    for i_f in range(nf):
        file_name2 = file_name+flist[i_f]+'.dat'
        filters_file = open(file_name2, 'r')
        for line in range(nlf[i_f]):
            filter_response = np.array(filters_file.readline().split(), dtype = float)
            wf[line, i_f] = filter_response[0]
            rf[line, i_f] = filter_response[1]
        filters_file.close()
    #normalize response in case the maximum is not 1
    # rmax=np.max(rf)
    # rf/=rmax
    print('2 -----> done')
    return nf, nlf, wf, rf


def readVega(zp_5556):
    print('3 - Reading Vega SED')
    #AB System 
    vname = 'calipso_files/vega_bgt2014.dat'
    nv = 0 #number of lines (Wavelenghts)
    vega_file = open(vname, 'r')
    line = '0'
    while line != '':
        line = vega_file.readline()
        nv += 1
    nv -= 1 
    vega_file.close()

    #Reading flux reference
    wv = np.zeros(nv)
    fv = np.zeros(nv)       
    vega_file = open(vname, 'r')
    for iv in range(nv):
        vega_data = np.array(vega_file.readline().split(), dtype = float)
        wv[iv] = vega_data[0]
        fv[iv] = vega_data[1]
  
    #NOW INTERPOLATE LINEARLY THE VALUE FOR LAMBDA = 5556. Easy with scipy
    # lin_interp = interp1d(wv, fv, kind = 'linear')
    # f5556 = lin_interp(5556.)
    # #Scale flux to be consistent with zp_5556
    # scalef=zp_5556/f5556
    # fv=fv*scalef

    #fv[:] = 1/(wv[:]*wv[:])

    print('3 -----> done')
    return wv, fv, nv
##################################################################################################



##################################################################################################
# MAGNITUDE CALCULATION FUNCTIONS, NOW IMPLEMENTED IN FORTRAN
##################################################################################################
@njit
def from_ws_to_wfilt(fs, wfilt, nfilt, ws):
    fs_new = np.zeros(nfilt)
    for i_f in range(nfilt):
        i_w = np.argmin(np.abs(ws - wfilt[i_f]))
        #linaer interpolation
        if ws[i_w] < wfilt[i_f]:
            fs_new[i_f] = ( fs[i_w] + (fs[i_w+1] - fs[i_w]) * 
                            (wfilt[i_f] - ws[i_w])/(ws[i_w+1] - ws[i_w]) )
        else:
            fs_new[i_f] = ( fs[i_w-1] + (fs[i_w] - fs[i_w-1]) * 
                            (wfilt[i_f] - ws[i_w-1])/(ws[i_w] - ws[i_w-1]) )

    return fs_new

@njit(parallel = True)
def trapecio(f, x):
    sum = 0.
    for ix in prange(len(x)-1):
        sum += (x[ix+1] - x[ix])*(f[ix] + f[ix+1])/2
    return sum

@njit
def mymagnitude(wfilt, rfilt, nfilt, ns, ws, fs):
    #IN ORIGINAL CODE
    #adapted from Cardiel's routine in the photometry code
    #it computes mag by convolving the spectrum (and Vega sed) with filter reposnse
    #by using trapezoidal rule
    #instead of interpolating arrays over lambda filter it gets
    #always the finest deltaL and interpolates on this

    #IN THIS CODE I USE trapezoidal rule but first I need to adapt (indices) ws and fs to wfilt
    wfilt = wfilt[:nfilt]
    rfilt = rfilt[:nfilt]
    fs_new = from_ws_to_wfilt(fs, wfilt, nfilt, ws)

    #CONVOLUCIONAMOS Y APLICAMOS LA REGLA DEL TRAPECIO PARA INTEGRAR
    sum_s = trapecio(wfilt*fs_new*rfilt, wfilt) 
    ###################################################################  ---- 
    sum_v = trapecio(0.11*rfilt/wfilt, wfilt) # AÇÒ ESTÀ MAL, MIRAR LA RUTINA DE FORTRAN!!! QUE ES LA QUE S'UTILITZA!!!!!
    mag = -2.5*log10(sum_s/sum_v) 
    #notar SUM_S es luminosidad y SUM_V es flujo, esto aún no es una magnitud aparente, hasta que no se le sume cfact, que convierte sum_s a flujo a una distancia DLUM
    return sum_s, mag

@njit
def mag_v1_0(ws, fs, ns, fmag, flux, nf, nlf, wf, rf, dlum, zeta):

    # unit conversion from lum[erg/s/A] to flux[erg/s/A/cm²] for sum_s, which is in units of  SOLAR LUMINOSITY
    cfact = 5. * log10(1.7685*1e8 * dlum ) # ESTO CONVIERTE sum_s DENTRO DE mag EN UN FLUJO (dividiendo por 4pidlum^2)
                                            # pasa dlum de MPC a CM y ademas pasa de Lo a 3.826*10^33

    # select filter in the lambda range of the spectrum and compute mag
    ws_min=ws[0]
    ws_max=fs[-1]

    for i_f in range(nf):
        if wf[0, i_f] > ws_min and wf[nlf[i_f]-1, i_f] < ws_max:
            sum_s, mag = mymagnitude(wf[:,i_f], rf[:,i_f], nlf[i_f], ns, ws, fs)
            fmag[i_f] = mag+cfact  # MAGNITUD ABSOLUTA
            flux[i_f] = sum_s #FLUJO



##################################################################################################
##################################################################################################


#FIND GALAXY TOTAL SPECTRUM, MAIN FUNCTION
@njit
def make_light(npart, mass, age, met, wavelenghts, SSP, age_span, Z_span, nw, nZ, nages, istart, iend, disp, lumg, 
               tam_i, tam_j, vel, nx, ny, clight):
    lump = np.zeros(npart)
    mass_cell = np.zeros((nx, ny))
    flux_cell = np.zeros((nx, ny, nw)) #clean flux
    flux_cell_sig = np.zeros((nx, ny, nw)) #shifted Doppler flux 
    fluxtot = np.zeros(nw) #flujo total de la galaxia
    vell_malla = np.zeros((nx, ny)) #velocidad media pesada en luminosidad en cada celda
    vell2_malla = np.zeros((nx, ny)) #velocidad al cuadrado media pesada en luminosidad en cada celda
    sigl_malla = np.zeros((nx, ny)) #dispersion de velocidades pesada en luminosidad para cada celda
    lum_malla = np.zeros((nx, ny)) #luminosidad de la celda
    twl = np.zeros((nx, ny)) #edad media pesada en luminosidad de la celda
    Zwl = np.zeros((nx, ny)) #metalicidad media pesada en luminosidad de la celda
    #################################################################################### LOOP OVER PARTICLES
    for ip in range(npart):
        dage = np.abs(age_span - age[ip])
        iage = np.argmin(dage)
        dmet = np.abs(Z_span - met[ip])
        imet = np.argmin(dmet)

        tam_ii=tam_i[ip]
        tam_jj=tam_j[ip]

        shift = 1.+vel[ip]/clight #desplazamiento doppler 
        
        lump[ip]=lumg[iage,imet]*mass[ip] #flujo que le corresponde a la partícula

        # ----- shift spectra, both wavel and fluxes by shift --> wave2, ssp2
        ssp2 = np.zeros(nw)
        wave2 = np.zeros(nw)
        for iw in range(istart, iend):
            waves = wavelenghts[iw]*shift #flujo y longitud de onda desplazadas
            fluxs = SSP[iage, imet, iw]/shift
            lpix = int((waves - wavelenghts[0])/disp)
            if istart <= lpix <= iend: #if the shifted lambda is outside the SSP lambdascale I don't consider it
                wave2[lpix] = waves
                ssp2[lpix] = fluxs
            # ---------- compute spectra for each  pixel
            fluxtot[iw] += mass[ip]*SSP[iage, imet, iw]
            flux_cell_sig[tam_ii, tam_jj, iw] += mass[ip]*ssp2[iw]
            flux_cell[tam_ii, tam_jj, iw] += mass[ip]*SSP[iage, imet, iw]
            
        # ---------- Quantities using luminosity as weight
        mass_cell[tam_ii, tam_jj] += mass[ip]
        vell_malla[tam_ii, tam_jj]+=lump[ip]*vel[ip]
        vell2_malla[tam_ii, tam_jj]+=lump[ip]*vel[ip]**2
        lum_malla[tam_ii, tam_jj]+=lump[ip]
        twl[tam_ii, tam_jj]+=lump[ip]*age[ip]
        Zwl[tam_ii, tam_jj]+=lump[ip]*met[ip]

    #################################################################### END LOOP OVER PARTICLES

    # ---------- Completing quantities using luminosity as weight
    for tam_ii in range(nx):
        for tam_jj in range(ny):
            if mass_cell[tam_ii, tam_jj]>0:
                vell_malla[tam_ii, tam_jj] /= lum_malla[tam_ii, tam_jj]
                vell2_malla[tam_ii, tam_jj] /= lum_malla[tam_ii, tam_jj]
                twl[tam_ii, tam_jj] /= lum_malla[tam_ii, tam_jj]
                Zwl[tam_ii, tam_jj] /= lum_malla[tam_ii, tam_jj]
    
    #------ Calculating velocity dispersion weightened with luminosity 
    for ip in range(npart):
        tam_ii=tam_i[ip]
        tam_jj=tam_j[ip]
        sigl_malla[tam_ii, tam_jj]+= ((vel[ip]-vell_malla[tam_ii, tam_jj])**2)*lump[ip]

    for tam_ii in range(nx):
        for tam_jj in range(ny):
            if mass_cell[tam_ii, tam_jj]>0:
                sigl_malla[tam_ii, tam_jj] /= lum_malla[tam_ii, tam_jj]

    return flux_cell, flux_cell_sig, fluxtot, vell_malla, sigl_malla, lum_malla, twl, Zwl, vell2_malla


@njit
def magANDfluxes(wavelenghts, nw, nf, nlf, wf, rf, wv, fv, nv, dlum, nx, ny, flux_cell, area_arc):
    SBf = np.zeros((nx, ny, nf)) #BRILLO SUPERFICIAL DE CADA CELDA EN mag/arcsec^2
    magf = np.zeros((nx, ny, nf)) #MAGNITUD APARENTE DE CADA CELDA A UNA DISTANCIA DLUM
    fluxf = np.zeros((nx, ny, nf)) #LUMINOSIDAD INTRÍNSECA DE CADA CELDA
    for tam_ii in range(nx):
        for tam_jj in range(ny):
            fmag = np.zeros(nf)
            flux = np.zeros(nf)
            mag_v1_0(wavelenghts, flux_cell[tam_ii,tam_jj,:], nw, fmag, flux, nf, nlf, wf, rf, wv, fv, nv, dlum)
            SBf[tam_ii,tam_jj,:] = fmag[:]+2.5*log10(area_arc) #SB a partir de la magnitud aparente y el area en arcsec^2
            fluxf[tam_ii, tam_jj, :] = flux[:]
            magf[tam_ii, tam_jj, :] = fmag[:]
    
    return SBf, magf, fluxf


#CALCULATING MEAN QUANTITIES IN EACH CELL
@njit
def mean_mesh2D(nx, ny, npart, mass, met, age, vel, tam_i, tam_j):
    num_cell = np.zeros((nx,ny), dtype = np.int64)
    mass_cell = np.zeros((nx,ny))
    sig_cell = np.zeros((nx,ny))
    twm = np.zeros((nx,ny))
    tmed = np.zeros((nx,ny))
    Zwm = np.zeros((nx,ny))
    Zmed = np.zeros((nx,ny))
    VCM_malla = np.zeros((nx, ny))
    for ip in range(npart):
        tam_ii = tam_i[ip]
        tam_jj = tam_j[ip]
        num_cell[tam_ii, tam_jj] += 1
        mass_cell[tam_ii, tam_jj] += mass[ip]
        VCM_malla[tam_ii, tam_jj] += vel[ip]
        twm[tam_ii, tam_jj] += mass[ip]*age[ip]
        tmed[tam_ii, tam_jj] += age[ip]
        Zwm[tam_ii, tam_jj] += mass[ip]*met[ip]
        Zmed[tam_ii, tam_jj] += met[ip]

    for tam_ii in range(nx):
        for tam_jj in range(ny):
            if num_cell[tam_ii, tam_jj]>0:
                VCM_malla[tam_ii, tam_jj] /= num_cell[tam_ii, tam_jj]
                twm[tam_ii, tam_jj] /= mass_cell[tam_ii, tam_jj]
                Zwm[tam_ii, tam_jj] /= mass_cell[tam_ii, tam_jj]
                tmed[tam_ii, tam_jj] /= num_cell[tam_ii, tam_jj]
                Zmed[tam_ii, tam_jj] /= num_cell[tam_ii, tam_jj]

    #Sigma
    for ip in range(npart):
        tam_ii = tam_i[ip]
        tam_jj = tam_j[ip]
        sig_cell[tam_ii, tam_jj] += ((vel[ip]-VCM_malla[tam_ii, tam_jj])**2)

    for tam_ii in range(nx):
        for tam_jj in range(ny):
            if num_cell[tam_ii, tam_jj]>0:
                sig_cell[tam_ii, tam_jj] /= num_cell[tam_ii, tam_jj]

    return num_cell, mass_cell, VCM_malla, sig_cell, twm, tmed, Zwm, Zmed


@njit
def put_particles_in_grid(grid_centers, x, y, z):
    npart = len(x)
    which_cell_x = np.zeros(npart, dtype = np.int32)
    which_cell_y = np.zeros(npart, dtype = np.int32)
    which_cell_z = np.zeros(npart, dtype = np.int32)
    for ip in range(npart):
        which_cell_x[ip] = np.argmin(np.abs(grid_centers-x[ip]))
        which_cell_y[ip] = np.argmin(np.abs(grid_centers-y[ip]))
        which_cell_z[ip] = np.argmin(np.abs(grid_centers-z[ip]))

    return which_cell_x, which_cell_y, which_cell_z


def main(calipso_input, star_particle_data, ncell, vel_LOS, tam_i, tam_j, effective_radius, rete):

    ####### INPUT FORMAT
    # ncell is the number of cells in each direction (x,y,z)
    # vel_LOS is the line of sight velocity of each particle
    # tam_i, tam_j are the particle indices in the grid for each particle
    # effective_radius is the effective radius of the galaxy in kpc

    [CLIGHT, WAVELENGHTS, SSP, 
     AGE_SPAN, Z_SPAN, MH_SPAN, N_AGES, N_Z, N_W, 
     N_F, N_LINES_FILTERS, W_FILTERS, RESPONSE_FILTERS, 
     USUN, GSUN, RSUN, ISUN,
     I_START, I_END, DISP, LUMG, 
     zeta, dlum, arcsec2kpc, area_arc] = calipso_input

    [npart, mass, met, age] = star_particle_data

    # Finding fluxes in each cell and luminosity weighted quantities
    (
     flux_cell, flux_cell_sig, fluxtot, vell_malla, 
     sigl_malla, lum_malla, twl, Zwl, vell2_malla
     ) = calipso.calipso.make_light(get_num_threads(), npart, mass, age, met, WAVELENGHTS, 
                                    SSP, AGE_SPAN, Z_SPAN,N_W, N_Z, N_AGES,
                                    I_START + 1, I_END + 1, DISP, LUMG, tam_i, tam_j, 
                                    vel_LOS, ncell, ncell, CLIGHT)
    
    # Magnitudes and fluxes through filters in each cell. Images.
    # First: total quantities
    fmag_tot, flux_tot = calipso.calipso.mag_v1_0(WAVELENGHTS, fluxtot, N_W, 
                                                N_F, N_LINES_FILTERS, np.max(N_LINES_FILTERS), W_FILTERS, RESPONSE_FILTERS, 
                                                dlum, zeta) 
    
    lumtotu = flux_tot[0]
    lumtotg = flux_tot[1]
    lumtotr = flux_tot[2]
    lumtoti = flux_tot[3]

    area_halo_com = np.pi*effective_radius**2 # comoving area of the halo in kpc^2
    area_halo_pys = area_halo_com * rete** 2 *units.length_to_mpc**2 # physical area of the halo in kpc^2
    area_halo_arc = area_halo_pys/(arcsec2kpc*arcsec2kpc) # physical area of the halo in arcsec^2

    sb_u_tot = fmag_tot[0]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 u filter
    sb_g_tot = fmag_tot[1]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 g filter
    sb_r_tot = fmag_tot[2]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 r filter
    sb_i_tot = fmag_tot[3]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 i filter

    # sb_u_tot -2.5*log10(flux_tot[0]/(4*np.pi*(dlum*1e6)**2)) + USUN + 21.572
    # sb_g_tot -2.5*log10(flux_tot[1]/(4*np.pi*(dlum*1e6)**2)) + GSUN + 21.572
    # sb_r_tot -2.5*log10(flux_tot[2]/(4*np.pi*(dlum*1e6)**2)) + RSUN + 21.572
    # sb_i_tot -2.5*log10(flux_tot[3]/(4*np.pi*(dlum*1e6)**2)) + ISUN + 21.572

    gr = fmag_tot[1]-fmag_tot[2] # color g-r
    ur = fmag_tot[0]-fmag_tot[2] # color u-r

    #Second: same quantities but for each cell and "visibility"

    sbf, magf, fluxf = calipso.calipso.magandfluxes(get_num_threads(), WAVELENGHTS, N_W, N_F, 
                                                    N_LINES_FILTERS, np.max(N_LINES_FILTERS),
                                                    W_FILTERS, RESPONSE_FILTERS,
                                                    dlum, zeta, ncell, ncell, flux_cell, area_arc)

    sbf[fluxf==0.] = np.nan # if flux is zero, magnitudes are undefined
    magf[fluxf==0.] = np.nan

    # Central surface brightness in each filter
    central_sb_u = np.nanmin(sbf[:,:,0])
    central_sb_g = np.nanmin(sbf[:,:,1])
    central_sb_r = np.nanmin(sbf[:,:,2])
    central_sb_i = np.nanmin(sbf[:,:,3])


    # Third: same quantities but for each cell and "visibility" CONSIDERING DOPPLER SHIFT
    # SBfdoppler, magfdoppler, fluxfdoppler = pycalipso.magANDfluxes(wavelenghts, nw, nf, nlf, wf, rf, wv, fv, nv,
    #                                                                 dlum, ncell, ncell, flux_cell_sig, area_arc)
    # SBfdoppler[fluxfdoppler==0.] = np.nan
    # magfdoppler[fluxfdoppler==0.] = np.nan

    return (fluxtot,
            lumtotu, lumtotg, lumtotr, lumtoti, 
            sb_u_tot, sb_g_tot, sb_r_tot, sb_i_tot, 
            central_sb_u, central_sb_g, central_sb_r, central_sb_i,
            gr, ur,
            sbf, magf, fluxf)
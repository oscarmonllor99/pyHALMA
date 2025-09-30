! Created on Mon May 22 2023
! In this module the heaviest parts of the CALIPSO code are implemented

MODULE calipso
    implicit none

contains

    SUBROUTINE make_light(ncores, npart, mass, age, met, wavelenghts, SSP, age_span, Z_span, nw, nZ, &
                            nages, istart, iend, disp, lumg, tam_i, tam_j, vel, nx, ny, clight, &
                            flux_cell, flux_cell_sig, fluxtot, vell_malla, sigl_malla, lum_malla, &
                            twl, Zwl, vell2_malla)
    use omp_lib
    implicit none
    !input
    integer :: ncores, npart, nw, nZ, nages, istart, iend, nx, ny
    real, dimension(npart) :: mass, age, met, vel
    integer, dimension(npart) :: tam_i, tam_j
    real, dimension(nw) :: wavelenghts
    real, dimension(nages, nZ, nw) :: SSP
    real, dimension(nages, nZ) :: lumg
    real :: disp, clight
    real, dimension(nages) :: age_span
    real, dimension(nZ) :: Z_span

    !subroutine variables
    integer :: ip, iage, imet, tam_ii, tam_jj, iw, lpix
    real :: shift, waves, fluxs
    real, dimension(nages):: dage
    real, dimension(nZ):: dmet
    real, dimension(npart) :: lump
    

    !output
    real, dimension(nx, ny) :: lum_malla, vell_malla, vell2_malla, sigl_malla, twl, Zwl
    real, dimension(nx, ny, nw) :: flux_cell, flux_cell_sig
    real, dimension(nw) :: fluxtot, ssp2, wave2

    ! FLAGS FOR THE PYTHON-FORTRAN WRAPPER

    !f2py intent(in) ncores, npart, mass, age, met, wavelenghts, SSP, age_span, Z_span, nw, nZ
    !f2py intent(in) nages, istart, iend, disp, lumg, tam_i, tam_j, vel, nx, ny, clight
    !f2py intent(out) flux_cell, flux_cell_sig, fluxtot, vell_malla, sigl_malla, lum_malla, twl, Zwl, vell2_malla

    !f2py depend(nx, ny)  lum_malla, vell_malla, vell2_malla, sigl_malla, twl, Zwl
    !f2py depend(nx, ny, nw) flux_cell, flux_cell_sig
    !f2py depend(npart) mass, age, met, vel, tam_i, tam_j
    !f2py depend(nw) wavelenghts, fluxtot
    !f2py depend(nages, nZ, nw) SSP
    !f2py depend(nages, nZ) lumg
    !f2py depend(nages) age_span
    !f2py depend(nZ) Z_span

    call OMP_SET_NUM_THREADS(ncores)

    ! initialize variables
    flux_cell(:,:,:) = 0.
    flux_cell_sig(:,:,:) = 0.
    fluxtot(:) = 0.
    vell_malla(:,:) = 0.
    sigl_malla(:,:) = 0.
    lum_malla(:,:) = 0.
    twl(:,:) = 0.
    Zwl(:,:) = 0.
    vell2_malla(:,:) = 0.

    ! !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(iage, imet, tam_ii, tam_jj, ip, iw, &
    ! !$OMP                                  lpix, shift, waves, fluxs, lump, dage, &
    ! !$OMP                                  dmet, ssp2, wave2)
    ! !$OMP DO REDUCTION(+: fluxtot, flux_cell, flux_cell_sig, vell_malla, sigl_malla, lum_malla, &
    ! !$OMP                 twl, Zwl, vell2_malla)
    do ip = 1, npart
        dage = abs(age_span - age(ip))
        iage = minloc(array = dage, dim = nages)
        dmet = abs(Z_span - met(ip))
        imet = minloc(array = dmet, dim = nZ)

        tam_ii = tam_i(ip)
        tam_jj = tam_j(ip)

        shift = 1. + vel(ip)/clight !doppler shift at low velocities

        lump(ip) = lumg(iage, imet)*mass(ip) !particle luminosity in g-band

        ! ----- CALCULATING SPECTRA, DOPPLER SHIFTED AND WITHOUT IT
        ssp2(:) = 0.
        wave2(:) = 0.

        do iw = istart, iend
            waves = wavelenghts(iw)*shift
            fluxs = SSP(iage, imet, iw)/shift
            lpix = int((waves - wavelenghts(1))/disp) + 1

            if ( (istart <= lpix) .and. (lpix <= iend) ) then
                wave2(lpix) = waves
                ssp2(lpix) = fluxs
            endif
            
            fluxtot(iw) = fluxtot(iw) + mass(ip)*SSP(iage, imet, iw)
            flux_cell(tam_ii, tam_jj, iw) = flux_cell(tam_ii, tam_jj, iw) + mass(ip)*SSP(iage, imet, iw)
            flux_cell_sig(tam_ii, tam_jj, iw) = flux_cell_sig(tam_ii, tam_jj, iw) + mass(ip)*ssp2(iw)
        enddo

        ! --------------- quantities using luminosity as weight
        lum_malla(tam_ii, tam_jj) = lum_malla(tam_ii, tam_jj) + lump(ip)
        vell_malla(tam_ii, tam_jj) = vell_malla(tam_ii, tam_jj) + lump(ip)*vel(ip)
        vell2_malla(tam_ii, tam_jj) = vell2_malla(tam_ii, tam_jj) + lump(ip)*vel(ip)**2
        twl(tam_ii, tam_jj) = twl(tam_ii, tam_jj) + lump(ip)*age(ip)
        Zwl(tam_ii, tam_jj) = Zwl(tam_ii, tam_jj) + lump(ip)*met(ip)  
    enddo
    ! !$OMP END DO
    ! !$OMP END PARALLEL

    ! --------------- completing quantities using luminosity as weight

    do tam_jj = 1,ny
    do tam_ii = 1,nx
        if (lum_malla(tam_ii, tam_jj) > 0) then
            vell_malla(tam_ii, tam_jj) = vell_malla(tam_ii, tam_jj)/lum_malla(tam_ii, tam_jj)
            vell2_malla(tam_ii, tam_jj) = vell2_malla(tam_ii, tam_jj)/lum_malla(tam_ii, tam_jj)
            twl(tam_ii, tam_jj) = twl(tam_ii, tam_jj)/lum_malla(tam_ii, tam_jj)
            Zwl(tam_ii, tam_jj) = Zwl(tam_ii, tam_jj)/lum_malla(tam_ii, tam_jj)
        endif
    enddo
    enddo

    ! ------ Calculating velocity dispersion weightened with luminosity 

    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(iage, imet, ip, lump, tam_ii, tam_jj)
    !$OMP DO REDUCTION(+: sigl_malla)
    do ip = 1, npart
        tam_ii = tam_i(ip)
        tam_jj = tam_j(ip)
        sigl_malla(tam_ii, tam_jj) = sigl_malla(tam_ii, tam_jj) &
                                     + lump(ip)*(vel(ip) - vell_malla(tam_ii, tam_jj))**2
    enddo
    !$OMP END DO
    !$OMP END PARALLEL

    do tam_jj = 1,ny
    do tam_ii = 1,nx   
        if (lum_malla(tam_ii, tam_jj) > 0) then
            sigl_malla(tam_ii, tam_jj) = sigl_malla(tam_ii, tam_jj)/lum_malla(tam_ii, tam_jj)
        endif
    enddo
    enddo
    
    END SUBROUTINE





    SUBROUTINE trapecio(f, x, n, sum) !trapezoidal rule
    implicit none
    real :: sum 
    integer :: ix, n
    real, dimension(n) :: f, x
    sum = 0.
    do ix = 1,n-1
        sum = sum + (x(ix+1) - x(ix))*(f(ix+1) + f(ix))/2
    enddo
    END SUBROUTINE



    SUBROUTINE from_ws_to_wfilt(fs, wfilt, nfilt, ws, ns, fs_new)
    implicit none
    ! input
    integer :: nfilt, ns
    real, dimension(nfilt) :: wfilt !wavelenghts of the filters
    real, dimension(ns) :: ws, fs !wavelenghts and spectra of the SSP
    
    ! local
    integer :: i_f, i_w

    ! output
    real, dimension(nfilt) :: fs_new !spectra of the SSP in the filter wavelenghts
    
    fs_new(:) = 0.
    do i_f = 1,nfilt
        !find between which wavelenghts of the SSP is the filter wavelength
        i_w = minloc(abs(ws - wfilt(i_f)), dim = 1)
        !linear interpolation with the two closest wavelenghts of the SSP
        if (ws(i_w) < wfilt(i_f)) then
            fs_new(i_f) = fs(i_w) + (fs(i_w+1) - fs(i_w)) * &
                          (wfilt(i_f) - ws(i_w))/(ws(i_w+1) - ws(i_w))
        else
            fs_new(i_f) = fs(i_w-1) + (fs(i_w) - fs(i_w-1)) * &
                          (wfilt(i_f) - ws(i_w-1))/(ws(i_w) - ws(i_w-1))
        endif
    enddo
    END SUBROUTINE


    
    SUBROUTINE mymagnitude(wfilt, rfilt, nfilt, ns, ws, fs, sum_s, mag)
    implicit none
    ! input
    integer :: nfilt, ns
    real, dimension(nfilt) :: wfilt, rfilt !wavelenghts and response functions of each filter
    real, dimension(ns) :: ws, fs !wavelenghts and spectra of the SSP

    ! local
    real, dimension(nfilt) :: fs_new !spectra of the SSP 

    ! output
    real :: intup, intdown, sum_s, mag !flux of the SSP and 3631 Jy constant SED in the filter wavelenghts and magnitude in vega system

    ! interpolation of the spectra of the SSP and Vega in the filter wavelenghts
    call from_ws_to_wfilt(fs, wfilt, nfilt, ws, ns, fs_new)

    ! convolution with the response function of the filter
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ! mag_AB_bandpass = -2.5*log10(Int1/Int2) - 2.4

    !!! In lambda, the integrals are:
    ! Int1 = Integral(f_lambda * R_lambda * lambda * dlambda)
    ! Int2 = Integral(R_lambda / lambda * dlambda) with lambda in Amstrongs and f_lambda in erg/s/cm^2/Ams
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    call trapecio(f = fs_new*rfilt*wfilt, x = wfilt, n = nfilt, sum = intup)
    call trapecio(f = rfilt/wfilt, x = wfilt, n = nfilt, sum = intdown)
    
    mag = -2.5*log10(intup/intdown) - 2.4 

    !!!!! One can also use the isophotal frecuency of the bandpass and evaluate the monocromatic AB magnitude:
    ! mag_AB_monocromatic = -2.5*log10(F_nu) - 48.6, with F_nu in erg/s/cm^2/Hz
    !!!!! In theory, both ways of calculating the magnitude, should give the same result.

    ! NOW CALCULATING THE TOTAL FLUX IN THIS BAND:
    call trapecio(f = fs_new*rfilt, x = wfilt, n = nfilt, sum = intup)
    sum_s = intup

    END SUBROUTINE




    SUBROUTINE mag_v1_0(ws, fs, ns, fmag, flux, nf, nlf, nmaxf, wf, rf, dlum, zeta)
    implicit none
    ! input
    integer :: ns, nf, nmaxf
    integer, dimension(nf) :: nlf !number of lines of each filter
    real, dimension(nmaxf,nf) :: wf, rf !wavelenghts and response functions of each filter
    real, dimension(ns) :: ws, fs !wavelenghts and spectra of the SSP
    real :: dlum !luminosity distance
    real :: zeta !redshift

    ! output
    real, dimension(nf) :: fmag, flux !magnitudes and fluxes of each filter

    ! local
    real :: cfact, ws_min, ws_max, sum_s, mag
    integer :: i_f
    real, dimension(ns) :: ws_k, fs_k !inverse k-correction
    
    !f2py intent(in) ws, fs, ns, nf, nlf, nmaxf, wf, rf, wv, fv, nv, dlum, zeta
    !f2py intent(out) fmag, flux

    !f2py depend(nf) nlf, fmag, flux
    !f2py depend(ns) ws, fs
    !f2py depend(nmaxf, nf) wf, rf
    !f2py depend(nv) wv, fv

    ! unit conversion from lum[erg/s/A] to flux[erg/s/cm²/A] for sum_s, which is in units of  SOLAR LUMINOSITY
    cfact = 5. * log10(1.7685*1e8 * dlum ) !ESTO CONVIERTE sum_s DENTRO DE mag EN UN FLUJO (dividiendo por 4pidlum^2)
                                            !pasa dlum de MPC a CM y ademas pasa de Lo a 3.826*10^33
    
    !Inverse K-CORRECTION, that is, the shift of the filter wavelenghts due to the redshift of the galaxy
    ! ws_k(:)=ws(:)*(1.+zeta)
    ! fs_k(:)=fs(:)/(1.+zeta)

    !select filter in the lambda range of the spectrum and compute mag
    ws_min = ws(1)
    ws_max = ws(ns)

    ! c in ams
    do i_f=1,nf
        if ( ( wf(1,i_f) > ws_min ) .and. ( wf(nlf(i_f),i_f) < ws_max ) ) then
            sum_s = 0.
            mag = 0.
            call mymagnitude(wf(:nlf(i_f), i_f), rf(:nlf(i_f), i_f), nlf(i_f), ns, ws, fs, &
                            sum_s, mag)
            fmag(i_f) = mag + cfact
            flux(i_f) = sum_s
            
            ! from VEGA to AB: https://www.astronomy.ohio-state.edu/martini.10/usefuldata.html
            ! if (i_f == 1) then
            !     fmag(i_f) = fmag(i_f) + 0.91 
            ! else if (i_f == 2) then
            !     fmag(i_f) = fmag(i_f) - 0.08
            ! else if (i_f == 3) then
            !     fmag(i_f) = fmag(i_f) + 0.16
            ! else if (i_f == 4) then
            !     fmag(i_f) = fmag(i_f) + 0.37
            ! endif
        endif
    enddo
    END SUBROUTINE




    SUBROUTINE magANDfluxes(ncores, wavelenghts, nw, nf, nlf, nmaxf, wf, rf, dlum, zeta, nx, ny, flux_cell, area_arc, &
                            SBf, magf, fluxf)
    use omp_lib
    implicit none

    ! input
    integer :: ncores, nw, nf, nx, ny, nmaxf
    integer, dimension(nf) :: nlf !number of lines of each filter
    real, dimension(nmaxf,nf) :: wf, rf !wavelenghts and response functions of each filter
    real, dimension(nw) :: wavelenghts !wavelenghts of the SSP
    real :: dlum, area_arc !luminosity distance and area of the pixel in arcsec^2
    real :: zeta !redshift
    real, dimension(nx,ny,nw) :: flux_cell !fluxes of each pixel in each wavelenght

    ! local
    real, dimension(nf) :: fmag, flux
    integer :: tam_ii, tam_jj

    ! output
    real, dimension(nx,ny,nf) :: SBf, magf, fluxf !surface brightness, magnitudes and fluxes of each filter in each pixel

    ! FLAGS FOR THE PYTHON-FORTRAN WRAPPER
    !f2py intent(in) ncores, wavelenghts, nw, nf, nlf, nmaxf, wf, rf
    !f2py intent(in) dlum, nx, ny, flux_cell, area_arc, zeta
    !f2py intent(out) SBf, magf, fluxf

    !f2py depend(nf) :: nlf
    !f2py depend(nmaxf,nf) :: wf, rf
    !f2py depend(nw) :: wavelenghts
    !f2py depend(nv) :: wv, fv
    !f2py depend(nx, ny, nw) :: flux_cell
    !f2py depend(nx, ny, nf) :: SBf, magf, fluxf
    
    call OMP_SET_NUM_THREADS(ncores)

    SBf(:,:,:) = 0.
    magf(:,:,:) = 0.
    fluxf(:,:,:) = 0.
        
    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(tam_ii, tam_jj, fmag, flux)
    !$OMP DO REDUCTION(+:SBf, magf, fluxf)
    do tam_jj = 1,ny
    do tam_ii = 1,nx
        fmag(:) = 0.
        flux(:) = 0.
        call mag_v1_0(wavelenghts, flux_cell(tam_ii, tam_jj, :), nw, fmag, flux, nf, nlf, nmaxf, wf, rf, dlum, zeta)
        SBf(tam_ii, tam_jj, :) = fmag(:) + 2.5*log10(area_arc)
        magf(tam_ii, tam_jj, :) = fmag(:)
        fluxf(tam_ii, tam_jj, :) = flux(:)
    enddo
    enddo
    !$OMP END DO
    !$OMP END PARALLEL
    END SUBROUTINE

END MODULE
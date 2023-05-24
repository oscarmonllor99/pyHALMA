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

    !f2py intent(in) ncores, npart, mass, age, met, wavelenghts, SSP, age_span, Z_span, nw, nZ,  nages, istart, iend, disp, lumg, tam_i, tam_j, vel, nx, ny, clight
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

    !$OMP PARALLEL SHARED(mass, age, met, age_span, nages, Z_span, nZ, tam_i, tam_j, clight, &
    !$OMP                 wavelenghts, SSP, lumg, disp, istart, iend, npart, vel, flux_cell, &
    !$OMP                 flux_cell_sig, fluxtot, vell_malla, sigl_malla, lum_malla, twl, Zwl, &
    !$OMP                 vell2_malla) &
    !$OMP          PRIVATE(iage, imet, tam_ii, tam_jj, ip, iw, lpix, shift, waves, fluxs, lump, &
    !$OMP                  dage, dmet, ssp2, wave2)
    !$OMP DO REDUCTION(+: fluxtot, flux_cell, flux_cell_sig, vell_malla, sigl_malla, lum_malla, &
    !$OMP                 twl, Zwl, vell2_malla)
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
    !$OMP END DO
    !$OMP END PARALLEL

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

    !$OMP PARALLEL SHARED(vel, lumg, mass, tam_i, tam_j, sigl_malla, vell_malla) &
    !$OMP          PRIVATE(iage, imet, ip, lump, tam_ii, tam_jj)
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
        sum = sum + (x(ix+1) - x(ix))*(f(ix+1) + f(ix))
    enddo
    END SUBROUTINE





    SUBROUTINE from_ws_to_wfilt(fs, wfilt, nfilt, ws, ns, fs_new)
    implicit none
    ! input
    integer :: nfilt, ns
    real, dimension(nfilt) :: wfilt !wavelenghts of the filters
    real, dimension(ns) :: ws, fs !wavelenghts and spectra of the SSP

    ! local
    integer :: i_s, i_w

    ! output
    real, dimension(nfilt) :: fs_new !spectra of the SSP in the filter wavelenghts
    
    fs_new(:) = 0.
    do i_s = 1,ns
        if ( wfilt(1) <= ws(i_s) .and. ws(i_s) <= wfilt(nfilt) ) then
            i_w = minloc(array = abs(wfilt - ws(i_s)), dim = 1)
            fs_new(i_w) = fs(i_s)
        endif
    enddo
    END SUBROUTINE




    SUBROUTINE mymagnitude(wfilt, rfilt, nfilt, ns, ws, fs, nv, wv, fv, sum_s, sum_v, mag)
    implicit none
    ! input
    integer :: nfilt, ns, nv
    real, dimension(nfilt) :: wfilt, rfilt !wavelenghts and response functions of each filter
    real, dimension(ns) :: ws, fs !wavelenghts and spectra of the SSP
    real, dimension(nv) :: wv, fv !wavelenghts and spectra of Vega

    ! local
    real, dimension(nfilt) :: fs_new, fv_new !spectra of the SSP and Vega in the filter wavelenghts

    ! output
    real :: sum_s, sum_v, mag !flux of the SSP and Vega in the filter wavelenghts and magnitude in vega system

    call from_ws_to_wfilt(fs, wfilt, nfilt, ws, ns, fs_new)
    call from_ws_to_wfilt(fv, wfilt, nfilt, wv, nv, fv_new)

    call trapecio(f = fs_new*rfilt, x = wfilt, n = nfilt, sum = sum_s)
    call trapecio(f = fv_new*rfilt, x = wfilt, n = nfilt, sum = sum_v)

    mag = -2.5*log10(sum_s/sum_v)
    END SUBROUTINE




    SUBROUTINE mag_v1_0(ws, fs, ns, fmag, flux, nf, nlf, nmaxf, wf, rf, wv, fv, nv, dlum)
    implicit none
    ! input
    integer :: ns, nf, nv, nmaxf
    integer, dimension(nf) :: nlf !number of lines of each filter
    real, dimension(nmaxf,nf) :: wf, rf !wavelenghts and response functions of each filter
    real, dimension(ns) :: ws, fs !wavelenghts and spectra of the SSP
    real, dimension(nv) :: wv, fv !wavelenghts and spectra of Vega
    real :: dlum !luminosity distance
    real, dimension(nf) :: fmag, flux !magnitudes and fluxes of each filter

    ! local
    real :: cfact, ws_min, ws_max, sum_s, sum_v, mag
    integer :: i_f
    
    ! unit conversion from lum[erg/s/A] to flux[erg/s/A/cm²] for sum_s, which is in units of  SOLAR LUMINOSITY
    cfact = 5. * log10(1.7685*1e8 * dlum ) !ESTO CONVIERTE sum_s DENTRO DE mag EN UN FLUJO (dividiendo por 4pidlum^2)
                                            !pasa dlum de MPC a CM y ademas pasa de Lo a 3.826*10^33
    ws_min = ws(1)
    ws_max = ws(ns)

    do i_f=1,nf
        if ( ( wf(1,i_f) > ws_min ) .and. ( wf(nlf(i_f),i_f) < ws_max ) ) then
            sum_s = 0.
            sum_v = 0.
            mag = 0.
            call mymagnitude(wf(:nlf(i_f), i_f), rf(:nlf(i_f), i_f), nlf(i_f), ns, ws, fs, nv, wv, fv, &
                            sum_s, sum_v, mag)
            fmag(i_f) = mag + cfact
            flux(i_f) = sum_s

            ! from VEGA to AB
            if (i_f == 1) then
                fmag(i_f) = fmag(i_f) + 0.91 
            else if (i_f == 2) then
                fmag(i_f) = fmag(i_f) - 0.08
            else if (i_f == 3) then
                fmag(i_f) = fmag(i_f) + 0.16
            else if (i_f == 4) then
                fmag(i_f) = fmag(i_f) + 0.37
            endif
        endif
    enddo
    END SUBROUTINE




    SUBROUTINE magANDfluxes(ncores, wavelenghts, nw, nf, nlf, nmaxf, wf, rf, wv, fv, nv, dlum, nx, ny, flux_cell, area_arc, &
                            SBf, magf, fluxf)
    use omp_lib
    implicit none

    ! input
    integer :: ncores, nw, nf, nx, ny, nv, nmaxf
    integer, dimension(nf) :: nlf !number of lines of each filter
    real, dimension(nmaxf,nf) :: wf, rf !wavelenghts and response functions of each filter
    real, dimension(nw) :: wavelenghts !wavelenghts of the SSP
    real, dimension(nv) :: wv, fv !wavelenghts and spectra of Vega
    real :: dlum, area_arc !luminosity distance and area of the pixel in arcsec^2
    real, dimension(nx,ny,nw) :: flux_cell !fluxes of each pixel in each wavelenght

    ! local
    real, dimension(nf) :: fmag, flux
    integer :: tam_ii, tam_jj

    ! output
    real, dimension(nx,ny,nf) :: SBf, magf, fluxf !surface brightness, magnitudes and fluxes of each filter in each pixel

    ! FLAGS FOR THE PYTHON-FORTRAN WRAPPER
    !f2py intent(in) ncores, wavelenghts, nw, nf, nlf, nmaxf, wf, rf, wv, fv, nv, dlum, nx, ny, flux_cell, area_arc
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

    !$OMP PARALLEL SHARED(nx, ny, wavelenghts, flux_cell, nw, nf, nlf, wf, rf, wv, nv, dlum, area_arc, &
    !$OMP                  SBf, magf, fluxf) &
    !$OMP          PRIVATE(tam_ii, tam_jj, fmag, flux)
    !$OMP DO REDUCTION(+:SBf, magf, fluxf)
    do tam_jj = 1,ny
    do tam_ii = 1,nx
        fmag(:) = 0.
        flux(:) = 0.
        call mag_v1_0(wavelenghts, flux_cell(tam_ii, tam_jj, :), nw, fmag, flux, nf, nlf, nmaxf, wf, rf, wv, fv, nv, dlum)
        SBf(tam_ii, tam_jj, :) = fmag(:) + 2.5*log10(area_arc)
        magf(tam_ii, tam_jj, :) = fmag(:)
        fluxf(tam_ii, tam_jj, :) = flux(:)
    enddo
    enddo
    !$OMP END DO
    !$OMP END PARALLEL
    END SUBROUTINE

END MODULE
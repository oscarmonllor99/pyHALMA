! Created on Mon May 24 2023
! In this module the heaviest parts of particle calculations are
! implemented. The module is called from the main program, which is
! written in Python. The module is compiled with gfortran through f2py.

MODULE particle
    implicit none

contains 

!**********************************************************************
       SUBROUTINE DIAGONALISE(INPUT_MATRIX, EIGENVALUES_OUT)
!**********************************************************************
!*      Diagonalise using the Jacobi rotations method
!**********************************************************************

       IMPLICIT NONE
       INTEGER N
       REAL*4  INPUT_MATRIX(1:3,1:3),EIGENVALUES_OUT(1:3)
       REAL*8 MATRIX(1:3,1:3),EIGEN_LOC(1:3)
       INTEGER III,JJJ,I,J
       REAL*8 AUX_ARR_1(100),AUX_ARR_2(100)
       REAL*8 COSINUS,VAR1,VAR2,SINUS,SUMA
       REAL*8 TANGENT,TANHALF,ROT_ANG,UPPER_LIM,SUM_ELEMENTS

       N=3

       ! WE'LL WORK WITH A, INPUT_MATRIX IS UNMODIFIED
       DO III=1,N
       DO JJJ=1,N
        MATRIX(III,JJJ)=INPUT_MATRIX(III,JJJ)
       END DO
       END DO

       DO III=1,N
         AUX_ARR_1(III)=MATRIX(III,III) ! diagonal
         EIGEN_LOC(III)=AUX_ARR_1(III) ! this will contain EIGENVALUES_OUT
         AUX_ARR_2(III)=0.d0
       END DO

!      SUM OF ELEMENTS, TO CONTROL ERRORS
       SUM_ELEMENTS=0.0
       DO III=1,N
       DO JJJ=III,N
        SUM_ELEMENTS=SUM_ELEMENTS+ABS(MATRIX(III,JJJ))
       END DO
       END DO

!      JACOBI LOOP, 100 ITERATIONS MAX
       DO I=1,100
        SUMA=0.D0
        DO III=1,N-1
        DO JJJ=III+1,N
         SUMA=SUMA+ABS(MATRIX(III,JJJ))
        END DO
        END DO
        IF (SUMA.LT.1.e-4*SUM_ELEMENTS) EXIT ! WE HAVE CONVERGED

        IF(I.LT.4) THEN
           UPPER_LIM=0.2D0*SUMA**2
        ELSE
           UPPER_LIM=0.D0
        END IF
        DO III=1,N-1
         DO JJJ=III+1, N
          VAR1=100.D0*ABS(MATRIX(III,JJJ))

          IF ((I.GT.4).AND. &        
          (ABS(EIGEN_LOC(III))+VAR1.EQ.ABS(EIGEN_LOC(III))).AND. &     
          (ABS(EIGEN_LOC(JJJ))+VAR1.EQ.ABS(EIGEN_LOC(JJJ)))) THEN

           MATRIX(III,JJJ)=0.D0
          ELSE IF (ABS(MATRIX(III,JJJ)).GT.UPPER_LIM) THEN
           VAR2=EIGEN_LOC(JJJ)-EIGEN_LOC(III)
           IF (ABS(VAR2)+VAR1.EQ.ABS(VAR2)) THEN
            TANGENT=MATRIX(III,JJJ)/VAR2
           ELSE
            ROT_ANG=0.5D0*VAR2/MATRIX(III,JJJ)
            TANGENT=1.D0/(ABS(ROT_ANG)+SQRT(1.D0+ROT_ANG**2))
            IF (ROT_ANG.lt.0.D0) TANGENT=-TANGENT
           END IF
           COSINUS=1.D0/SQRT(1.D0+TANGENT**2)
           SINUS=TANGENT*COSINUS
           TANHALF=SINUS/(1.D0+COSINUS)
           VAR2=TANGENT*MATRIX(III,JJJ)
           AUX_ARR_2(III)=AUX_ARR_2(III)-VAR2
           AUX_ARR_2(JJJ)=AUX_ARR_2(JJJ)+VAR2
           EIGEN_LOC(III)=EIGEN_LOC(III)-VAR2
           EIGEN_LOC(JJJ)=EIGEN_LOC(JJJ)+VAR2
           MATRIX(III,JJJ)=0.D0
           DO J=1,III-1
            VAR1=MATRIX(J,III)
            VAR2=MATRIX(J,JJJ)
            MATRIX(J,III)=VAR1-SINUS*(VAR2+VAR1*TANHALF)
            MATRIX(J,JJJ)=VAR2+SINUS*(VAR1-VAR2*TANHALF)
           END DO
           DO J=III+1,JJJ-1
            VAR1=MATRIX(III,J)
            VAR2=MATRIX(J,JJJ)
            MATRIX(III,J)=VAR1-SINUS*(VAR2+VAR1*TANHALF)
            MATRIX(J,JJJ)=VAR2+SINUS*(VAR1-VAR2*TANHALF)
           END DO
           DO J=JJJ+1,N
            VAR1=MATRIX(III,J)
            VAR2=MATRIX(JJJ,J)
            MATRIX(III,J)=VAR1-SINUS*(VAR2+VAR1*TANHALF)
            MATRIX(JJJ,J)=VAR2+SINUS*(VAR1-VAR2*TANHALF)
          END DO

          END IF  !(I.GT.4)
         END DO !JJJ=III+1,N
        END DO !III=1,N-1
        DO III=1,N
         AUX_ARR_1(III)=AUX_ARR_1(III)+AUX_ARR_2(III)
         EIGEN_LOC(III)=AUX_ARR_1(III)
         AUX_ARR_2(III)=0.D0
        END DO
       END DO !I=1,100

!       MOVE THE EIGENVALUES_OUT TO THE OUTPUT VARIABLE
       DO III=1,N
        EIGENVALUES_OUT(III)=EIGEN_LOC(III)
       END DO

        RETURN
       END SUBROUTINE

!**********************************************************************
    SUBROUTINE SORT_EIGEN(EIGENVALUES,N)
!**********************************************************************
!*      Sorts the eigenvalues in increasing order
!**********************************************************************
    IMPLICIT NONE

    INTEGER N
    REAL*4  EIGENVALUES(N)

    INTEGER II,JJ,KK
    REAL*4 VALOR

    DO II=1,N-1
        KK=II
        VALOR=EIGENVALUES(II)
        DO JJ=II+1,N
        IF(EIGENVALUES(JJ).GE.VALOR) THEN
        KK=JJ
        VALOR=EIGENVALUES(JJ)
        END IF
        END DO
        IF(KK.NE.II) THEN
        EIGENVALUES(KK)=EIGENVALUES(II)
        EIGENVALUES(II)=VALOR
        END IF
    END DO

    RETURN
    END SUBROUTINE 

!**********************************************************************************
    SUBROUTINE halo_shape(ncore, npart, x, y, z, mass, eigenvalues)
!**********************************************************************************
    use omp_lib
    implicit none
    ! Input
    INTEGER, INTENT(IN) :: ncore, npart
    REAL, DIMENSION(npart), INTENT(IN) :: x, y, z, mass

    ! Output
    REAL, DIMENSION(3) :: eigenvalues

    !f2py intent(in) :: ncore, npart, x, y, z, mass
    !f2py intent(out) :: eigenvalues
    !f2py depend(npart) :: x, y, z, mass

    ! Local
    real :: mass_tot
    real, dimension(3) :: rvec
    integer :: ip, j, i, ii
    real, dimension(3,3) :: inertia_tensor

    call OMP_SET_NUM_THREADS(ncore)

    !Inertia tensor calculation
    inertia_tensor(:,:) = 0.
    eigenvalues(:) = 0.
    mass_tot = 0.

    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ip)
    !$OMP DO REDUCTION(+:inertia_tensor)
    do ip=1,npart
        rvec(1) = x(ip)
        rvec(2) = y(ip)
        rvec(3) = z(ip)
        do j=1,3
        do i=1,3
            inertia_tensor(i,j) = inertia_tensor(i,j) + mass(ip)*rvec(i)*rvec(j)
        end do
        end do
        mass_tot = mass_tot + mass(ip)
    end do
    !$OMP END DO
    !$OMP END PARALLEL

    ! Inertia tensor normalization
    inertia_tensor = inertia_tensor/mass_tot

    ! Inertia tensor diagonalization
    call diagonalise(inertia_tensor, eigenvalues)
    call sort_eigen(eigenvalues, 3)

    do ii=1,3
        eigenvalues(ii) = sqrt(eigenvalues(ii))
    end do
    END SUBROUTINE 

!***********************************************************************************************
    SUBROUTINE sigma_projections(ncore, npart, grid, n_cell, part_list, st_x, st_y, st_z, &
                                 st_vx, st_vy, st_vz, st_mass, cx, cy, cz, R05x, R05y, R05z, ll, &
                                 SIG_1D_x_05, SIG_1D_y_05, SIG_1D_z_05, V_sigma, lambda)
!************************************************************************************************
    use omp_lib
    implicit none

    ! Input
    integer, intent(in) :: ncore, npart, n_cell
    real, intent(in) :: cx, cy, cz, R05x, R05y, R05z, ll
    real, dimension(n_cell), intent(in) :: grid
    integer, dimension(npart), intent(in) :: part_list
    real, dimension(:), intent(in) :: st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass

    ! Output
    real, intent(out) :: SIG_1D_x_05, SIG_1D_y_05, SIG_1D_z_05, V_sigma, lambda

    !f2py intent(in) :: ncore, npart, grid, n_cell, part_list, st_x, st_y, st_z
    !f2py intent(in) :: st_vx, st_vy, st_vz, st_mass, cx, cy, cz, R05x, R05y, R05z, ll
    !f2py intent(out) :: SIG_1D_x_05, SIG_1D_y_05, SIG_1D_z_05, V_sigma, lambda

    !f2py depend(npart) :: part_list
    !f2py depend(n_cell) :: grid

    ! Local
    integer, dimension(n_cell, n_cell) :: quantas_x, quantas_y, quantas_z
    real, dimension(n_cell, n_cell) :: VCM_x, VCM_y, VCM_z, SD_x, SD_y, SD_z, SIG_1D_x, SIG_1D_y, SIG_1D_z
    integer :: counter_x, counter_y, counter_z
    integer :: ip, ipp, ix, iy, iz
    real :: dist_x, dist_y, dist_z, dx, dy, dz
    real :: sumV, sumSigma, sumUp, sumDown, Rbin
    real :: V_sigma_z, lambda_z, V_sigma_y, lambda_y, V_sigma_x, lambda_x

    call OMP_SET_NUM_THREADS(ncore)


    VCM_x(:,:) = 0.
    VCM_y(:,:) = 0.
    VCM_z(:,:) = 0.
    SD_x(:,:) = 0.
    SD_y(:,:) = 0.
    SD_z(:,:) = 0.
    quantas_x(:,:) = 0
    quantas_y(:,:) = 0
    quantas_z(:,:) = 0

    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ip, ipp, ix, iy, iz)
    !$OMP DO REDUCTION(+:VCM_x, VCM_y, VCM_z, SD_x, SD_y, SD_z, quantas_x, quantas_y, quantas_z)
    do ip=1,npart
        ipp = part_list(ip)
        ix = minloc(abs(grid - (st_x(ipp) - cx)), dim = 1)
        iy = minloc(abs(grid - (st_y(ipp) - cy)), dim = 1)
        iz = minloc(abs(grid - (st_z(ipp) - cz)), dim = 1)
        VCM_x(iy,iz) = VCM_x(iy,iz) + st_vx(ipp)*st_mass(ipp)
        VCM_y(ix,iz) = VCM_y(ix,iz) + st_vy(ipp)*st_mass(ipp)
        VCM_z(ix,iy) = VCM_z(ix,iy) + st_vz(ipp)*st_mass(ipp)
        SD_x(iy,iz) = SD_x(iy,iz) + st_mass(ipp)
        SD_y(ix,iz) = SD_y(ix,iz) + st_mass(ipp)
        SD_z(ix,iy) = SD_z(ix,iy) + st_mass(ipp)
        quantas_x(iy,iz) = quantas_x(iy,iz) + 1
        quantas_y(ix,iz) = quantas_y(ix,iz) + 1
        quantas_z(ix,iy) = quantas_z(ix,iy) + 1
    enddo
    !$OMP END DO
    !$OMP END PARALLEL

    !to avoid division by zero
    where (SD_x(:,:) /= 0.) VCM_x(:,:) = VCM_x(:,:)/SD_x(:,:)
    where (SD_y(:,:) /= 0.) VCM_y(:,:) = VCM_y(:,:)/SD_y(:,:)
    where (SD_z(:,:) /= 0.) VCM_z(:,:) = VCM_z(:,:)/SD_z(:,:)



    SIG_1d_x = 0.
    SIG_1d_y = 0.
    SIG_1d_z = 0.

    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ip, ipp, ix, iy, iz)
    !$OMP DO REDUCTION(+:SIG_1d_x, SIG_1d_y, SIG_1d_z)
    do ip=1,npart
        ipp = part_list(ip) 
        ix = minloc(abs(grid - (st_x(ipp) - cx)), dim = 1)
        iy = minloc(abs(grid - (st_y(ipp) - cy)), dim = 1)
        iz = minloc(abs(grid - (st_z(ipp) - cz)), dim = 1)
        SIG_1d_x(iy, iz) = SIG_1d_x(iy, iz) + (st_vx(ipp) - VCM_x(iy, iz))**2
        SIG_1d_y(ix, iz) = SIG_1d_y(ix, iz) + (st_vy(ipp) - VCM_y(ix, iz))**2
        SIG_1d_z(ix, iy) = SIG_1d_z(ix, iy) + (st_vz(ipp) - VCM_z(ix, iy))**2
    enddo
    !$OMP END DO
    !$OMP END PARALLEL

    where (quantas_x(:,:) /= 0) SIG_1d_x(:,:) = sqrt(SIG_1d_x(:,:)/quantas_x(:,:))
    where (quantas_y(:,:) /= 0) SIG_1d_y(:,:) = sqrt(SIG_1d_y(:,:)/quantas_y(:,:))
    where (quantas_z(:,:) /= 0) SIG_1d_z(:,:) = sqrt(SIG_1d_z(:,:)/quantas_z(:,:))




    SIG_1d_x_05 = 0.
    SIG_1d_y_05 = 0.
    SIG_1d_z_05 = 0.
    counter_x = 0
    counter_y = 0
    counter_z = 0

    !$OMP PARALLEL DEFAULT(SHARED) PRIVATE(ip, ipp, ix, iy, iz, dx, dy, dz, dist_x, dist_y, dist_z)
    !$OMP DO REDUCTION(+:SIG_1D_x_05, SIG_1D_y_05, SIG_1D_z_05, counter_x, counter_y, counter_z)
    do ip=1,npart
        ipp = part_list(ip)
        dx = st_x(ipp) - cx
        dy = st_y(ipp) - cy
        dz = st_z(ipp) - cz
        ix = minloc(abs(grid - dx), dim = 1)
        iy = minloc(abs(grid - dy), dim = 1)
        iz = minloc(abs(grid - dz), dim = 1)
        dist_x = sqrt(dy**2 + dz**2)
        dist_y = sqrt(dx**2 + dz**2)
        dist_z = sqrt(dx**2 + dy**2)

        if (dist_x < R05x) then
            SIG_1d_x_05 = SIG_1d_x_05 + SIG_1d_x(iy, iz)
            counter_x = counter_x + 1
        endif

        if (dist_y < R05y) then
            SIG_1d_y_05 = SIG_1d_y_05 + SIG_1d_y(ix, iz)
            counter_y = counter_y + 1
        endif

        if (dist_z < R05z) then
            SIG_1d_z_05 = SIG_1d_z_05 + SIG_1d_z(ix, iy)
            counter_z = counter_z + 1
        endif
    enddo
    !$OMP END DO
    !$OMP END PARALLEL

    if (counter_x > 0) then
        SIG_1d_x_05 = SIG_1d_x_05/counter_x
    endif

    if (counter_y > 0) then
        SIG_1d_y_05 = SIG_1d_y_05/counter_y
    endif

    if (counter_z > 0) then
        SIG_1d_z_05 = SIG_1d_z_05/counter_z
    endif

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !V/sigma and lambda part (Fast-Slow rotator)
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    V_sigma_x = 0.
    lambda_x = 0.
    V_sigma_y = 0.
    lambda_y = 0.
    V_sigma_z = 0.
    lambda_z = 0.

    !!! XY plane
    sumV = 0.
    sumSigma = 0.
    sumUp = 0.
    sumDown = 0.
    do iy=1,n_cell
    do ix=1,n_cell
        Rbin = sqrt(grid(ix)**2 + grid(iy)**2)
        if (Rbin < R05z + 2*ll) then
            !vsigma
            sumV = sumV + VCM_z(ix, iy)**2 * SD_z(ix, iy)
            sumSigma = sumSigma + SIG_1D_z(ix, iy)**2 * SD_z(ix, iy)
            !lambda
            sumUp = sumUp + SD_z(ix, iy) * Rbin * abs(VCM_z(ix, iy))
            sumDown = sumDown + SD_z(ix, iy) * Rbin * sqrt(VCM_z(ix, iy)**2 + SIG_1D_z(ix, iy)**2)
        endif
    enddo

    enddo
    if (sumSigma > 0.) then
        V_sigma_z = sqrt(sumV/sumSigma)
    endif
    if (sumDown > 0.) then
        lambda_z = sumUp/sumDown
    endif

    !!! XZ plane
    sumV = 0.
    sumSigma = 0.
    sumUp = 0.
    sumDown = 0.
    do iz=1,n_cell
    do ix=1,n_cell
        Rbin = sqrt(grid(ix)**2 + grid(iz)**2)
        if (Rbin < R05y + 2*ll) then
            !vsigma
            sumV = sumV + VCM_y(ix, iz)**2 * SD_y(ix, iz)
            sumSigma = sumSigma + SIG_1D_y(ix, iz)**2 * SD_y(ix, iz)
            !lambda
            sumUp = sumUp + SD_y(ix, iz) * Rbin * abs(VCM_y(ix, iz))
            sumDown = sumDown + SD_y(ix, iz) * Rbin * sqrt(VCM_y(ix, iz)**2 + SIG_1D_y(ix, iz)**2)
        endif
    enddo
    enddo

    if (sumSigma > 0.) then
        V_sigma_y = sqrt(sumV/sumSigma)
    endif
    if (sumDown > 0.) then
        lambda_y = sumUp/sumDown
    endif

    !!! YZ plane
    sumV = 0.
    sumSigma = 0.
    sumUp = 0.
    sumDown = 0.
    do iz=1,n_cell
    do iy=1,n_cell
        Rbin = sqrt(grid(iy)**2 + grid(iz)**2)
        if (Rbin < R05x + 2*ll) then
            !vsigma
            sumV = sumV + VCM_x(iy, iz)**2 * SD_x(iy, iz)
            sumSigma = sumSigma + SIG_1D_x(iy, iz)**2 * SD_x(iy, iz)
            !lambda
            sumUp = sumUp + SD_x(iy, iz) * Rbin * abs(VCM_x(iy, iz))
            sumDown = sumDown + SD_x(iy, iz) * Rbin * sqrt(VCM_x(iy, iz)**2 + SIG_1D_x(iy, iz)**2)
        endif
    enddo
    enddo

    if (sumSigma > 0.) then
        V_sigma_x = sqrt(sumV/sumSigma)
    endif
    if (sumDown > 0.) then
        lambda_x = sumUp/sumDown
    endif

    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !AVERAGE
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    V_sigma = (V_sigma_x + V_sigma_y + V_sigma_z)/3.
    lambda = (lambda_x + lambda_y + lambda_z)/3.

    END SUBROUTINE




    SUBROUTINE brute_force_binding_energy(ncores, ntotal, &
                    total_mass, total_x, total_y, total_z, &
                    ntest, test_x, test_y, test_z, binding_energy)
    use omp_lib
    implicit none

    !input
    integer, intent(in) :: ntotal, ntest, ncores
    real, dimension(ntotal), intent(in) :: total_mass, total_x, total_y, total_z
    real, dimension(ntest), intent(in) :: test_x, test_y, test_z

    !local
    integer :: ip, ip2
    real :: r

    !output
    real, dimension(ntest), intent(out) :: binding_energy

    !!!!!!!!! FORTRAN / PYTHON WRAPPER
    !f2py intent(in) :: ncores, ntotal, total_mass, total_x, total_y, total_z, ntest, test_x, test_y, test_z
    !f2py intent(out) :: binding_energy
    !f2py depend(ntotal) :: total_mass, total_x, total_y, total_z
    !f2py depend(ntest) :: test_x, test_y, test_z

    call OMP_SET_NUM_THREADS(ncores)

    binding_energy(:) = 0.

    !$OMP PARALLEL DEFAULT(SHARED) &
    !$OMP          PRIVATE(ip, ip2, r)
    !$OMP DO REDUCTION(+:binding_energy)
    do ip=1,ntest
    do ip2=1,ntotal
        if ( total_x(ip2) /= test_x(ip)) then
        if ( total_y(ip2) /= test_y(ip)) then
        if ( total_z(ip2) /= test_z(ip)) then
            r = sqrt( (total_x(ip2)-test_x(ip))**2 &
                    + (total_y(ip2)-test_y(ip))**2 &
                    + (total_z(ip2)-test_z(ip))**2 )
            binding_energy(ip) = binding_energy(ip) + total_mass(ip2) / r
        endif
        endif
        endif
    enddo
    enddo
    !$OMP END DO
    !$OMP END PARALLEL

    END SUBROUTINE




    ! SUBROUTINE gpu_brute_force_binding_energy(ntotal, &
    !                 total_mass, total_x, total_y, total_z, &
    !                 ntest, test_x, test_y, test_z, binding_energy)
    ! use omp_lib
    ! implicit none

    ! !input
    ! integer, intent(in) :: ntotal, ntest
    ! real, dimension(ntotal), intent(in) :: total_mass, total_x, total_y, total_z
    ! real, dimension(ntest), intent(in) :: test_x, test_y, test_z

    ! !local
    ! integer :: ip, ip2
    ! real :: r

    ! !output
    ! real, dimension(ntest), intent(out) :: binding_energy

    ! !!!!!!!!! FORTRAN / PYTHON WRAPPER
    ! !f2py intent(in) :: ntotal, total_mass, total_x, total_y, total_z, ntest, test_x, test_y, test_z
    ! !f2py intent(out) :: binding_energy
    ! !f2py depend(ntotal) :: total_mass, total_x, total_y, total_z
    ! !f2py depend(ntest) :: test_x, test_y, test_z

    ! binding_energy(:) = 0.

    ! !$omp target data map(to: total_mass(1:ntotal), total_x(1:ntotal), total_y(1:ntotal), total_z(1:ntotal), &
    ! !$omp                     test_x(1:ntest), test_y(1:ntest), test_z(1:ntest)) &
    ! !$omp             map(from: binding_energy(1:ntest))
    ! !$omp target teams distribute parallel do reduction(+:binding_energy)       
    ! do ip=1,ntest
    ! do ip2=1,ntotal
    !     if ( total_x(ip2) /= test_x(ip) ) then
    !     if ( total_y(ip2) /= test_y(ip) ) then
    !     if ( total_z(ip2) /= test_z(ip) ) then
    !         r = sqrt( (total_x(ip2)-test_x(ip))**2 &
    !                 + (total_y(ip2)-test_y(ip))**2 &
    !                 + (total_z(ip2)-test_z(ip))**2 )
    !         binding_energy(ip) = binding_energy(ip) + total_mass(ip2) / r
    !     endif
    !     endif
    !     endif
    ! enddo
    ! enddo
    ! !$omp end target teams distribute parallel do
    ! !$omp end target data

    ! END SUBROUTINE


END MODULE
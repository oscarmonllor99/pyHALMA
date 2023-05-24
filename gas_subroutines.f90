! Created on Mon May 24 2023
! In this module the heaviest part of the gas unbinding is done

MODULE gas
    implicit none

contains 

    SUBROUTINE brute_force_binding_energy(ncores, ntotal, &
                                          total_mass, total_x, total_y, total_z, &
                                          ngas, gas_x, gas_y, gas_z, binding_energy)
        use omp_lib
        implicit none

        !input
        integer, intent(in) :: ntotal, ngas, ncores
        real, dimension(ntotal), intent(in) :: total_mass, total_x, total_y, total_z
        real, dimension(ngas), intent(in) :: gas_x, gas_y, gas_z

        !local
        integer :: ip, ip2
        real :: r

        !output
        real, dimension(ngas), intent(out) :: binding_energy

        !!!!!!!!! FORTRAN / PYTHON WRAPPER
        !f2py intent(in) :: ncores, ntotal, total_mass, total_x, total_y, total_z, ngas, gas_x, gas_y, gas_z
        !f2py intent(out) :: binding_energy
        !f2py depend(ntotal) :: total_mass, total_x, total_y, total_z
        !f2py depend(ngas) :: gas_x, gas_y, gas_z

        call OMP_SET_NUM_THREADS(ncores)

        binding_energy(:) = 0.

        !$OMP PARALLEL SHARED(total_mass, total_x, total_y, total_z, gas_x, gas_y, gas_z, binding_energy) &
        !$OMP          PRIVATE(ip, ip2, r)
        !$OMP DO REDUCTION(+:binding_energy)
        do ip=1,ngas
            do ip2=1,ntotal
                if ( total_x(ip2) /= gas_x(ip)) then
                if ( total_y(ip2) /= gas_y(ip)) then
                if ( total_z(ip2) /= gas_z(ip)) then
                    r = sqrt( (total_x(ip2)-gas_x(ip))**2 &
                            + (total_y(ip2)-gas_y(ip))**2 &
                            + (total_z(ip2)-gas_z(ip))**2 )
                    binding_energy(ip) = binding_energy(ip) + total_mass(ip2) / r
                endif
                endif
                endif
            enddo
        enddo
        !$OMP END DO
        !$OMP END PARALLEL

    END SUBROUTINE


END MODULE
import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange, set_num_threads, get_num_threads
import numba
import sys
import time
from masclet_framework import units, tools
import particle
        

@njit
def patch_to_particles(patch_res, patch_rx, patch_ry, patch_rz, patch_nx, patch_ny, 
                       patch_nz, patch_delta, patch_cr0amr, patch_solapst, patch_vx,
                       patch_vy, patch_vz, patch_temp, box, rho_B):

    #max number of gas particles this patch can provide
    max_part_num = patch_nx * patch_ny * patch_nz
    #initialize particle data
    part_x = np.zeros(max_part_num)
    part_y = np.zeros(max_part_num)
    part_z = np.zeros(max_part_num)
    part_vx = np.zeros(max_part_num)
    part_vy = np.zeros(max_part_num)
    part_vz = np.zeros(max_part_num)
    part_mass = np.zeros(max_part_num)
    part_temp = np.zeros(max_part_num)

    #loop over cells, checking if they are inside the box
    x0 = patch_rx - patch_res/2 #Center of the left-bottom-front cell
    y0 = patch_ry - patch_res/2
    z0 = patch_rz - patch_res/2
    partNum = 0
    for ix in range(patch_nx):
        x = x0 + ix*patch_res #cell center
        if x > box[0] and x < box[1]:
            for iy in range(patch_ny):
                y = y0 + iy*patch_res
                if y > box[2] and y < box[3]:
                    for iz in range(patch_nz):
                        z = z0 + iz*patch_res
                        if z > box[4] and z < box[5]:
                            #check if cell is not refined or solaped
                            if patch_cr0amr[ix,iy,iz] and patch_solapst[ix,iy,iz]:
                                part_x[partNum] = x
                                part_y[partNum] = y
                                part_z[partNum] = z
                                part_vx[partNum] = patch_vx[ix,iy,iz]
                                part_vy[partNum] = patch_vy[ix,iy,iz]
                                part_vz[partNum] = patch_vz[ix,iy,iz]
                                part_mass[partNum] = (1 + patch_delta[ix,iy,iz]) * rho_B * patch_res**3 # CARE, THIS IS COMOVING, NOT PHYSICAL
                                part_temp[partNum] = patch_temp[ix,iy,iz]
                                partNum += 1

    return part_x[:partNum], part_y[:partNum], part_z[:partNum], part_vx[:partNum], part_vy[:partNum], part_vz[:partNum], part_mass[:partNum], part_temp[:partNum]



def AMRgrid_to_particles(L, ncoarse, grid_data, gas_data, Rrps, cx, cy, cz, rho_B):
    ##################################################################################
    # This routine aims to pass from the AMR grid to a particle representation of the gas
    # We will put a gas particle in the center of each cell, with a mass equal to the cell mass
    # STEPS:
    #   1) Find which patches contribute to the halo
    #   2) Loop over the patches and for each cell inside which is not solaped or refined, put a particle
    #      with the corresponding mass, position, temperature and velocity
    ##################################################################################
    # LOAD DATA
    # GRID
    nl = grid_data[2]
    npatch = grid_data[5] #number of patches in each level, starting in l=0
    patchnx = grid_data[6] #patchnx (...): x-extension of each patch (in level l cells) (and Y and Z)
    patchny = grid_data[7]
    patchnz = grid_data[8]
    patchrx = grid_data[12] #patchrx (...): physical position of the center of each patch first ¡l-1! cell (and Y and Z)
    patchry = grid_data[13] # in Mpc
    patchrz = grid_data[14]

    # GAS
    gas_delta = gas_data[0] #delta = rho/rho_B - 1
    gas_vx = gas_data[1]
    gas_vy = gas_data[2]
    gas_vz = gas_data[3]
    gas_temp = gas_data[4]
    gas_cr0amr = gas_data[5]
    gas_solapst = gas_data[6]

    # DEFINE BOX IN WHICH WE WILL LOOK FOR PATCHES
    Rbox = Rrps
    box = np.array([cx-Rbox, cx+Rbox, cy-Rbox, cy+Rbox, cz-Rbox, cz+Rbox])

    # FIND WHICH PATCHES CONTRIBUTE TO THE HALO
    which_patches = tools.which_patches_inside_box(box, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, L, ncoarse)
    patch_level = tools.create_vector_levels(npatch)

    # LOOP OVER THE PATCHES
    gas_particles_x = np.array([])
    gas_particles_y = np.array([])
    gas_particles_z = np.array([])
    gas_particles_mass = np.array([])
    gas_particles_temp = np.array([])
    gas_particles_vx = np.array([])
    gas_particles_vy = np.array([])
    gas_particles_vz = np.array([])

    for patch in which_patches:  
        #PATCH DATA
        l = patch_level[patch]
        if l >= 1: #avoid the coarsest level
            patch_res = (L/ncoarse)/2**l
            patch_rx = patchrx[patch]
            patch_ry = patchry[patch]
            patch_rz = patchrz[patch]
            patch_nx = patchnx[patch]
            patch_ny = patchny[patch]
            patch_nz = patchnz[patch]
            patch_delta = gas_delta[patch]
            patch_cr0amr = gas_cr0amr[patch]
            patch_solapst = gas_solapst[patch]
            patch_vx = gas_vx[patch]
            patch_vy = gas_vy[patch]
            patch_vz = gas_vz[patch]
            patch_temp = gas_temp[patch]
            #CREATE PARTICLES FROM THE PATCH
            part_x, part_y, part_z, part_vx, part_vy, part_vz, part_mass, part_temp = patch_to_particles(patch_res, patch_rx, patch_ry, patch_rz, patch_nx, patch_ny, 
                                                                                                        patch_nz, patch_delta, patch_cr0amr, patch_solapst, patch_vx,
                                                                                                        patch_vy, patch_vz, patch_temp, box, rho_B)
            
            #APPEND PARTICLES TO THE GAS PARTICLES ARRAY
            gas_particles_x = np.append(gas_particles_x, part_x)
            gas_particles_y = np.append(gas_particles_y, part_y)
            gas_particles_z = np.append(gas_particles_z, part_z)
            gas_particles_vx = np.append(gas_particles_vx, part_vx)
            gas_particles_vy = np.append(gas_particles_vy, part_vy)
            gas_particles_vz = np.append(gas_particles_vz, part_vz)
            gas_particles_mass = np.append(gas_particles_mass, part_mass)
            gas_particles_temp = np.append(gas_particles_temp, part_temp)

    # VX, VY, VZ are in c = 1 units, we need to convert them to km/s
    gas_particles_vx *= 3e5
    gas_particles_vy *= 3e5
    gas_particles_vz *= 3e5

    return gas_particles_x, gas_particles_y, gas_particles_z, gas_particles_vx, gas_particles_vy, gas_particles_vz, gas_particles_mass, gas_particles_temp




@njit(numba.float64[:](numba.float64[:], numba.float64[:], numba.float64[:], numba.float64[:], numba.float64[:], numba.float64[:], numba.float64[:]),
      parallel=True, fastmath=True, cache=True
      ) 
def brute_force_binding_energy(total_mass, total_x, total_y, total_z, gas_x, gas_y, gas_z):
    partNum_gas = len(gas_x)
    binding_energy = np.zeros(partNum_gas)
    for ip in range(partNum_gas):
        r = np.sqrt((total_x - gas_x[ip])**2 + (total_y - gas_y[ip])**2 + (total_z - gas_z[ip])**2)
        reduction = 0.
        for ip2 in prange(len(r)):
            if r[ip2] > 0.:
               reduction += total_mass[ip2]/r[ip2]
        
        binding_energy[ip] = reduction
        
    return binding_energy



def brute_force_binding_energy_fortran(total_mass, total_x, total_y, total_z, test_x, test_y, test_z):

    # match the types with the fortran routine
    ntotal_f90 = np.int32(len(total_mass))    
    ntest_f90 = np.int32(len(test_x))
    if ntest_f90 == 0:
        return np.array([])
    
    total_mass_f90 = np.float32(total_mass)
    total_x_f90 = np.float32(total_x)
    total_y_f90 = np.float32(total_y)
    total_z_f90 = np.float32(total_z)
    test_x_f90 = np.float32(test_x)
    test_y_f90 = np.float32(test_y)
    test_z_f90 = np.float32(test_z)
    
    # call the fortran routine
    ncores_f90 = np.int32(get_num_threads())

    t0 = time.time()
    binding_energy = particle.particle.brute_force_binding_energy(ncores_f90, ntotal_f90, 
                                                                  total_mass_f90, total_x_f90, 
                                                                  total_y_f90, total_z_f90,
                                                                  ntest_f90, test_x_f90, 
                                                                  test_y_f90, test_z_f90)
    t1 = time.time()
    # print('CPU', t1-t0, 's')

    # binding_energy = particle.particle.gpu_brute_force_binding_energy(ntotal_f90,
    #                                                                 total_mass_f90, total_x_f90,
    #                                                                 total_y_f90, total_z_f90,
    #                                                                 ntest_f90, test_x_f90,
    #                                                                 test_y_f90, test_z_f90)
    # t2 = time.time()
    # print('GPU', t2-t1, 's')


    return binding_energy



def RPS(rete, L, ncoarse, grid_data, gas_data, masclet_dm_data, masclet_st_data, cx, cy, cz, vx, vy, vz, Rrps, rho_B):
    ##################################################################################
    # This routine aims to calculate the bound/unbound mass gas fraction of each halo
    # For that, we have to pass from the AMR grid to a particle representation of the gas
    # We will put a gas particle in the center of each cell, with a mass equal to the cell mass
    # Then, with gas, star and dark matter particles we calculate the binding energy of each gas particle
    # If the binding energy is negative, the gas particle is bound to the halo
    # We can then calculate the bound/unbound mass gas fraction of each halo
    ##################################################################################

    #####################  LOAD PARTICLE DATA
    # DM
    dm_x = masclet_dm_data[0]
    dm_y = masclet_dm_data[1]
    dm_z = masclet_dm_data[2]
    dm_mass = masclet_dm_data[3]*units.mass_to_sun #in Msun
    # Search for the DM particles inside R05
    dm_gcd = np.sqrt((dm_x-cx)**2 + (dm_y-cy)**2 + (dm_z-cz)**2) # galaxy-centric distance
    inside_Rrps = dm_gcd < Rrps
    dm_x = dm_x[inside_Rrps]
    dm_y = dm_y[inside_Rrps]
    dm_z = dm_z[inside_Rrps]
    dm_mass = dm_mass[inside_Rrps]

    # STARS
    st_x = masclet_st_data[0]
    st_y = masclet_st_data[1]
    st_z = masclet_st_data[2]
    st_mass = masclet_st_data[6]*units.mass_to_sun #in Msun
    # Search for the star particles inside R05
    st_gcd = np.sqrt((st_x-cx)**2 + (st_y-cy)**2 + (st_z-cz)**2) # galaxy-centric distance
    inside_Rrps = st_gcd < Rrps
    st_x = st_x[inside_Rrps]
    st_y = st_y[inside_Rrps]
    st_z = st_z[inside_Rrps]
    st_mass = st_mass[inside_Rrps]
    #####################

    #####################  GAS AMR TO GAS PARTICLES
    gas_x, gas_y, gas_z, gas_vx, gas_vy, gas_vz, gas_mass, gas_temp = AMRgrid_to_particles(L, ncoarse, grid_data, gas_data, Rrps, cx, cy, cz, rho_B)

    # CHECK THAT THE GAS PARTICLES ARE INSIDE R05
    gas_gcd = np.sqrt((gas_x-cx)**2 + (gas_y-cy)**2 + (gas_z-cz)**2) # galaxy-centric distance
    inside_Rrps = gas_gcd < Rrps
    gas_x = gas_x[inside_Rrps]
    gas_y = gas_y[inside_Rrps]
    gas_z = gas_z[inside_Rrps]
    gas_vx = gas_vx[inside_Rrps]
    gas_vy = gas_vy[inside_Rrps]
    gas_vz = gas_vz[inside_Rrps]
    gas_mass = gas_mass[inside_Rrps]
    gas_temp = gas_temp[inside_Rrps]
    
    # FROM COMOVING VOLUME TO PHYSICAL VOLUME
    gas_mass *= rete**3

    #####################  

    #####################  CALCULATE TOTAL ENERGY OF EACH GAS PARTICLE

    # ARRAYS CONTAINING ALL PARTICLES POSITIONS AND MASSES
    total_x = np.concatenate((gas_x, st_x, dm_x))
    total_y = np.concatenate((gas_y, st_y, dm_y))
    total_z = np.concatenate((gas_z, st_z, dm_z))
    total_mass = np.concatenate((gas_mass, st_mass, dm_mass))

    # CALCULATE BINDING ENERGY OF EACH GAS PARTICLE
    binding_energy = brute_force_binding_energy_fortran(total_mass, total_x, total_y, total_z, 
                                                        gas_x, gas_y, gas_z)
    binding_energy = - binding_energy # binding energy is negative

    # Now the variable binding_energy is in units of Msun/mpc, we need to convert it to km^2/s^2
    G_const = 4.3*1e-3 # in units of (km/s)^2 pc/Msun
    G_const *= 1e-6 # in units of (km/s)^2 Mpc/Msun

    binding_energy *= G_const # km^2 s^-2

    # Consider that the binding energy is twice the calculated value
    binding_energy *= 2. 

    # CALCULTATE TOTAL ENERGY OF EACH GAS PARTICLE
    gas_v2 = 0.5 * ( (gas_vx-vx)**2 + (gas_vy-vy)**2 + (gas_vz-vz)**2) # km^2 s^-2

    # TOTAL ENERGY OF EACH GAS PARTICLE
    total_energy = gas_v2 + binding_energy # km^2 s^-2


    # BOUND AND UNBOUND GAS PARTICLES
    unbound = total_energy > 0.
    bound = total_energy <= 0.
    # COLD AND HOT GAS PARTICLES
    cold = gas_temp < 5*1e4
    hot = gas_temp >= 5*1e4

    ##################### RETURN VARIABLES
    total_gas_mass = np.sum(gas_mass)
    cold_bound_gas_mass = np.sum(gas_mass[cold*bound])
    if total_gas_mass != 0.:
        frac_cold_gas_mass = cold_bound_gas_mass/total_gas_mass
    else:
        frac_cold_gas_mass = 0.
    unbound_cold_gas_mass = np.sum(gas_mass[unbound*cold])
    unbound_hot_gas_mass = np.sum(gas_mass[unbound*hot])

    return total_gas_mass, frac_cold_gas_mass, unbound_cold_gas_mass, unbound_hot_gas_mass


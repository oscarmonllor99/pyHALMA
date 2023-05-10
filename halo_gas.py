import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import sys

# our modules
sys.path.append('/home/monllor/projects/')
from masclet_framework import units, tools
import misctools

@njit
def patch_to_particles(patch_res, patch_rx, patch_ry, patch_rz, patch_nx, patch_ny, 
                       patch_nz, patch_density, patch_cr0amr, patch_solapst, patch_vx,
                       patch_vy, patch_vz, patch_temp, box):

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
                            if patch_cr0amr[ix,iy,iz] == 1 and patch_solapst[ix,iy,iz] == 1:
                                part_x[partNum] = x
                                part_y[partNum] = y
                                part_z[partNum] = z
                                part_vx[partNum] = patch_vx[ix,iy,iz]
                                part_vy[partNum] = patch_vy[ix,iy,iz]
                                part_vz[partNum] = patch_vz[ix,iy,iz]
                                part_mass[partNum] = patch_density[ix,iy,iz] * patch_res**3 # CARE, THIS IS COMOVING, NOT PHYSICAL
                                part_temp[partNum] = patch_temp[ix,iy,iz]
                                partNum += 1

    return part_x[:partNum], part_y[:partNum], part_z[:partNum], part_vx[:partNum], part_vy[:partNum], part_vz[:partNum], part_mass[:partNum], part_temp[:partNum]



def AMRgrid_to_particles(L, ncoarse, grid_data, gas_data, R05, cx, cy, cz):
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
    gas_delta = gas_data[0]
    gas_density = misctools.delta_to_rho(gas_delta)
    gas_cr0amr = gas_data[1]
    gas_solapst = gas_data[2]
    gas_vx = gas_data[3]*3e5 #in km/s
    gas_vy = gas_data[4]*3e5 #in km/s
    gas_vz = gas_data[5]*3e5 #in km/s
    gas_temp = gas_data[7]

    # DEFINE BOX IN WHICH WE WILL LOOK FOR PATCHES
    Rbox = 1.*R05
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
        patch_res = (L/ncoarse)/2**l
        patch_rx = patchrx[patch]
        patch_ry = patchry[patch]
        patch_rz = patchrz[patch]
        patch_nx = patchnx[patch]
        patch_ny = patchny[patch]
        patch_nz = patchnz[patch]
        patch_density = gas_density[patch]
        patch_cr0amr = gas_cr0amr[patch]
        patch_solapst = gas_solapst[patch]
        patch_vx = gas_vx[patch]
        patch_vy = gas_vy[patch]
        patch_vz = gas_vz[patch]
        patch_temp = gas_temp[patch]

        #CREATE PARTICLES FROM THE PATCH
        part_x, part_y, part_z, part_vx, part_vy, part_vz, part_mass, part_temp = patch_to_particles(patch_res, patch_rx, patch_ry, patch_rz, patch_nx, patch_ny, 
                                                                                                    patch_nz, patch_density, patch_cr0amr, patch_solapst, patch_vx,
                                                                                                    patch_vy, patch_vz, patch_temp, box)
        
        #APPEND PARTICLES TO THE GAS PARTICLES ARRAY
        gas_particles_x = np.append(gas_particles_x, part_x)
        gas_particles_y = np.append(gas_particles_y, part_y)
        gas_particles_z = np.append(gas_particles_z, part_z)
        gas_particles_vx = np.append(gas_particles_vx, part_vx)
        gas_particles_vy = np.append(gas_particles_vy, part_vy)
        gas_particles_vz = np.append(gas_particles_vz, part_vz)
        gas_particles_mass = np.append(gas_particles_mass, part_mass)
        gas_particles_temp = np.append(gas_particles_temp, part_temp)

    return gas_particles_x, gas_particles_y, gas_particles_z, gas_particles_vx, gas_particles_vy, gas_particles_vz, gas_particles_mass, gas_particles_temp


@njit(fastmath=True) #allow not that strict math precision
def brute_force_binding_energy(total_mass, total_x, total_y, total_z, gas_x, gas_y, gas_z):
    partNum_gas = len(gas_x)
    binding_energy = np.zeros(partNum_gas)
    for ip in range(partNum_gas):
        r = np.sqrt((total_x - gas_x[ip])**2 + (total_y - gas_y[ip])**2 + (total_z - gas_z[ip])**2)
        # avoid r = 0
        r[r<1e-8] = np.nan
        binding_energy[ip] = np.nansum(total_mass/r)

    return binding_energy

def RPS(L, ncoarse, grid_data, gas_data, masclet_dm_data, masclet_st_data, R05, cx, cy, cz, vx, vy, vz):
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
    dm_mass = masclet_st_data[3]*units.mass_to_sun #in Msun
    # Search for the DM particles inside R05
    dm_gcd = np.sqrt((dm_x-cx)**2 + (dm_y-cy)**2 + (dm_z-cz)**2) # galaxy-centric distance
    inside_R05 = dm_gcd < R05
    dm_x = dm_x[inside_R05]
    dm_y = dm_y[inside_R05]
    dm_z = dm_z[inside_R05]
    dm_mass = dm_mass[inside_R05]

    # STARS
    st_x = masclet_st_data[0]
    st_y = masclet_st_data[1]
    st_z = masclet_st_data[2]
    st_mass = masclet_st_data[6]*units.mass_to_sun #in Msun
    # Search for the star particles inside R05
    st_gcd = np.sqrt((st_x-cx)**2 + (st_y-cy)**2 + (st_z-cz)**2) # galaxy-centric distance
    inside_R05 = st_gcd < R05
    st_x = st_x[inside_R05]
    st_y = st_y[inside_R05]
    st_z = st_z[inside_R05]
    #####################

    #####################  GAS AMR TO GAS PARTICLES
    gas_x, gas_y, gas_z, gas_vx, gas_vy, gas_vz, gas_mass, gas_temp = AMRgrid_to_particles(L, ncoarse, grid_data, gas_data, R05, cx, cy, cz)
    # CHECK THAT THE GAS PARTICLES ARE INSIDE R05
    gas_gcd = np.sqrt((gas_x-cx)**2 + (gas_y-cy)**2 + (gas_z-cz)**2) # galaxy-centric distance
    inside_R05 = gas_gcd < R05
    gas_x = gas_x[inside_R05]
    gas_y = gas_y[inside_R05]
    gas_z = gas_z[inside_R05]
    gas_vx = gas_vx[inside_R05]
    gas_vy = gas_vy[inside_R05]
    gas_vz = gas_vz[inside_R05]
    gas_mass = gas_mass[inside_R05]
    gas_temp = gas_temp[inside_R05]

    # take into account that gas_mass should be corrected by the scale factor, 
    # since it is density*cell_volume and this volume is in comoving coordinates
    #####################  

    #####################  CALCULATE TOTAL ENERGY OF EACH GAS PARTICLE

    # ARRAYS CONTAINING ALL PARTICLES POSITIONS AND MASSES
    total_x = np.concatenate((gas_x, st_x, dm_x))
    total_y = np.concatenate((gas_y, st_y, dm_y))
    total_z = np.concatenate((gas_z, st_z, dm_z))
    total_mass = np.concatenate((gas_mass, st_mass, dm_mass))

    # CALCULATE BINDING ENERGY OF EACH GAS PARTICLE

    binding_energy = brute_force_binding_energy(total_mass, total_x, total_y, total_z, gas_x, gas_y, gas_z)

    # Now the variable binding_energy is in units of Msun/mpc, we need to convert it to km^2/s^2
    G_const = units.G_isu #  6.674e-11 # m^3 kg^-1 s^-2
    G_const /= units.mass_to_sun #  m^3 Msun^-1 s^-2
    G_const *= units.m_to_mpc #  m^2 mpc Msun^-1 s^-2
    G_const *= 1e-6 # km^2 mpc Msun^-1 s^-2

    binding_energy *= G_const # km^2 s^-2

    # CALCULTATE TOTAL ENERGY OF EACH GAS PARTICLE
    gas_v2 = 0.5 * ( (gas_vx-vx)**2 + (gas_vy-vy)**2 + (gas_vz-vz)**2) # km^2 s^-2

    # TOTAL ENERGY OF EACH GAS PARTICLE
    total_energy = gas_v2 + binding_energy # km^2 s^-2

    # BOUNDED AND UNBOUNDED GAS PARTICLES
    unbounded = total_energy > 0.
    bounded = total_energy <= 0.

    # COLD AND HOT GAS PARTICLES
    cold = gas_temp < 5*1e4
    hot = gas_temp >= 5*1e4

    ##################### RETURN VARIABLES
    total_gas_mass = np.sum(gas_mass)
    bounded_gas_mass = np.sum(gas_mass[bounded])
    unbounded_gas_mass = np.sum(gas_mass[unbounded])
    cold_gas_mass = np.sum(gas_mass[cold])
    hot_gas_mass = np.sum(gas_mass[hot])
    bounded_cold_gas_mass = np.sum(gas_mass[bounded*cold])
    bounded_hot_gas_mass = np.sum(gas_mass[bounded*hot])

    return total_gas_mass, bounded_gas_mass, unbounded_gas_mass, cold_gas_mass, hot_gas_mass, bounded_cold_gas_mass, bounded_hot_gas_mass


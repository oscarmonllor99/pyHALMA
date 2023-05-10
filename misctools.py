"""
HALMA-MASCLET FRAMEWORK

Created on Mon Mar 27 2022

@author: ÓSCAR MONLLOR BERBEGAL
"""

import numpy as np
from numba import njit

# DESCRIPTION
# Misc functions and tools for the HALMA-MASCLET framework

############################################################################################################
############################################################################################################
############################################################################################################
# BLOC 1: Particles/points inside/outside box/sphere/inclined_box etc.
############################################################################################################
############################################################################################################
############################################################################################################

##########################################################
# Function to find the particles inside a box
##########################################################
@njit(fastmath = True)
def which_particles_inside_box(box_limits, dm_positions_x, dm_positions_y, dm_positions_z):
    particles_inside = np.zeros(len(dm_positions_x), dtype = np.int64)
    next_loc = 0
    for particle in range(len(dm_positions_x)):
        inside_x = box_limits[0] < dm_positions_x[particle] < box_limits[1]
        inside_y = box_limits[2] < dm_positions_y[particle] < box_limits[3]
        inside_z = box_limits[4] < dm_positions_z[particle] < box_limits[5]
        if inside_x and inside_y and inside_z:
            particles_inside[next_loc] = particle
            next_loc += 1
    return particles_inside[:next_loc]

##########################################################
# Function to find the particles inside a sphere
##########################################################
@njit(fastmath = True)
def which_particles_inside_sphere(Rmax, center, dm_positions_x, dm_positions_y, dm_positions_z):
    particles_inside = np.zeros(len(dm_positions_x), dtype = np.int64)
    x0 = center[0]
    y0 = center[1]
    z0 = center[2]
    next_loc = 0
    for particle in range(len(dm_positions_x)):
        x = dm_positions_x[particle]
        y = dm_positions_y[particle]
        z = dm_positions_z[particle]
        R = ( (x-x0)**2 + (y-y0)**2 + (z-z0)**2 )**0.5
        if R < Rmax:
            particles_inside[next_loc] = particle
            next_loc += 1

    return particles_inside[:next_loc]

##########################################################
# Function to find the particles inside an inclined box
# The box is defined by the 4 corners P1, P2, P4, P5
# that is, P1, the left, bottom, back corner, and the 
# adjacent corners P2, P4, P5, which are the right, top, front corners
# example:
# P1 = (- wx)*ux + (-wy)*uy + (-wz)*uz + np.array([cx, cy, cz])
# P2 = (+ wx)*ux + (-wy)*uy + (-wz)*uz + np.array([cx, cy, cz])
# P4 = (- wx)*ux + (+wy)*uy + (-wz)*uz + np.array([cx, cy, cz])
# P5 = (- wx)*ux + (-wy)*uy + (+wz)*uz + np.array([cx, cy, cz])
##########################################################
@njit
def point_inside_inclined_box(P1, P2, P4, P5, u, v, w, x, y, z):
    point = np.array([x,y,z])
    inside = False
    if np.dot(u, P1) < np.dot(u, point) < np.dot(u, P2):
        if np.dot(v, P1) < np.dot(v, point) < np.dot(v, P4):
            if np.dot(w, P1) < np.dot(w, point) < np.dot(w, P5):
                inside = True

    return inside

##########################################################
# Function to find if a point (cell center) is inside a galaxy
# aimed to avoid considering ISM in ICM calculations
##########################################################
@njit
def point_inside_galaxy(xgal, ygal, zgal, Rgal, x, y, z):
    inside = False
    dx = xgal - x
    dy = ygal - y
    dz = zgal - z
    if (dx**2 + dy**2 + dz**2)**0.5 < 1. * Rgal:
        inside = True
    return inside





############################################################################################################
############################################################################################################
############################################################################################################
# BLOC 2: Cosmology related functions
############################################################################################################
############################################################################################################
############################################################################################################


##########################################################
# Function to pass from density contrast to density
##########################################################
@njit(fastmath = True)    
def delta_to_rho(delta):
    for parche in range(len(delta)):
        (nx, ny, nz) = np.shape(delta[parche])
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    delta[parche][i,j,k] = (1 + delta[parche][i,j,k]) #in units of rho_B

    return delta





############################################################################################################
############################################################################################################
############################################################################################################
# BLOC 3: Catalogue related functions
############################################################################################################
############################################################################################################
############################################################################################################

##########################################################
# Function to reduce the HALMA catalogue to those iterations of interest
##########################################################
def reduce_halma_catalogue(initial_iteration, final_iteration, total_iteration_data, total_halo_data):
    for init_it_halma in range(len(total_iteration_data)):
        if total_iteration_data[init_it_halma]['it_masclet'] == initial_iteration:
            break

    for final_it_halma in range(len(total_iteration_data)):
        if total_iteration_data[final_it_halma]['it_masclet'] == final_iteration:
            break

    return total_iteration_data[init_it_halma:final_it_halma+1], total_halo_data[init_it_halma:final_it_halma+1]

##########################################################
# Function to find the progenitors of a given halo
##########################################################
def tree(total_iteration_data, total_halo_data, final_it_halma):

    num_iter = len(total_halo_data)
    num_halos_total = total_iteration_data[final_it_halma]['nhal']

    merger_tree = np.zeros((num_halos_total, final_it_halma + 1, 3))
    
    """
    MERGER TREE FORMAT:
        
    [HALO ID IN LAST ITERATION, ITERATION, 0 FOR MAIN PROGENITOR FOR THIS ITERATION
                                           1 NUMBER OF MERGERS
                                           2 MAJOR MERGER -1 NO, +1 YES
    
    """
    for halo in range(num_halos_total):
            it = final_it_halma
            halo_data = total_halo_data[it]
            progenitor_1 = halo_data[halo]['father1']
            num_mergers = halo_data[halo]['nmerg']
            merger_type = halo_data[halo]['mergType']
            merger_tree[halo, it, 0] = progenitor_1
            merger_tree[halo, it, 1] = num_mergers
            merger_tree[halo, it, 2] = merger_type
            
            while (progenitor_1 != 0) and (it > 0):
                it -= 1
                halo_data = total_halo_data[it]
                
                progenitor_1_new = halo_data[progenitor_1-1]['father1']
                num_mergers = halo_data[progenitor_1-1]['nmerg']
                merger_type = halo_data[progenitor_1-1]['mergType']
                
                progenitor_1 = progenitor_1_new
                
                merger_tree[halo, it, 0] = progenitor_1
                merger_tree[halo, it, 1] = num_mergers
                merger_tree[halo, it, 2] = merger_type
                
            
        
    return merger_tree

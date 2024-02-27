# Friends-of-Friends algorithm for pyHALMA
# Implemented by means of scipy.spatial.KDTree
# 19/02/2024

import numpy as np
from numba import get_num_threads
import multiprocessing
from scipy.spatial import KDTree
#################################################
from python_scripts import halo_properties as hp



############################################################################################################
# INTERNAL FRIENDS OF FRIENDS
############################################################################################################

def friends_of_friends(data, linking_length):
    #Build tree
    tree = KDTree(data)

    # Number of particles
    npart = data.shape[0]

    final_groups = []
    already_looked = np.zeros(npart, dtype=bool)
    already_friend = np.zeros(npart, dtype=bool)
    # Speed hugely depends on iterations_to_clean
    #        low-values --> slow due to extensive use of np.unique
    #        high-values --> slow due to the list of friends growing indefinitely
    # Good values are around 50-100
    
    iterations_to_clean = 64
    for ip in range(npart):
        # Check if it has been already looked and remove it from the list
        if already_looked[ip]:
            continue

        # If not, find the friends of the particle and mark them as already looked
        friends = tree.query_ball_point(data[ip], linking_length)

        # If the particle has no friends, continue
        if len(friends) <= 1:
            continue

        friends_save = friends.copy()
        already_looked[ip] = True
        already_friend[friends] = True

        it_ip = 0
        while len(friends) > 0:
            # Take the first friend
            ip2 = friends[0]
            friends.pop(0)

            # Check if it has been already looked and remove it from the list
            if already_looked[ip2]:
                continue
            
            # If it has not been already looked, add its friends to the list of friends
            query = tree.query_ball_point(data[ip2], linking_length)
            query = [x for x in query if not already_looked[x]]
            query = [x for x in query if not already_friend[x]]
    
            friends += query
            friends_save += query
            already_looked[ip2] = True

            # np.unique and already_friends are used to remove duplicates, otherwise the list of friends may grow indefinitely
            if it_ip % iterations_to_clean == 0 and it_ip != 0:
                #friends = np.unique(np.array(friends)).tolist()
                already_friend[friends] = True

            it_ip += 1

        friends_save = np.unique(np.array(friends_save)).tolist()
        final_groups.append(friends_save)

    return final_groups



############################################################################################################
# INTERNAL FRIENDS OF FRIENDS WRAPPERS
############################################################################################################
# serial version
def friends_of_friends_serial(st_x, st_y, st_z, linking_length):
    data = np.array((st_x, st_y, st_z)).T
    return friends_of_friends(data, linking_length)



# parallel version
# domain decomposition in 3D subdomains (one subdomain per process)
def friends_of_friends_parallel(st_x, st_y, st_z, linking_length, L, minp, st_mass):
    #How many cores to use (has to be n³, n=2,3,4,...) in order
    #for the subdomains to be cubic
    mandatories = np.array([1, 8, 64, 512])
    ncores = np.max(mandatories[mandatories <= get_num_threads()])
    
    #Define subdomains
    nsides = int(np.round(ncores**(1/3)))
    sub_L = L/nsides
    sub_x_buffer = np.zeros((ncores, 2))
    sub_y_buffer = np.zeros((ncores, 2))
    sub_z_buffer = np.zeros((ncores, 2))
    sub_x = np.zeros((ncores, 2))
    sub_y = np.zeros((ncores, 2))
    sub_z = np.zeros((ncores, 2))
    # We add a buffer to the subdomains to
    # take into account particles that are close to the boundary in
    # other subdomains
    # We will check for duplicates in the buffer zone after the FOF
    buffer = 0.1 #100 kpc
    
    ic = 0
    for ix in range(nsides):
        for iy in range(nsides):
            for iz in range(nsides):
                sub_x[ic,0] = -L/2 + ix*sub_L
                sub_x[ic,1] = -L/2 + (ix+1)*sub_L
                sub_y[ic,0] = -L/2 + iy*sub_L
                sub_y[ic,1] = -L/2 + (iy+1)*sub_L
                sub_z[ic,0] = -L/2 + iz*sub_L
                sub_z[ic,1] = -L/2 + (iz+1)*sub_L
                sub_x_buffer[ic,0] = sub_x[ic,0] - buffer
                sub_x_buffer[ic,1] = sub_x[ic,1] + buffer
                sub_y_buffer[ic,0] = sub_y[ic,0] - buffer
                sub_y_buffer[ic,1] = sub_y[ic,1] + buffer
                sub_z_buffer[ic,0] = sub_z[ic,0] - buffer
                sub_z_buffer[ic,1] = sub_z[ic,1] + buffer
                ic += 1

    #Split data
    data_sub = []
    core_particles = []
    all_particle_list = np.arange(st_x.shape[0])
    for ic in range(ncores):
        mask =        (st_x >= sub_x_buffer[ic,0]) & (st_x < sub_x_buffer[ic,1])
        mask = mask & (st_y >= sub_y_buffer[ic,0]) & (st_y < sub_y_buffer[ic,1])
        mask = mask & (st_z >= sub_z_buffer[ic,0]) & (st_z < sub_z_buffer[ic,1])
        
        data_x = st_x[mask] 
        data_y = st_y[mask] 
        data_z = st_z[mask] 

        data_ic = np.array((data_x, data_y, data_z)).T
        data_sub.append(data_ic)
        core_particles.append(all_particle_list[mask])


    # Compute friends of friends in parallel --> each process computes the friends of friends in a subdomain
    with multiprocessing.get_context('fork').Pool(ncores) as pool:
        results = pool.starmap(friends_of_friends, [(data_sub[i], linking_length) for i in range(ncores)])
        
    # Join groups
    groups = []
    for ic in range(ncores):
        groups_ic = results[ic]
        for ig in range(len(groups_ic)):
             if len(groups_ic[ig]) > minp:
                groups.append((ic, core_particles[ic][groups_ic[ig]]))

    # Check center of mass of groups is inside the corresponding subdomain
    # to avoid duplicates
    final_groups = []
    for ig in range(len(groups)):
        ic = groups[ig][0]
        group = groups[ig][1]
        (cx, cy, cz, _) = hp.center_of_mass(group, st_x, st_y, st_z, st_mass)
        if (     sub_x[ic,0] <= cx <= sub_x[ic,1] 
             and sub_y[ic,0] <= cy <= sub_y[ic,1] 
             and sub_z[ic,0] <= cz <= sub_z[ic,1]   ):
            final_groups.append(group)
    
    return final_groups



############################################################################################################
# PYFOF WRAPPERS
############################################################################################################
# serial version FOR PYFOF
def pyfof_friends_of_friends_serial(st_x, st_y, st_z, linking_length):
    import pyfof
    data = np.array((st_x, st_y, st_z)).T.astype(np.float64)
    return pyfof.friends_of_friends(data, linking_length)


# parallel version FOR PYFOF
# domain decomposition in 3D subdomains (one subdomain per process)
def pyfof_friends_of_friends_parallel(st_x, st_y, st_z, linking_length, L, minp, st_mass):
    import pyfof
    #How many cores to use (has to be n³, n=2,4,8...) in order
    #for the subdomains to be cubic and cpu-friendly
    mandatories = np.array([1, 8, 64, 512])
    ncores = np.max(mandatories[mandatories <= get_num_threads()])
    
    #Define subdomains
    nsides = int(np.round(ncores**(1/3)))
    sub_L = L/nsides
    sub_x_buffer = np.zeros((ncores, 2))
    sub_y_buffer = np.zeros((ncores, 2))
    sub_z_buffer = np.zeros((ncores, 2))
    sub_x = np.zeros((ncores, 2))
    sub_y = np.zeros((ncores, 2))
    sub_z = np.zeros((ncores, 2))
    # Add a buffer to the subdomains to take into account particles 
    #    that are close to the boundary.
    # Will check for duplicates in the buffer zone after FoF.
    buffer = 0.1 #100 kpc
    ic = 0
    for ix in range(nsides):
        for iy in range(nsides):
            for iz in range(nsides):
                sub_x[ic,0] = -L/2 + ix*sub_L
                sub_x[ic,1] = -L/2 + (ix+1)*sub_L
                sub_y[ic,0] = -L/2 + iy*sub_L
                sub_y[ic,1] = -L/2 + (iy+1)*sub_L
                sub_z[ic,0] = -L/2 + iz*sub_L
                sub_z[ic,1] = -L/2 + (iz+1)*sub_L
                sub_x_buffer[ic,0] = sub_x[ic,0] - buffer
                sub_x_buffer[ic,1] = sub_x[ic,1] + buffer
                sub_y_buffer[ic,0] = sub_y[ic,0] - buffer
                sub_y_buffer[ic,1] = sub_y[ic,1] + buffer
                sub_z_buffer[ic,0] = sub_z[ic,0] - buffer
                sub_z_buffer[ic,1] = sub_z[ic,1] + buffer
                ic += 1

    #Split data
    data_sub = []
    core_particles = []
    all_particle_list = np.arange(st_x.shape[0])
    for ic in range(ncores):
        mask =        (st_x >= sub_x_buffer[ic,0]) & (st_x < sub_x_buffer[ic,1])
        mask = mask & (st_y >= sub_y_buffer[ic,0]) & (st_y < sub_y_buffer[ic,1])
        mask = mask & (st_z >= sub_z_buffer[ic,0]) & (st_z < sub_z_buffer[ic,1])
        
        data_x = st_x[mask] 
        data_y = st_y[mask] 
        data_z = st_z[mask] 

        data_ic = np.array((data_x, data_y, data_z)).T.astype(np.float64)
        data_sub.append(data_ic)
        core_particles.append(all_particle_list[mask])


    # Compute friends of friends in parallel --> each process computes the friends of friends in a subdomain
    with multiprocessing.get_context('fork').Pool(ncores) as pool:
        results = pool.starmap(pyfof.friends_of_friends, [(data_sub[i], linking_length) for i in range(ncores)])
        
    # Join groups
    groups = []
    for ic in range(ncores):
        groups_ic = results[ic]
        for ig in range(len(groups_ic)):
             if len(groups_ic[ig]) > minp:
                groups.append((ic, core_particles[ic][groups_ic[ig]]))

    # Check center of mass of groups is inside the corresponding subdomain
    # to avoid duplicates
    final_groups = []
    for ig in range(len(groups)):
        ic = groups[ig][0]
        group = groups[ig][1]
        (cx, cy, cz, _) = hp.center_of_mass(group, st_x, st_y, st_z, st_mass)
        if (     sub_x[ic,0] <= cx <= sub_x[ic,1] 
             and sub_y[ic,0] <= cy <= sub_y[ic,1] 
             and sub_z[ic,0] <= cz <= sub_z[ic,1]   ):
            final_groups.append(group)
    
    return final_groups
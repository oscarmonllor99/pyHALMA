# Friends-of-Friends algorithm for pyHALMA
# Implemented by means of scipy.spatial.KDTree
# 19/02/2024

import numpy as np
from numba import get_num_threads, njit
from multiprocessing import Pool
from tqdm import tqdm
from scipy.spatial import KDTree
import time

def friends_of_friends(data, linking_length, L):
    #Build tree
    tree = KDTree(data, boxsize=np.array([L,L,L]), leafsize = 64)

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

    for ip in tqdm(range(npart)):
        # Check if it has been already looked and remove it from the list
        if already_looked[ip]:
            continue

        # If not, find the friends of the particle and mark them as already looked
        friends = tree.query_ball_point(data[ip], linking_length)
        if len(friends) < 1:
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

            # np.unique is used to remove duplicates, otherwise the list of friends may grow indefinitely
            if it_ip % iterations_to_clean == 0 and it_ip != 0:
                #friends = np.unique(np.array(friends)).tolist()
                already_friend[friends] = True

            it_ip += 1

        friends_save = np.unique(np.array(friends_save)).tolist()
        final_groups.append(friends_save)

    return final_groups



# serial version
def friends_of_friends_serial(st_x, st_y, st_z, linking_length, L):
    data = np.array((st_x, st_y, st_z)).T
    data = data + L/2 # shift to 0, L
    data[data >= L] = data[data >= L] - L # periodic boundary conditions
    return friends_of_friends(data, linking_length, L)



# Find which groups have problematic particles
# Merge those with the same problematic particles
def merge_groups(groups, problematic_particles):
    for ip in problematic_particles:
        groups_to_merge = []
        for ig in range(len(groups)):
            if ip in groups[ig]:
                groups_to_merge.append(ig)

        if len(groups_to_merge) > 1:
            new_group = []
            for ig in groups_to_merge:
                new_group += groups[ig]
                groups[ig] = []
            
            new_group = np.unique(new_group)
            groups.append(new_group)

    return groups

# parallel version
# domain decomposition in 3D subdomains (one subdomain per process)
def friends_of_friends_parallel(st_x, st_y, st_z, linking_length, L, minp):
    #How many cores to use (has to be n³, n=2,3,4,...) in order
    #for the subdomains to be cubic
    mandatories = np.array([1, 8, 27, 64])
    ncores = np.max(mandatories[mandatories <= get_num_threads()])
    

    #Define subdomains
    nsides = int(np.round(ncores**(1/3)))
    sub_L = L/nsides
    sub_x = np.zeros((ncores, 2))
    sub_y = np.zeros((ncores, 2))
    sub_z = np.zeros((ncores, 2))
    
    # We add a buffer to the subdomains to
    # take into account particles that are close to the boundary in
    # other subdomains
    buffer = 2*linking_length
    ic = 0
    for ix in range(nsides):
        for iy in range(nsides):
            for iz in range(nsides):
                sub_x[ic,0] = -L/2 + ix*sub_L     - buffer
                sub_x[ic,1] = -L/2 + (ix+1)*sub_L + buffer
                sub_y[ic,0] = -L/2 + iy*sub_L     - buffer
                sub_y[ic,1] = -L/2 + (iy+1)*sub_L + buffer
                sub_z[ic,0] = -L/2 + iz*sub_L     - buffer
                sub_z[ic,1] = -L/2 + (iz+1)*sub_L + buffer
                ic += 1

    t0 = time.time()

    #Split data
    data_sub = []
    core_particles = []
    all_particle_list = np.arange(st_x.shape[0])
    for ic in range(ncores):
        mask =        (st_x >= sub_x[ic,0]) & (st_x < sub_x[ic,1])
        mask = mask & (st_y >= sub_y[ic,0]) & (st_y < sub_y[ic,1])
        mask = mask & (st_z >= sub_z[ic,0]) & (st_z < sub_z[ic,1])
        
        data_x = st_x[mask] - sub_x[ic,0] # shift to 0, sub_L_x
        data_y = st_y[mask] - sub_y[ic,0] # shift to 0, sub_L_y
        data_z = st_z[mask] - sub_z[ic,0] # shift to 0, sub_L_z

        data_ic = np.array((data_x, data_y, data_z)).T
        data_ic[data_ic >= sub_L] = data_ic[data_ic >= sub_L] - sub_L # periodic boundary conditions
        data_sub.append(data_ic)
        core_particles.append(all_particle_list[mask])

    print('Data split in', time.time()-t0, 's')

    t0 = time.time()
    # Compute friends of friends in parallel --> each process computes the friends of friends in a subdomain
    with Pool(ncores) as pool:
        results = pool.starmap(friends_of_friends, [(data_sub[i], linking_length, sub_L) for i in range(ncores)])

    # Reconstruct final groups
    groups = []
    for ic in range(ncores):
        groups_ic = results[ic]
        for ig in range(len(groups_ic)):
             if len(groups_ic[ig]) >= minp:
                groups.append(core_particles[ic][groups_ic[ig]])
    
    print('Parallel computation in', time.time()-t0, 's')

    # t0 = time.time()
    # #Find problematic particles
    # #Particles that are close to the boundary of the subdomain
    # problematic_particles = []
    # for ic in range(ncores):
    #     mask =        (np.abs(st_x - sub_x[ic,0]) < buffer) | (np.abs(st_x - sub_x[ic,1]) < buffer)
    #     mask = mask | (np.abs(st_y - sub_y[ic,0]) < buffer) | (np.abs(st_y - sub_y[ic,1]) < buffer)
    #     mask = mask | (np.abs(st_z - sub_z[ic,0]) < buffer) | (np.abs(st_z - sub_z[ic,1]) < buffer)
    #     problematic_particles += all_particle_list[mask].tolist()

    # final_groups = merge_groups(groups, problematic_particles)
    final_groups = groups
    print('Problematic particles in', time.time()-t0, 's')
    # Reconstruct final groups

    return final_groups
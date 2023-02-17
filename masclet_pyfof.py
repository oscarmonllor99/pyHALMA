import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pyfof
import sys
from numba import njit
sys.path.append('/Users/monllor/projects/')
from masclet_framework import read_masclet

"""
Help on module pyfof:

NAME
    pyfof - Created on Thu Mar 26 17:59:28 2015

DESCRIPTION
    @author: sljg2

FUNCTIONS
    friends_of_friends(...)
        Computes friends-of-friends clustering of data. Distances are computed
        using a euclidian metric.
        
            :param data: A numpy array with dimensions (npoints x ndim)
        
            :param linking_length: The linking length between cluster members
        
            :rtype: A list of lists of indices in each cluster type

DATA
    __all__ = ['friends_of_friends']
    __copyright__ = 'Copyright 2015 Simon Gibbons'
    __test__ = {}

VERSION
    0.1.3-dev

AUTHOR
    Simon Gibbons (sljg2@ast.cam.ac.uk)

FILE
    /Users/monllor/opt/anaconda3/lib/python3.9/site-packages/pyfof.cpython-39-darwin.so

"""

with open('masclet_pyfof.dat', 'r') as f:
    f.readline()
    f.readline()
    first,last,step = np.array(f.readline().split()[0].split(','), dtype = np.int32)
    f.readline()
    nx, ny, nz = np.array(f.readline().split()[0].split(','), dtype = np.int32)
    f.readline()
    ache, omega0, t0 =  np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    z0, L = np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    ll, =  np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    minp, =  np.array(f.readline().split()[0].split(','), dtype = np.int32)
    f.readline()
    old_masclet = bool(int(f.readline()))
    f.readline()
    path_results = f.readline()

it_count = 0
#loop over iterations


########## FUNCTIONS TO BE CALLED ########## 
def good_groups(groups):
    good_index = np.zeros((len(groups), ), dtype = bool)  
    for ig in range(len(groups)):
        if len(groups[ig]) >= minp:
            good_index[ig] = True
    return good_index
########## ########## ########## ########## ########## 



for iteration in range(first, last+step, step):
    #open MASCLET files
    masclet_st_data = read_masclet.read_clst(iteration, path = path_results, parameters_path=path_results, 
                                                    digits=5, max_refined_level=1000, 
                                                    output_deltastar=False, verbose=False, output_position=True, 
                                                    output_velocity=False, output_mass=True, output_temp=False,
                                                    output_metalicity=False, output_id=False, are_BH = not old_masclet)

    st_positions_x = masclet_st_data[0] #in MPC
    st_positions_y = masclet_st_data[1]
    st_positions_z = masclet_st_data[2]

    data = np.vstack((st_positions_x, st_positions_y, st_positions_z)).T
    data = data.astype(np.float64)

    #APPLY FOF
    groups = pyfof.friends_of_friends(data = data, linking_length = ll)
    groups = np.array(groups)
    #CLEAN THOSE HALOES with npart < minp
    groups = groups[good_groups(groups)]

    #CALCULATE CM, CM_vel AND phase-space cleaning


    #CALCULATE HALO PROPERTIES

    it_count += 1



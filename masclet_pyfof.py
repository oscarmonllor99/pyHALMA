import numpy as np
import matplotlib.pyplot as plt
import photutils
import pyimfit
from photutils.isophote import EllipseGeometry, Ellipse, build_ellipse_model
from photutils.aperture import EllipticalAperture
from scipy.ndimage import gaussian_filter1d, gaussian_filter
import pyfof
import sys
from scipy.ndimage import rotate
from scipy.optimize import curve_fit, least_squares
from scipy.interpolate import RegularGridInterpolator
from numba import njit, prange, set_num_threads
import time
#Our things
sys.path.append('/home/monllor/projects/')
from masclet_framework import read_masclet, units, particles
import sph3D
from tqdm import tqdm

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
    sig_fil, q_fil = np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    old_masclet = bool(int(f.readline()))
    f.readline()
    write_particles = bool(int(f.readline()))
    f.readline()
    sersic_flag = bool(int(f.readline()))
    f.readline()
    path_halo_particles = f.readline()[:-1]
    f.readline()
    path_results = f.readline()[:-1]
    f.readline()
    catalogue_name = f.readline()[:-1]
    f.readline()
    ncore = int(f.readline())



########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
##########     FUNCTIONS CALLED BY MAIN     ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 

def good_groups(groups):
    good_index = np.zeros((len(groups), ), dtype = bool)  
    for ig in range(len(groups)):
        if len(groups[ig]) >= minp:
            good_index[ig] = True
    return good_index

@njit(parallel = True)
def total_mass(part_list, st_mass):
    M = 0.
    npart = len(part_list)
    for ip in prange(npart):
        ipp = part_list[ip]
        M += st_mass[ipp]
    return M

@njit(parallel = True)
def center_of_mass(part_list, st_x, st_y, st_z, st_mass):
    cx = 0.
    cy = 0.
    cz = 0.
    M = 0.
    npart = len(part_list)
    for ip in prange(npart):
        ipp = part_list[ip]
        cx += st_mass[ipp]*st_x[ipp]
        cy += st_mass[ipp]*st_y[ipp]
        cz += st_mass[ipp]*st_z[ipp]
        M += st_mass[ipp]

    if M > 0:
        return cx/M, cy/M, cz/M, M
    else:
        return 0., 0., 0., 0.

@njit(parallel = True)
def CM_velocity(M, part_list, st_vx, st_vy, st_vz, st_mass):
    vx = 0.
    vy = 0.
    vz = 0.
    npart = len(part_list)
    for ip in prange(npart):
        ipp = part_list[ip]
        vx += st_mass[ipp]*st_vx[ipp]
        vy += st_mass[ipp]*st_vy[ipp]
        vz += st_mass[ipp]*st_vz[ipp]
    
    if M>0.:
        return vx/M, vy/M, vz/M
    else:
        return 0., 0., 0.

@njit(parallel = True)
def tully_fisher_velocity(part_list, cx, cy, cz, st_x, st_y, st_z, vx, vy, vz, st_vx, st_vy, st_vz, st_mass, RAD05):
    mass_contribution = 0.
    v_TF = 0.
    npart = len(part_list)
    for ip in prange(npart):
        ipp = part_list[ip]
        dx = cx - st_x[ipp]
        dy = cy - st_y[ipp]
        dz = cz - st_z[ipp]
        dist = (dx**2 + dy**2 + dz**2)**0.5
        if 0.9*RAD05 < dist < 1.1*RAD05:
            v_TF += ( (st_vx[ipp] - vx)**2 + (st_vy[ipp] - vy)**2 + (st_vz[ipp] - vz)**2 )**0.5 * st_mass[ipp]
            mass_contribution += st_mass[ipp]
            
    return v_TF/mass_contribution

@njit
def furthest_particle(cx, cy, cz, part_list, st_x, st_y, st_z):
    RRHH = 0.
    npart = len(part_list)
    for ip in range(npart):
        ipp = part_list[ip]
        dx = cx - st_x[ipp]
        dy = cy - st_y[ipp]
        dz = cz - st_z[ipp]
        dist = (dx**2 + dy**2 + dz**2)**0.5
        if dist > RRHH:
            RRHH = dist

    return RRHH

@njit
def calc_cell(cx, cy, cz, grid, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, vcm_cell, mass_cell, quantas_cell, sig3D_cell):
    npart = len(part_list)
    n_cell = len(grid)
    for ip in range(npart):
        ipp = part_list[ip]
        x_halo = st_x[ipp] - cx
        y_halo = st_y[ipp] - cy
        z_halo = st_z[ipp] - cz
        ix = np.argmin(np.abs(grid - x_halo))
        iy = np.argmin(np.abs(grid - y_halo))
        iz = np.argmin(np.abs(grid - z_halo))
        vcm_cell[ix, iy, iz, 0] += st_vx[ipp]*st_mass[ipp]
        vcm_cell[ix, iy, iz, 1] += st_vy[ipp]*st_mass[ipp]
        vcm_cell[ix, iy, iz, 2] += st_vz[ipp]*st_mass[ipp]
        mass_cell[ix, iy, iz] += st_mass[ipp]
        quantas_cell[ix, iy, iz] += 1
    
    #velocity normalization
    for ix in range(n_cell):
        for iy in range(n_cell):
            for iz in range(n_cell):
                if quantas_cell[ix, iy, iz]>0:
                    vcm_cell[ix, iy, iz, :] /= mass_cell[ix, iy, iz]

    for ip in range(npart):
        ipp = part_list[ip]
        x_halo = st_x[ipp] - cx
        y_halo = st_y[ipp] - cy
        z_halo = st_z[ipp] - cz
        ix = np.argmin(np.abs(grid - x_halo))
        iy = np.argmin(np.abs(grid - y_halo))
        iz = np.argmin(np.abs(grid - z_halo)) 
        dVx =  vcm_cell[ix, iy, iz, 0] - st_vx[ipp]
        dVy =  vcm_cell[ix, iy, iz, 1] - st_vy[ipp]
        dVz =  vcm_cell[ix, iy, iz, 2] - st_vz[ipp]
        sig3D_cell[ix, iy, iz] += (dVx**2 + dVy**2 + dVz**2)

    #sigma normalization
    for ix in range(n_cell):
        for iy in range(n_cell):
            for iz in range(n_cell):
                if quantas_cell[ix, iy, iz]>0:
                    sig3D_cell[ix, iy, iz] = (sig3D_cell[ix, iy, iz]/3.0/quantas_cell[ix, iy, iz])**0.5

    return vcm_cell, mass_cell, quantas_cell, sig3D_cell

@njit
def clean_cell(cx, cy, cz, M, RRHH, grid, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, vcm_cell, mass_cell, quantas_cell, sig3D_cell, ll, q_fil, sig_fil):
    npart = len(part_list)
    control = np.ones(npart, dtype=np.int32)
    cleaned = 0
    remaining = npart
    
    fac_sig = 2.0 #fac_sig
    for ip in range(npart):
        ipp = part_list[ip]
        x_halo = st_x[ipp] - cx
        y_halo = st_y[ipp] - cy
        z_halo = st_z[ipp] - cz
        ix = np.argmin(np.abs(grid - x_halo))
        iy = np.argmin(np.abs(grid - y_halo))
        iz = np.argmin(np.abs(grid - z_halo))
        #si hay más partículas que el mínimo requerido q_fil
        if quantas_cell[ix, iy, iz] > q_fil:
            dVx =  vcm_cell[ix, iy, iz, 0] - st_vx[ipp]
            dVy =  vcm_cell[ix, iy, iz, 1] - st_vy[ipp]
            dVz =  vcm_cell[ix, iy, iz, 2] - st_vz[ipp]
            bas = (dVx**2 + dVy**2 + dVz**2)**0.5
            if quantas_cell[ix, iy, iz] > 2:
                bas = bas/sig3D_cell[ix, iy, iz]
            else:
                bas = 0.

            #si la sigma de la particula es mayor que un factor
            # fac_sig*sig_fil veces la sigma de la celda, se desecha
            if bas >= fac_sig*sig_fil:
                control[ip] = 0
                cleaned += 1
                remaining -= 1
        else:
            control[ip] = 0
            cleaned += 1
            remaining -= 1

    clean_particles = part_list[np.argwhere(control).flatten()]
    #NEW GRID
    cx, cy, cz, M = center_of_mass(clean_particles, st_x, st_y, st_z, st_mass)
    RRHH = furthest_particle(cx, cy, cz, clean_particles, st_x, st_y, st_z)
    grid = np.arange(-(RRHH+ll), RRHH+ll, 2*ll)
    n_cell = len(grid)
    vcm_cell = np.zeros((n_cell, n_cell, n_cell, 3))
    mass_cell = np.zeros((n_cell, n_cell, n_cell))
    quantas_cell = np.zeros((n_cell, n_cell, n_cell))
    sig3D_cell = np.zeros((n_cell, n_cell, n_cell))
    #REDUCE FAC SIG and CONTINUE CLEANING until fac_sig = 1 or frac <= 0.1+
    fac_sig *= 0.75
    if remaining > 0:
        frac = cleaned/remaining
        while frac > 0.1: #10% tolerance
            #calculate quantities in grid
            vcm_cell, mass_cell, quantas_cell, sig3D_cell = calc_cell(cx, cy, cz, grid, clean_particles, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, vcm_cell, mass_cell, quantas_cell, sig3D_cell)
            cleaned = 0
            for ip in range(npart):
                if bool(control[ip]):
                    ipp = part_list[ip]
                    x_halo = st_x[ipp] - cx
                    y_halo = st_y[ipp] - cy
                    z_halo = st_z[ipp] - cz
                    ix = np.argmin(np.abs(grid - x_halo))
                    iy = np.argmin(np.abs(grid - y_halo))
                    iz = np.argmin(np.abs(grid - z_halo))
                    #si hay más partículas que el mínimo requerido q_fil
                    if quantas_cell[ix, iy, iz] > q_fil:
                        dVx =  vcm_cell[ix, iy, iz, 0] - st_vx[ipp]
                        dVy =  vcm_cell[ix, iy, iz, 1] - st_vy[ipp]
                        dVz =  vcm_cell[ix, iy, iz, 2] - st_vz[ipp]
                        bas = (dVx**2 + dVy**2 + dVz**2)**0.5
                        if quantas_cell[ix, iy, iz] >= 2:
                            bas = bas/sig3D_cell[ix, iy, iz]
                        else:
                            bas = 0.

                        #si la sigma de la particula es mayor que un factor
                        # fac_sig*sig_fil veces la sigma de la celda, se desecha
                        if bas >= fac_sig*sig_fil:
                            control[ip] = 0
                            cleaned += 1
                            remaining -= 1
                    else:
                        control[ip] = 0
                        cleaned += 1
                        remaining -= 1

            #NEW GRID
            clean_particles = part_list[np.argwhere(control).flatten()]
            cx, cy, cz, M = center_of_mass(clean_particles, st_x, st_y, st_z, st_mass)
            RRHH = furthest_particle(cx, cy, cz, clean_particles, st_x, st_y, st_z)
            grid = np.arange(-(RRHH+ll), RRHH+ll, 2*ll)
            n_cell = len(grid)
            vcm_cell = np.zeros((n_cell, n_cell, n_cell, 3))
            mass_cell = np.zeros((n_cell, n_cell, n_cell))
            quantas_cell = np.zeros((n_cell, n_cell, n_cell))
            sig3D_cell = np.zeros((n_cell, n_cell, n_cell))

            if remaining == 0:
                return 0., 0., 0., 0., 0, control
            else:
                #FRACTION OF CLEANED PARTICLES
                frac = cleaned/remaining
                #New fac_sig
                fac_sig *= 0.75
                fac_sig = max(1., fac_sig)

    return cx, cy, cz, M, RRHH, control


@njit
def escape_velocity_cleaning(cx, cy, cz, vx, vy, vz, M, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, control, factor_v):
    npart = len(part_list)
    mass = M*units.sun_to_kg
    G = units.G_isu
    for ip in range(npart):
        if bool(control[ip]):
            ipp = part_list[ip]
            dx = cx - st_x[ipp]
            dy = cy - st_y[ipp]
            dz = cz - st_z[ipp]
            dist = (dx**2 + dy**2 + dz**2)**0.5 * units.mpc_to_m

            v_esc = (2*G*mass/dist)**0.5

            dvx = vx - st_vx[ipp]
            dvy = vy - st_vy[ipp]
            dvz = vz - st_vz[ipp]
            V = (dvx**2 + dvy**2 + dvz**2)**0.5 * 1e3

            if V > factor_v*v_esc:
                control[ip] = 0

    return control


@njit
def half_mass_radius(cx, cy, cz, M, part_list, st_x, st_y, st_z, st_mass):
    npart = len(part_list)
    #FIRST SORT PARTICLES BY DISTANCE TO CM

    RAD_part = np.zeros(npart)
    for ip in range(npart):
        ipp = part_list[ip]
        dx = cx - st_x[ipp]
        dy = cy - st_y[ipp]
        dz = cz - st_z[ipp]
        dist = (dx**2 + dy**2 + dz**2)**0.5
        RAD_part[ip] = dist

    RAD_sorted_index = np.argsort(RAD_part)
    part_list_sorted = part_list[RAD_sorted_index]
    RAD_part = np.sort(RAD_part)
    RAD05 = np.min(RAD_part)
    mass_sum = 0.
    for ip in range(npart):
        ipp = part_list_sorted[ip]
        mass_sum += st_mass[ipp]
        if mass_sum > 0.5*M:
            break
        else:
            RAD05 = RAD_part[ip]

    return RAD05

@njit
def half_mass_radius_proj(cx, cy, cz, M, part_list, st_x, st_y, st_z, st_mass):
    npart = len(part_list)
    #FIRST SORT PARTICLES BY DISTANCE TO CM

    RAD_part_x = np.zeros(npart)
    RAD_part_y = np.zeros(npart)
    RAD_part_z = np.zeros(npart)
    for ip in range(npart):
        ipp = part_list[ip]
        dx = cx - st_x[ipp]
        dy = cy - st_y[ipp]
        dz = cz - st_z[ipp]
        dist_x = (dy**2 + dz**2)**0.5
        dist_y = (dx**2 + dz**2)**0.5
        dist_z = (dx**2 + dy**2)**0.5
        RAD_part_x[ip] = dist_x
        RAD_part_y[ip] = dist_y
        RAD_part_z[ip] = dist_z

    RAD_sorted_index_x = np.argsort(RAD_part_x)
    part_list_sorted_x = part_list[RAD_sorted_index_x]
    RAD_part_x = np.sort(RAD_part_x)
    RAD05_x = np.min(RAD_part_x)

    mass_sum_x = 0.
    for ip in range(npart):
        ipp = part_list_sorted_x[ip]
        mass_sum_x += st_mass[ipp]
        if mass_sum_x > 0.5*M:
            break
        else:
            RAD05_x = RAD_part_x[ip]

    RAD_sorted_index_y = np.argsort(RAD_part_y)
    part_list_sorted_y = part_list[RAD_sorted_index_y]
    RAD_part_y = np.sort(RAD_part_y)
    RAD05_y = np.min(RAD_part_y)

    mass_sum_y = 0.
    for ip in range(npart):
        ipp = part_list_sorted_y[ip]
        mass_sum_y += st_mass[ipp]
        if mass_sum_y > 0.5*M:
            break
        else:
            RAD05_y = RAD_part_y[ip]

    RAD_sorted_index_z = np.argsort(RAD_part_z)
    part_list_sorted_z = part_list[RAD_sorted_index_z]
    RAD_part_z = np.sort(RAD_part_z)
    RAD05_z = np.min(RAD_part_z)

    mass_sum_z = 0.
    for ip in range(npart):
        ipp = part_list_sorted_z[ip]
        mass_sum_z += st_mass[ipp]
        if mass_sum_z > 0.5*M:
            break
        else:
            RAD05_z = RAD_part_z[ip]

    return RAD05_x, RAD05_y, RAD05_z


@njit(parallel = True)
def angular_momentum(M, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, cx, cy, cz, vx, vy, vz):
    lx = 0.
    ly = 0.
    lz = 0.
    npart = len(part_list)
    for ip in prange(npart):
        ipp = part_list[ip]
        vvx = st_vx[ipp] - vx
        vvy = st_vy[ipp] - vy
        vvz = st_vz[ipp] - vz
        rx = st_x[ipp] - cx
        ry = st_y[ipp] - cy
        rz = st_z[ipp] - cz

        lx += st_mass[ipp]*(ry*vvz - rz*vvy)
        ly += st_mass[ipp]*(rz*vvx - rx*vvz)
        lz += st_mass[ipp]*(rx*vvy - ry*vvz)

    return lx/M, ly/M, lz/M

@njit
def density_peak(part_list, st_x, st_y, st_z, st_mass):
    N_cells = 50
    grid_x = np.linspace(np.min(st_x[part_list]) - ll, np.max(st_x[part_list]) + ll, N_cells)
    grid_y = np.linspace(np.min(st_y[part_list]) - ll, np.max(st_y[part_list]) + ll, N_cells)
    grid_z = np.linspace(np.min(st_z[part_list]) - ll, np.max(st_z[part_list]) + ll, N_cells)

    nx = len(grid_x)
    ny = len(grid_y)
    nz = len(grid_z)

    cell_mass = np.zeros((nx,ny,nz))
    
    Npart = len(part_list)
    for ip in range(Npart):
        ipp = part_list[ip]
        ix = np.argmin(np.abs(grid_x - st_x[ipp]))
        iy = np.argmin(np.abs(grid_y - st_y[ipp]))
        iz = np.argmin(np.abs(grid_z - st_z[ipp]))
        cell_mass[ix, iy, iz] += st_mass[ipp]

    xmax = int(nx/2)
    ymax = int(ny/2)
    zmax = int(nz/2)
    mass_max = 0.
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                if cell_mass[ix, iy, iz] > mass_max:
                    xmax = ix
                    ymax = iy
                    zmax = iz
                    mass_max = cell_mass[ix, iy, iz]


    return grid_x[xmax], grid_y[ymax], grid_z[zmax]

@njit(parallel = True)
def star_formation(part_list, st_mass, st_age, cosmo_time, dt):
    mass_sfr = 0.
    Npart = len(part_list)
    for ip in prange(Npart):
        ipp = part_list[ip]
        if st_age[ipp] > (cosmo_time-1.1*dt): #10% tolerance
               mass_sfr += st_mass[ipp]

    return mass_sfr 

@njit(parallel = True)
def sigma_effective(part_list, R05, st_x, st_y, st_z, st_vx, st_vy, st_vz, cx, cy, cz, vx, vy, vz):
    Npart = len(part_list)
    sigma_05 = 0.
    part_inside = 0
    for ip in prange(Npart):
        ipp = part_list[ip]
        dx = cx - st_x[ipp]
        dy = cy - st_y[ipp]
        dz = cz - st_z[ipp]
        dist = (dx**2 + dy**2 + dz**2)**0.5
        if dist < R05:
            sigma_05 += (st_vx[ipp]-vx)**2 + (st_vy[ipp]-vy)**2 +(st_vz[ipp]-vz)**2
            part_inside += 1
    
    if part_inside > 0.:
        return np.sqrt(sigma_05/3.0/part_inside)
    else:
        return 0.

@njit
def sigma_projections(grid, n_cell, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, cx, cy, cz, R05x, R05y, R05z):
    Npart = len(part_list)
    quantas_x = np.ones((n_cell, n_cell), dtype = np.int32) #CUANTAS PARTÍCULAS EN CADA CELDA
    quantas_y = np.ones((n_cell, n_cell), dtype = np.int32)
    quantas_z = np.ones((n_cell, n_cell), dtype = np.int32)
    VCM_x = np.zeros((n_cell, n_cell)) #VELOCIDAD DEL CENTRO DE MASAS DE CADA CELDA EN LA DIRECCIÓN DE VISIÓN 
    VCM_y = np.zeros((n_cell, n_cell))
    VCM_z= np.zeros((n_cell, n_cell))
    SD_x = np.ones((n_cell, n_cell)) #DENSIDAD SUPERFICIAL EN MASA
    SD_y = np.ones((n_cell, n_cell))
    SD_z = np.ones((n_cell, n_cell))
    for ip in range(Npart):
        ipp = part_list[ip]
        ix = np.argmin(np.abs(grid - (st_x[ipp]-cx)))
        iy = np.argmin(np.abs(grid - (st_y[ipp]-cy)))
        iz = np.argmin(np.abs(grid - (st_z[ipp]-cz)))
        VCM_x[iy, iz] += st_vx[ipp]*st_mass[ipp]
        VCM_y[ix, iz] += st_vy[ipp]*st_mass[ipp]
        VCM_z[ix, iy] += st_vz[ipp]*st_mass[ipp]
        SD_x[iy, iz] += st_mass[ipp]
        SD_y[ix, iz] += st_mass[ipp]
        SD_z[ix, iy] += st_mass[ipp]
        quantas_x[iy, iz] += 1
        quantas_y[ix, iz] += 1
        quantas_z[ix, iy] += 1

    VCM_x /= SD_x
    VCM_y /= SD_y
    VCM_z /= SD_z


    SIG_1D_x = np.zeros((n_cell, n_cell)) #DISPERISIÓN DE VELOCIDADES EN CADA CELDA
    SIG_1D_y = np.zeros((n_cell, n_cell))
    SIG_1D_z = np.zeros((n_cell, n_cell))
    for ip in range(Npart):
        ipp = part_list[ip]
        ix = np.argmin(np.abs(grid - (st_x[ipp]-cx)))
        iy = np.argmin(np.abs(grid - (st_y[ipp]-cy)))
        iz = np.argmin(np.abs(grid - (st_z[ipp]-cz)))
        SIG_1D_x[iy, iz] += (VCM_x[iy, iz]-st_vx[ipp])**2
        SIG_1D_y[ix, iz] += (VCM_y[ix, iz]-st_vy[ipp])**2
        SIG_1D_z[ix, iy] += (VCM_z[ix, iy]-st_vz[ipp])**2

    SIG_1D_x = np.sqrt(SIG_1D_x/quantas_x)
    SIG_1D_y = np.sqrt(SIG_1D_y/quantas_y)
    SIG_1D_z = np.sqrt(SIG_1D_z/quantas_z)



    SIG_1D_x_05 = 0.
    counter_x = 0.

    SIG_1D_y_05 = 0.
    counter_y = 0.

    SIG_1D_z_05 = 0.
    counter_z = 0.

    for ip in range(Npart):
        ipp = part_list[ip]
        dx = cx - st_x[ipp]
        dy = cy - st_y[ipp]
        dz = cz - st_z[ipp]
        ix = np.argmin(np.abs(grid - dx))
        iy = np.argmin(np.abs(grid - dy))
        iz = np.argmin(np.abs(grid - dz))
        dist_x = (dy**2 + dz**2)**0.5
        dist_y = (dx**2 + dz**2)**0.5
        dist_z = (dx**2 + dy**2)**0.5

        if dist_x < R05x:
            SIG_1D_x_05 += SIG_1D_x[iy, iz]
            counter_x += 1

        if dist_y < R05y:
            SIG_1D_y_05 += SIG_1D_x[ix, iz]
            counter_y += 1

        if dist_z < R05z:
            SIG_1D_z_05 += SIG_1D_z[ix, iy]
            counter_z += 1

    ###########################################
    #V/sigma and lambda part (Fast-Slow rotator)
    ###########################################
    sumVz = 0.
    sumSigmaz = 0.
    sumup_z = 0.
    sumdown_z = 0.
    for ix in range(n_cell):
        for iy in range(n_cell):
            Rbin = (grid[ix]**2 + grid[iy]**2)**0.5
            if Rbin < R05z + 2*ll: #Tolerance of 1 cell
                #vsigma
                sumVz += VCM_z[ix, iy]**2 * SD_z[ix, iy]
                sumSigmaz += SIG_1D_z[ix, iy]**2 * SD_z[ix, iy]
                #lambda
                sumup_z += SD_z[ix, iy] * Rbin * abs(VCM_z[ix, iy])
                sumdown_z += SD_z[ix, iy] * Rbin * (VCM_z[ix, iy]**2 + SIG_1D_z[ix, iy]**2)**0.5

    V_sigma_z = (sumVz/sumSigmaz)**0.5
    lambda_z = sumup_z/sumdown_z

    sumVy = 0.
    sumSigmay = 0.
    sumup_y = 0.
    sumdown_y = 0.
    for ix in range(n_cell):
        for iz in range(n_cell):
            Rbin = (grid[ix]**2 + grid[iz]**2)**0.5
            if Rbin < R05y + 2*ll: #Tolerance of 1 cell
                #vsigma
                sumVy += VCM_y[ix, iz]**2 * SD_y[ix, iz]
                sumSigmay += SIG_1D_y[ix, iz]**2 * SD_y[ix, iz]
                #lambda
                sumup_y += SD_y[ix, iz] * Rbin * abs(VCM_y[ix, iz])
                sumdown_y += SD_y[ix, iz] * Rbin * (VCM_y[ix, iz]**2 + SIG_1D_y[ix, iz]**2)**0.5

    V_sigma_y = (sumVy/sumSigmay)**0.5
    lambda_y = sumup_y/sumdown_y

    sumVx = 0.
    sumSigmax = 0.
    sumup_x = 0.
    sumdown_x = 0.
    for iy in range(n_cell):
        for iz in range(n_cell):
            Rbin = (grid[iy]**2 + grid[iz]**2)**0.5
            if Rbin < R05x + 2*ll: #Tolerance of 1 cell
                #vsigma
                sumVx += VCM_x[iy, iz]**2 * SD_x[iy, iz]
                sumSigmax += SIG_1D_x[iy, iz]**2 * SD_x[iy, iz]
                #lambda
                sumup_x += SD_x[iy, iz] * Rbin * abs(VCM_x[iy, iz])
                sumdown_x += SD_x[iy, iz] * Rbin * (VCM_x[iy, iz]**2 + SIG_1D_x[iy, iz]**2)**0.5

    V_sigma_x = (sumVx/sumSigmax)**0.5
    lambda_x = sumup_x/sumdown_x

    #AVERAGE
    V_sigma = (V_sigma_x + V_sigma_y + V_sigma_z)/3.
    lambda_ = (lambda_x + lambda_y + lambda_z)/3.

    ###########################################
    # RETURN
    ###########################################
    if counter_x > 0 and counter_y > 0 and counter_z > 0:
        return SIG_1D_x_05/counter_x, SIG_1D_y_05/counter_y, SIG_1D_z_05/counter_z, V_sigma, lambda_
    else:
        return 0., 0., 0., 0., 0.
    

@njit(parallel = True)
def avg_age_metallicity(part_list, st_age, st_met, st_mass, cosmo_time):
    mass = 0.
    avg_age = 0.
    avg_age_mass = 0.
    avg_met = 0.
    avg_met_mass = 0.
    Npart = len(part_list)
    for ip in prange(Npart):
        ipp = part_list[ip]
        avg_age += (cosmo_time - st_age[ipp])
        avg_age_mass += (cosmo_time - st_age[ipp])*st_mass[ipp]
        avg_met += st_met[ipp]
        avg_met_mass += st_met[ipp]*st_mass[ipp]
        mass += st_mass[ipp]
    
    return avg_age/Npart, avg_age_mass/mass, avg_met/Npart, avg_met_mass/mass


def photutils_fit(R05, R05x, R05y, R05z, R_fit_min, R_fit_max, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz):
    ###################################################################################
    #PHOTUTILS IS A PACKAGE THAT ALLOWS TO FIT ISOPHOTES TO A 2D IMAGE
    #IT IS BASED ON THE ALGORITHM OF JEDRZEJEWSKI (1987) AND BENDINELLI ET AL. (1990)
    #PROBLEM: CALCULATIONS DONE IN PYTHON, SO IT IS SLOW (BUT IT WORKS)
    ###################################################################################

    #FIRST, BUILD AN ELLIPSE MODEL
    #x0, y0 --> CENTRE OF THE GALAXY in pixels
    #sma --> semi-major axis in pixels
    #eps --> ellipticity
    #pa --> position angle of sma (in radians) relative to the x axis

    sma = R05/res #FROM mpc to pixels
    sma_xy = int(R05z/res)
    sma_xz = int(R05y/res)
    sma_yz = int(R05x/res)

    sma_fit_min = R_fit_min/res
    sma_fit_max = R_fit_max/res
    sma_fit_0 = (sma_fit_min + sma_fit_max)/2.

    argmax_xy = np.argmax(star_density_2D_xy)
    x0_xy = np.unravel_index(argmax_xy, star_density_2D_xy.shape)[0]
    y0_xy = np.unravel_index(argmax_xy, star_density_2D_xy.shape)[1]
    geometry_xy = EllipseGeometry(x0 = x0_xy, y0 = y0_xy, sma = sma, eps = 0.3, pa = 45 / 180 * np.pi)

    argmax_xz = np.argmax(star_density_2D_xz)
    x0_xz = np.unravel_index(argmax_xz, star_density_2D_xz.shape)[0]
    y0_xz = np.unravel_index(argmax_xz, star_density_2D_xz.shape)[1]
    geometry_xz = EllipseGeometry(x0 = x0_xz, y0 = y0_xz, sma = sma, eps = 0.3, pa = 45 / 180 * np.pi)

    argmax_yz = np.argmax(star_density_2D_yz)
    x0_yz = np.unravel_index(argmax_yz, star_density_2D_yz.shape)[0]
    y0_yz = np.unravel_index(argmax_yz, star_density_2D_yz.shape)[1]
    geometry_yz = EllipseGeometry(x0 = x0_yz, y0 = y0_yz, sma = sma, eps = 0.3, pa = 45 / 180 * np.pi)
    
    # PLOT TO CHECK
    # aper = EllipticalAperture((geometry_xy.x0, geometry_xy.y0), geometry_xy.sma,
    #                         geometry_xy.sma * (1 - geometry_xy.eps),
    #                         geometry_xy.pa)
    # plt.imshow(star_density_2D_xy.T, origin='lower', cmap='viridis')
    # aper.plot(color='white')
    # plt.show()

    #NOW FIT ---> MOST EXPENSIVE PART, THE MORE PIXELS, THE MORE TIME SPENT
    #SERSIC PROFILE
    def sersic(R, Re, Ie, n):
        bn = 2*n - 1/3 + 4/(405*n)
        return Ie*np.exp( -bn*( (R/Re)**(1/n) - 1 ) )
    
    minpoints = 10 # minimum number of points to fit
    #XY PLANE
    try:
        ellipse_xy = Ellipse(star_density_2D_xy, geometry_xy)
        isolist_xy = ellipse_xy.fit_image(minsma=sma_fit_min, maxsma=sma_fit_max, sma0 = sma_fit_0)
        ellipticity_list_xy = isolist_xy.eps
        intens_list_xy = isolist_xy.intens
        sma_list_xy = isolist_xy.sma

        #PLOT TO CHECK
        # model_image = build_ellipse_model(star_density_2D_xy.shape, isolist_xy)
        # plt.imshow(model_image.T)
        # plt.show

        if len(intens_list_xy) >= minpoints:
            #FIT THE SERSIC INDEX
            sma_list_xy = np.array(sma_list_xy)
            intens_list_xy = np.array(intens_list_xy)
            ellipticity_list_xy = np.array(ellipticity_list_xy)
            #NOW, FIT
            guess_xy = [sma, np.max(intens_list_xy), 1.]
            param_xy, _ = curve_fit(sersic, sma_list_xy, intens_list_xy, p0 = guess_xy)
            n_xy = param_xy[2]
            #WE SELECT THE ELLIPTICITY OF THE CLOSEST ISOPHOTE TO THE HALF LIGHT RADIUS
            eps_xy = ellipticity_list_xy[np.argmin(np.abs(sma_list_xy - sma_xy))]

            # CHECK FIT
            # plt.plot(sma_list_xy, intens_list_xy, 'o')
            # plt.plot(sma_list_xy, sersic(sma_list_xy, param_xy[0], param_xy[1], param_xy[2]))
            # plt.show()
        else:
            n_xy = np.nan
            eps_xy = np.nan

    except:
        n_xy = np.nan
        eps_xy = np.nan

    #XZ PLANE
    try:
        ellipse_xz = Ellipse(star_density_2D_xz, geometry_xz)
        isolist_xz = ellipse_xz.fit_image(minsma=sma_fit_min, maxsma=sma_fit_max, sma0 = sma_fit_0)
        ellipticity_list_xz = isolist_xz.eps
        intens_list_xz = isolist_xz.intens
        sma_list_xz = isolist_xz.sma

        #PLOT TO CHECK
        # model_image = build_ellipse_model(star_density_2D_xz.shape, isolist_xz)
        # plt.imshow(model_image.T)
        # plt.show

        if len(intens_list_xz) >= minpoints:
            #FIT THE SERSIC INDEX
            sma_list_xz = np.array(sma_list_xz)
            intens_list_xz = np.array(intens_list_xz)
            ellipticity_list_xz = np.array(ellipticity_list_xz)
            #NOW, FIT
            guess_xz = [sma, np.max(intens_list_xz), 1.]
            param_xz, _ = curve_fit(sersic, sma_list_xz, intens_list_xz, p0 = guess_xz)
            n_xz = param_xz[2]
            #WE SELECT THE ELLIPTICITY OF THE CLOSEST ISOPHOTE TO THE HALF LIGHT RADIUS
            eps_xz = ellipticity_list_xz[np.argmin(np.abs(sma_list_xz - sma_xz))]

        else:
            n_xz = np.nan
            eps_xz = np.nan

    except:
        n_xz = np.nan
        eps_xz = np.nan

    #YZ PLANE
    try:
        ellipse_yz = Ellipse(star_density_2D_yz, geometry_yz)
        isolist_yz = ellipse_yz.fit_image(minsma=sma_fit_min, maxsma=sma_fit_max, sma0 = sma_fit_0)
        ellipticity_list_yz = isolist_yz.eps
        intens_list_yz = isolist_yz.intens
        sma_list_yz = isolist_yz.sma
        
        #PLOT TO CHECK
        # model_image = build_ellipse_model(star_density_2D_yz.shape, isolist_yz)
        # plt.imshow(model_image.T)
        # plt.show

        if len(intens_list_yz) >= minpoints:
            #FIT THE SERSIC INDEX
            sma_list_yz = np.array(sma_list_yz)
            intens_list_yz = np.array(intens_list_yz)
            ellipticity_list_yz = np.array(ellipticity_list_yz)
            #NOW, FIT
            guess_yz = [sma, np.max(intens_list_yz), 1.]
            param_yz, _ = curve_fit(sersic, sma_list_yz, intens_list_yz, p0 = guess_yz)
            n_yz = param_yz[2]
            #WE SELECT THE ELLIPTICITY OF THE CLOSEST ISOPHOTE TO THE HALF LIGHT RADIUS
            eps_yz = ellipticity_list_yz[np.argmin(np.abs(sma_list_yz - sma_yz))]

        else:
            n_yz = np.nan
            eps_yz = np.nan

    except:
        n_yz = np.nan
        eps_yz = np.nan

    #NOW, WE COMPUTE THE AVERAGE SERSIC INDEX AND ELLIPTICITY
    n = np.nanmean([n_xy, n_xz, n_yz]) #average ignoring nans
    eps = np.nanmean([eps_xy, eps_xz, eps_yz]) #average ignoring nans

    return n, eps


def pyimfit_fit(R05x, R05y, R05z, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz):
    ########################################################################################################
    # PyImfit is a Python wrapper around the (C++-based) image-fitting program Imfit (Erwin 2015)
    #
    # Calculations are done in C++, thus it is much faster than photutils, specially for 
    # images with a low number of pixels like the ones from cosmological simulations (e.g. ~128x128 at most)
    ########################################################################################################

    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!      WARNING         !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Imfit was written to follow the standard 2D array indexing conventions of FITS, IRAF, and (e.g.) 
    # SAOimage DS9, which are 1-based and column-major. This means that the center of the first pixel (in the lower-left of the image) 
    # has coordinates (x,y) = (1.0,1.0); the lower-left corner of that pixel has coordinates (0.5,0.5), 
    # and the upper-right corner of the same pixel is at (1.5,1.5). The first coordinate (“x”) is the column number; 
    # the second (“y”) is the row number.

    # To allow one to use Imfit configuration files with PyImfit, PyImfit adopts the same column-major, 1-based indexing standard. 
    # The most obvious way this shows up is in the X0,Y0 coordinates for the centers of function sets.

    #Python (and in particular NumPy), on the other hand, is 0-based and row-major. This means that the first pixel 
    # in the image is at (0,0); it also means that the first index is the row number.

    #To translate coordinate systems, remember that a pixel with Imfit/PyImfit coordinates 
    # x,y would be found in a NumPy array at array[y0 - 1, x0 - 1].
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    #CREATE A GALAXY MODEL FOR EACH PLANE

    #x0, y0 --> CENTRE OF THE GALAXY in pixels
    #sma --> semi-major axis in pixels
    #eps --> ellipticity
    #pa --> position angle of sma (in radians) relative to the x axis

    def galaxy_model(x0, y0, pa, eps, n, I_e, r_e, Imax):
        model = pyimfit.SimpleModelDescription()
        model.x0.setValue(x0, [x0-10, x0+10]) #x0 and y0 are the centre of the galaxy
        model.y0.setValue(y0, [y0-10, y0+10]) #It is important not to give too much freedom to the centre of the galaxy !!!!
        galaxy_profile = pyimfit.make_imfit_function('Sersic') 
        galaxy_profile.PA.setValue(pa, [0, 180]) #position angle of the semi-major axis
        galaxy_profile.ell.setValue(eps, [0, 1]) #ellipticity
        galaxy_profile.n.setValue(n, [0.5, 5]) #sersic index
        galaxy_profile.I_e.setValue(I_e, [0., Imax]) #surface brightness at the effective radius
        galaxy_profile.r_e.setValue(r_e, [0.5*r_e, 1.5*r_e]) #effective radius
        model.addFunction(galaxy_profile)
        return model

    sma_xy = int(R05z/res)
    sma_xz = int(R05y/res)
    sma_yz = int(R05x/res)

    #xy plane
    argmax_xy = np.argmax(star_density_2D_xy)
    x0_xy = np.unravel_index(argmax_xy, star_density_2D_xy.shape)[1] + 1
    y0_xy = np.unravel_index(argmax_xy, star_density_2D_xy.shape)[0] + 1
    I_e_xy = 0.01*np.max(star_density_2D_xy)
    model_desc_xy = galaxy_model(x0 = x0_xy, y0 = y0_xy, pa = 95., eps = 0.3, 
                                    n = 1., I_e = I_e_xy, r_e = sma_xy,
                                    Imax = np.max(star_density_2D_xy))
    
    #xz plane
    argmax_xz = np.argmax(star_density_2D_xz)
    x0_xz = np.unravel_index(argmax_xz, star_density_2D_xz.shape)[1] + 1
    y0_xz = np.unravel_index(argmax_xz, star_density_2D_xz.shape)[0] + 1
    I_e_xz = 0.01*np.max(star_density_2D_xz)
    model_desc_xz = galaxy_model(x0 = x0_xz, y0 = y0_xz, pa = 95., eps = 0.3,
                                    n = 1., I_e = I_e_xz, r_e = sma_xz,
                                    Imax = np.max(star_density_2D_xz))    
    #yz plane
    argmax_yz = np.argmax(star_density_2D_yz)
    x0_yz = np.unravel_index(argmax_yz, star_density_2D_yz.shape)[1] + 1
    y0_yz = np.unravel_index(argmax_yz, star_density_2D_yz.shape)[0] + 1
    I_e_yz = 0.01*np.max(star_density_2D_yz)
    model_desc_yz = galaxy_model(x0 = x0_yz, y0 = y0_yz, pa = 95., eps = 0.3,
                                    n = 1., I_e = I_e_yz, r_e = sma_yz,
                                    Imax = np.max(star_density_2D_yz))
    
    #CREATE THE FITTER OBJECTS
    imfitter_xy = pyimfit.Imfit(model_desc_xy)
    imfitter_xz = pyimfit.Imfit(model_desc_xz)
    imfitter_yz = pyimfit.Imfit(model_desc_yz)

    #ELIMINATE ZEROES (NOT ALLOWED BY IMFIT), REPLACE THEM WITH NANs, WHICH ARE IGNORED BY IMFIT
    star_density_2D_xy_flat = star_density_2D_xy.flatten()
    star_density_2D_xy_flat[star_density_2D_xy_flat == 0] = np.nan
    star_density_2D_xy = star_density_2D_xy_flat.reshape(star_density_2D_xy.shape)

    star_density_2D_xz_flat = star_density_2D_xz.flatten()
    star_density_2D_xz_flat[star_density_2D_xz_flat == 0] = np.nan
    star_density_2D_xz = star_density_2D_xz_flat.reshape(star_density_2D_xz.shape)

    star_density_2D_yz_flat = star_density_2D_yz.flatten()
    star_density_2D_yz_flat[star_density_2D_yz_flat == 0] = np.nan
    star_density_2D_yz = star_density_2D_yz_flat.reshape(star_density_2D_yz.shape)

    #LOAD THE DATA
    imfitter_xy.loadData(star_density_2D_xy)
    imfitter_xz.loadData(star_density_2D_xz)
    imfitter_yz.loadData(star_density_2D_yz)

    #FIT
    solver = 'LM' #'LM' (faster but can be trapped in local minimum), 'NM' (Slower, but more robust)
    
    try:
        result_xy = imfitter_xy.doFit(solver = solver)
        converged_xy = result_xy.fitConverged
        bestfit_parameters_xy = result_xy.params
    except:
        converged_xy = False
        bestfit_parameters_xy = np.nan

    try:
        result_xz = imfitter_xz.doFit(solver = solver)
        converged_xz = result_xz.fitConverged
        bestfit_parameters_xz = result_xz.params
    except:
        converged_xz = False
        bestfit_parameters_xz = np.nan

    try:
        result_yz = imfitter_yz.doFit(solver = solver)
        converged_yz = result_yz.fitConverged
        bestfit_parameters_yz = result_yz.params
    except:
        converged_yz = False
        bestfit_parameters_yz = np.nan

    # GET SERSIC INDEX AND ELLIPTICITY
    # --> ORDER OF PARAMETERS (Sérsic profile): x0, y0, PA, ell, n, I_e, r_e
    if converged_xy:
        n_xy = bestfit_parameters_xy[4]
        eps_xy = bestfit_parameters_xy[3]
    else:
        n_xy = np.nan
        eps_xy = np.nan

    if converged_xz:
        n_xz = bestfit_parameters_xz[4]
        eps_xz = bestfit_parameters_xz[3]
    else:
        n_xz = np.nan
        eps_xz = np.nan

    if converged_yz:
        n_yz = bestfit_parameters_yz[4]
        eps_yz = bestfit_parameters_yz[3]
    else:
        n_yz = np.nan
        eps_yz = np.nan

    # PLOT
    # bestfit_model_im_xy = imfitter_xy.getModelImage() #2D array
    # bestfit_model_im_xz = imfitter_xz.getModelImage()
    # bestfit_model_im_yz = imfitter_yz.getModelImage()
    # Imshow
    # fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    # ax[0].imshow(bestfit_model_im_xy.T)
    # ax[1].imshow(bestfit_model_im_xz.T)
    # ax[2].imshow(bestfit_model_im_yz.T)
    # plt.show()

    #NOW, WE COMPUTE THE AVERAGE SERSIC INDEX AND ELLIPTICITY
    n = np.nanmean([n_xy, n_xz, n_yz]) #average ignoring nans
    eps = np.nanmean([eps_xy, eps_xz, eps_yz]) #average ignoring nans

    return n, eps



def sersic_fit(R05, R05x, R05y, R05z, R_fit_min, R_fit_max, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz, fit_mode = 'photutils'):
    #ERROR CONTROL    
    fit_modes = ['photutils', 'pyimfit']
    if fit_mode not in fit_modes:
        raise ValueError('mode must be one of %r.' % fit_modes)
    ###########################################################

    #FIT
    if fit_mode == 'photutils':
        n, eps = photutils_fit(R05, R05x, R05y, R05z, R_fit_min, R_fit_max, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz)

    # NOT WORKING YET
    if fit_mode == 'pyimfit':
        n, eps = pyimfit_fit(R05x, R05y, R05z, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz)
        
    return n, eps

def sersic_index(part_list, st_x, st_y, st_z, cx, cy, cz, R05, R05x, R05y, R05z):
    # In order to capture correctyle the slope of the profile, we need to fit between
    # the right radius. This is because the profile s not a perfect sersic, and it 
    # has a bump (or flattening) at the center and in the outer regions the
    # profile is not well defined.

    R_fit_min = 0.
    R_fit_max = 1.5*R05

    #ONLY CONSIDER PARTICLES WITHIN R_sersic
    x_pos = st_x[part_list]
    y_pos = st_y[part_list]
    z_pos = st_z[part_list]
    R_pos = ((x_pos-cx)**2 + (y_pos-cy)**2 + (z_pos-cz)**2)**0.5

    part_list = part_list[R_pos < R_fit_max]
    x_pos = x_pos[R_pos < R_fit_max]
    y_pos = y_pos[R_pos < R_fit_max]
    z_pos = z_pos[R_pos < R_fit_max]


    # NOW CONVERT TO POSITIONS BETWEEN 0, 2*R_sersic, f4(float32)
    x_pos = np.float32(x_pos - cx + R_fit_max) # kpc
    y_pos = np.float32(y_pos - cy + R_fit_max)
    z_pos = np.float32(z_pos - cz + R_fit_max)

    #DEFINING THE GRID
    partNum = np.int32(len(part_list))
    L_box = np.float32(2*R_fit_max) #kpc
    res = np.float32(      ll     )  # IMPORTANT!!!! RESOLUTION OF THE GRID FOR THE SERSIC INDEX
                                       # IT SHOULD BE -->LL<--, BUT SMALL GALAXIES ARE A PROBLEM
                                       # BIGGEST GALAXIES ARE NOT A PROBLEM, SINCE THERE IS A MAXIMUM NUMBER OF CELLS
                                       # SEE BELOW
    ncell = np.int32(min( max(L_box/res, 32), 64) )
    res = np.float32(L_box/ncell)   # RECALCULATE RES IN CASE I CHANGED IT
    kneigh = np.int32(16) # h distance in SPH kernel is calculated as the distance to the "kneigh" nearest neighbour
                          # the higher the kneigh value, the more time it will take
    # CALL FORTRAN 
    field = np.ones(partNum, dtype = np.float32) #CONSIDER ALL PARTICLES AS EQUALLY MASSIVE
    star_density_3D, _ = sph3D.sph.main(x_pos, y_pos, z_pos, L_box, L_box, L_box, field, kneigh, ncell, ncell, ncell, partNum)
    
    #NOW, SURFACE DENSITY IN EACH PLANE
    star_density_2D_xy = np.mean(star_density_3D, axis = 2)
    star_density_2D_xz = np.mean(star_density_3D, axis = 1)
    star_density_2D_yz = np.mean(star_density_3D, axis = 0)


    #EXTRAPOLATION TO GET A BETTER RESOLUTION, with scipy.regular_grid_interpolator
    # grid_faces = np.linspace(0, L_box, ncell+1)
    # grid_centers = (grid_faces[1:] + grid_faces[:-1])/2.

    # interp_xy = RegularGridInterpolator((grid_centers, grid_centers), star_density_2D_xy, bounds_error=False, fill_value=None, method = 'linear')
    # interp_xz = RegularGridInterpolator((grid_centers, grid_centers), star_density_2D_xz, bounds_error=False, fill_value=None, method = 'linear')
    # interp_yz = RegularGridInterpolator((grid_centers, grid_centers), star_density_2D_yz, bounds_error=False, fill_value=None, method = 'linear')

    # n_extrapolate = 256
    # res = L_box/n_extrapolate
    # grid_faces_finner = np.linspace(0, L_box, n_extrapolate+1)
    # grid_centers_finner = (grid_faces_finner[1:] + grid_faces_finner[:-1])/2.
    # X, Y = np.meshgrid(grid_centers_finner, grid_centers_finner)

    # star_density_2D_xy = interp_xy((X, Y))
    # star_density_2D_xz = interp_xz((X, Y))
    # star_density_2D_yz = interp_yz((X, Y))

    # APPLYING A GAUSSIAN FILTER TO SMOOTH THE SURFACE DENSITY
    # sfilter = 2.
    # star_density_2D_xy = gaussian_filter(star_density_2D_xy, sigma = sfilter)
    # star_density_2D_xz = gaussian_filter(star_density_2D_xz, sigma = sfilter)
    # star_density_2D_yz = gaussian_filter(star_density_2D_yz, sigma = sfilter)

    # PLOT
    # fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    # ax[0].imshow(star_density_2D_xy.T, origin = 'lower')
    # ax[1].imshow(star_density_2D_xz.T, origin = 'lower')
    # ax[2].imshow(star_density_2D_yz.T, origin = 'lower')
    # plt.show()

    #FITTING THE SERSIC PROFILE
    n, eps = sersic_fit(R05, R05x, R05y, R05z, R_fit_min, R_fit_max, res, star_density_2D_xy, star_density_2D_xz, star_density_2D_yz, fit_mode = 'photutils')

    return n, eps


def write_to_HALMA_catalogue(total_iteration_data, total_halo_data, name = 'halma_catalogue_from_PYFOF.res'):
    catalogue = open(name, 'w')
    num_iter = len(total_iteration_data)
    catalogue.write(str(num_iter))
    catalogue.write('\n')
    gap = ''
    for it_halma in range(num_iter):
        catalogue.write('===========================================================================================================')
        catalogue.write('===========================================================================================================')
        catalogue.write('===========================================================================================================')
        catalogue.write('===========================================================================================================')
        catalogue.write('\n')
        it_values = [*total_iteration_data[it_halma].values()]
        for value in it_values:
            catalogue.write(str(value)+'          ')

        
        first_strings = ['Halo','n','Mass','Mass','Mass','frac', 'm_rps','m_rps','m_SFR', 
                         ' R ','R_05','R_05','R_05','R_05','R_05', 'sigma','sigma','sig_x',
                         'sig_y','sig_z','j', 'c_x','c_y','c_z', 'V_x','V_y','V_z','Pro.','Pro.',
                         'n','type', 'age','age', 'Z','Z', 'V/Sigma', 'lambda', 'v_TF', 'sersic', '1-b/a']
        
        second_strings = ['ID',' part', ' * ','*_vis','gas','g_cold',  'cold','hot','  * ', 'max','3D',
                        '1D','1D_x','1D_y','1D_z', '05_3D','05_1D','05_1D','05_1D','05_1D',
                        '  ', 'kpc','kpc','kpc', 'km/s','km/s','km/s',
                        '(1)','(2)','merg','merg','m_weig','mean', 'm_weig','mean', '  ', '  ', 'km/s', '  ', '  ']

        first_line = f'{first_strings[0]:6s}{first_strings[1]:10s}{first_strings[2]:15s}{first_strings[3]:15s}\
{first_strings[4]:15s}{first_strings[5]:8s}{first_strings[6]:15s}{first_strings[7]:15s}{first_strings[8]:15s}\
{first_strings[9]:10s}{first_strings[10]:10s}{first_strings[11]:10s}{first_strings[12]:10s}{first_strings[13]:10s}{first_strings[14]:10s}\
{first_strings[15]:10s}{first_strings[16]:10s}{first_strings[17]:10s}{first_strings[18]:10s}{first_strings[19]:10s}{first_strings[20]:10s}\
{first_strings[21]:10s}{first_strings[22]:10s}{first_strings[23]:10s}{first_strings[24]:10s}{first_strings[25]:10s}{first_strings[26]:10s}\
{first_strings[27]:6s}{first_strings[28]:6s}{first_strings[29]:6s}{first_strings[30]:6s}{first_strings[31]:9s}{first_strings[32]:9s}\
{first_strings[33]:11s}{first_strings[34]:11s}{first_strings[35]:11s}{first_strings[36]:11s}{first_strings[37]:11s}{first_strings[38]:11s}{first_strings[39]:11s}'
        
        second_line = f'{second_strings[0]:6s}{second_strings[1]:10s}{second_strings[2]:15s}{second_strings[3]:15s}\
{second_strings[4]:15s}{second_strings[5]:8s}{second_strings[6]:15s}{second_strings[7]:15s}{second_strings[8]:15s}\
{second_strings[9]:10s}{second_strings[10]:10s}{second_strings[11]:10s}{second_strings[12]:10s}{second_strings[13]:10s}{second_strings[14]:10s}\
{second_strings[15]:10s}{second_strings[16]:10s}{second_strings[17]:10s}{second_strings[18]:10s}{second_strings[19]:10s}{second_strings[20]:10s}\
{second_strings[21]:10s}{second_strings[22]:10s}{second_strings[23]:10s}{second_strings[24]:10s}{second_strings[25]:10s}{second_strings[26]:10s}\
{second_strings[27]:6s}{second_strings[28]:6s}{second_strings[29]:6s}{second_strings[30]:6s}{second_strings[31]:9s}{second_strings[32]:9s}\
{second_strings[33]:11s}{second_strings[34]:11s}{second_strings[35]:11s}{second_strings[36]:11s}{second_strings[37]:11s}{second_strings[38]:11s}{second_strings[39]:11s}'
        
        catalogue.write('\n')
        catalogue.write('------------------------------------------------------------------------------------------------------------')
        catalogue.write('------------------------------------------------------------------------------------------------------------')
        catalogue.write('------------------------------------------------------------------------------------------------------------')
        catalogue.write('------------------------------------------------------------------------------------------------------------')
        catalogue.write('\n')
        catalogue.write('      '+first_line)
        catalogue.write('\n')
        catalogue.write('      '+second_line)
        catalogue.write('\n')
        catalogue.write('------------------------------------------------------------------------------------------------------------')
        catalogue.write('------------------------------------------------------------------------------------------------------------')
        catalogue.write('------------------------------------------------------------------------------------------------------------')
        catalogue.write('------------------------------------------------------------------------------------------------------------')
        catalogue.write('\n')
        nhal = total_iteration_data[it_halma]['nhal']
        for ih in range(nhal):
            ih_values = [*total_halo_data[it_halma][ih].values()]
            catalogue_line = f'{ih_values[0]:6d}{gap}{ih_values[1]:10d}{gap}{ih_values[2]:15.6e}{gap}\
{ih_values[3]:15.6e}{gap}{ih_values[4]:15.6e}{gap}{ih_values[5]:8.4f}{gap}\
{ih_values[6]:15.6e}{gap}{ih_values[7]:15.6e}{gap}{ih_values[8]:15.6e}{gap}\
{ih_values[9]:10.2f}{gap}{ih_values[10]:10.2f}{gap}{ih_values[11]:10.2f}{gap}\
{ih_values[12]:10.2f}{gap}{ih_values[13]:10.2f}{gap}{ih_values[14]:10.2f}{gap}\
{ih_values[15]:10.2f}{gap}{ih_values[16]:10.2f}{gap}{ih_values[17]:10.2f}{gap}\
{ih_values[18]:10.2f}{gap}{ih_values[19]:10.2f}{gap}{ih_values[20]:10.2f}{gap}\
{ih_values[21]:10.2f}{gap}{ih_values[22]:10.2f}{gap}{ih_values[23]:10.2f}{gap}\
{ih_values[24]:10.2f}{gap}{ih_values[25]:10.2f}{gap}{ih_values[26]:10.2f}{gap}\
{ih_values[27]:6d}{gap}{ih_values[28]:6d}{gap}{ih_values[29]:6d}{gap}{ih_values[30]:6d}{gap}\
{ih_values[31]:9.3f}{gap}{ih_values[32]:9.3f}{gap}{ih_values[33]:11.3e}{gap}{ih_values[34]:11.3e}{gap}\
{ih_values[35]:11.2f}{gap}{ih_values[36]:11.2f}{gap}{ih_values[37]:11.2f}{gap}{ih_values[38]:11.2f}{gap}{ih_values[39]:11.2f}{gap}'
            
            catalogue.write(catalogue_line)
            catalogue.write('\n')
    
    catalogue.close()

########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 






########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
# M             A           I           N   ##########
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 


re0 = 1.0  #factor de escala a z = 0

print('****************************************************')
print('******************** MASCLET pyfof *****************')
print('****************************************************')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('CHECK!!!! --> LINKING LENGHT (kpc)', ll*1e3)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print()
print('----> Using', ncore,'CPU threads')

###############################
set_num_threads(ncore)
###############################

total_iteration_data = []
total_halo_data = []

#loop over iterations
oripas_before = []
OMM = [] #HALOS PREVIOUS ITERATION
for it_count, iteration in enumerate(range(first, last+step, step)):
    print()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('             Iteration', iteration)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Opening MASCLET files')
    print()
    #open MASCLET files
    
    masclet_grid_data = read_masclet.read_grids(iteration, path=path_results, parameters_path=path_results, 
                                                digits=5, read_general=True, read_patchnum=False, read_dmpartnum=False,
                                                read_patchcellextension=False, read_patchcellposition=False, read_patchposition=False,
                                                read_patchparent=False)
    cosmo_time = masclet_grid_data[1]
    zeta = masclet_grid_data[4]
    rete = re0 / (1.0+zeta)
    
    dt = 0.
    if it_count>0:
        dt = (cosmo_time - total_iteration_data[it_count-1]['t'])*units.time_to_yr/1e9

    print('Cosmo time (Gyr):', cosmo_time*units.time_to_yr/1e9)
    print('Redshift (z):', zeta)

    # #READ GAS IF RPS
    # if rps_flag:
    #     print('RPS == True !!!! ')
    #     print('     Opening grid and clus files')

    #     grid_data = read_masclet.read_grids(iteration, path=path_results, parameters_path=path_results, digits=5, 
    #                                                 read_general=True, read_patchnum=True, read_dmpartnum=False,
    #                                                 read_patchcellextension=True, read_patchcellposition=True, read_patchposition=True,
    #                                                 read_patchparent=False)
    #     nl = grid_data[2]
    #     npatch = grid_data[5] #number of patches in each level, starting in l=0
    #     patchnx = grid_data[6] #patchnx (...): x-extension of each patch (in level l cells) (and Y and Z)
    #     patchny = grid_data[7]
    #     patchnz = grid_data[8]
    #     patchrx = grid_data[12] #patchrx (...): physical position of the center of each patch first ¡l-1! cell (and Y and Z)
    #     patchry = grid_data[13] # in Mpc
    #     patchrz = grid_data[14]

    #     gas_data = read_masclet.read_clus(iteration, path=path_results, parameters_path=path_results, digits=5, max_refined_level=1000, output_delta=True, 
    #                                       output_v=True, output_pres=False, output_pot=True, output_opot=False, output_temp=True, output_metalicity=False,
    #                                       output_cr0amr=True, output_solapst=True, is_mascletB=False, output_B=False, is_cooling=True, verbose=False)

    #     gas_delta = gas_data[0]
    #     gas_density = misctools.delta_to_rho(gas_delta)
    #     gas_cr0amr = gas_data[1]
    #     gas_solapst = gas_data[2]
    #     gas_vx = gas_data[3]*3e5 #in km/s
    #     gas_vy = gas_data[4]*3e5 #in km/s
    #     gas_vz = gas_data[5]*3e5 #in km/s
    #     gas_pot = gas_data[6]
    #     gas_temp = gas_data[7]

    masclet_st_data = read_masclet.read_clst(iteration, path = path_results, parameters_path=path_results, 
                                                    digits=5, max_refined_level=1000, 
                                                    output_deltastar=False, verbose=False, output_position=True, 
                                                    output_velocity=True, output_mass=True, output_time=True,
                                                    output_metalicity=True, output_id=True, are_BH = not old_masclet)

    st_x = masclet_st_data[0]
    st_y = masclet_st_data[1]
    st_z = masclet_st_data[2]
    st_vx = masclet_st_data[3]*3e5 #in km/s
    st_vy = masclet_st_data[4]*3e5 #in km/s
    st_vz = masclet_st_data[5]*3e5 #in km/s
    st_mass = masclet_st_data[6]*units.mass_to_sun #in Msun
    st_age = masclet_st_data[7]*units.time_to_yr/1e9 #in Gyr
    st_met = masclet_st_data[8]
    st_oripa = masclet_st_data[9] #necessary for mergers

    # if old_masclet:

    #     ###### IN 2017 SIMULATION FIX ORIPAS ############
    #     print('!!!!!!!!!!!!! Fixing ORIPA (2017 or older simulations)')
    #     st_oripa = particles.correct_positive_oripa(st_oripa, st_mass)
    #     ################################################

    data = np.vstack((st_x, st_y, st_z)).T
    data = data.astype(np.float64)
    if len(data) > 0: #HAY ESTRELLAS
        #APPLY FOF
        print()
        print('----------> FoF begins <--------')
        print()
        groups = pyfof.friends_of_friends(data = data, linking_length = ll)
        groups = np.array(groups, dtype=object)
        #CLEAN THOSE HALOES with npart < minp
        groups = groups[good_groups(groups)]
        print(len(groups),'haloes found')
        print()
        #CALCULATE CM, CM_vel AND phase-space cleaning
        print('---> Phase-space cleaning begins <---')
        CX = []
        CY = []
        CZ = []
        MM = []
        RMAX = []
        NPART = []
        new_groups = []
        for ihal in tqdm(range(len(groups))):
            part_list = np.array(groups[ihal])
            cx, cy, cz, M = center_of_mass(part_list, st_x, st_y, st_z, st_mass)
            M0 = M
            #Create 3D grid with cellsize 2*LL for cleaning
            RRHH = furthest_particle(cx, cy, cz, part_list, st_x, st_y, st_z)
            grid = np.arange(-(RRHH+ll), RRHH+ll, 2*ll)
            n_cell = len(grid)
            vcm_cell = np.zeros((n_cell, n_cell, n_cell, 3))
            mass_cell = np.zeros((n_cell, n_cell, n_cell))
            quantas_cell = np.zeros((n_cell, n_cell, n_cell))
            sig3D_cell = np.zeros((n_cell, n_cell, n_cell))
            #CALCULATE cell quantities
            vcm_cell, mass_cell, quantas_cell, sig3D_cell = calc_cell(cx, cy, cz, grid, part_list, st_x, st_y, st_z, st_vx, st_vy, 
                                                                    st_vz, st_mass, vcm_cell, mass_cell, quantas_cell, sig3D_cell)
            
            #DO THE CLEANING: Right CM, Mass and Furthest particle
            cx, cy, cz, M, RRHH, control = clean_cell(cx, cy, cz, M, RRHH, grid, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, vcm_cell, mass_cell, quantas_cell, sig3D_cell, ll, q_fil, sig_fil)
            #vx, vy, vz = CM_velocity(M, part_list[control.astype(bool)], st_vx, st_vy, st_vz, st_mass)
            #FASTER CLEANING --> ESCAPE VELOCITY IN ITS POSITION, CONSIDERING MASS WITHOUT CLEANING M0 and SPHERICAL SIMMETRY
            # factor_v = 4
            # control = escape_velocity_cleaning(cx, cy, cz, vx, vy, vz, M0, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, control, factor_v)
            #CLEANING DONE
            control = control.astype(bool)
            npart = len(part_list[control])
            cx, cy, cz, M = center_of_mass(part_list[control], st_x, st_y, st_z, st_mass)
            RRHH = furthest_particle(cx, cy, cz, part_list[control], st_x, st_y, st_z)
            if npart > minp:
                CX.append(cx)
                CY.append(cy)
                CZ.append(cz)
                MM.append(M)
                RMAX.append(RRHH)
                NPART.append(npart)
                new_groups.append(part_list[control])

        CX = np.array(CX)
        CY = np.array(CY)
        CZ = np.array(CZ)
        MM = np.array(MM)
        RMAX = np.array(RMAX)
        NPART = np.array(NPART).astype(np.int32)
        NHAL = len(new_groups)
        NPARTHAL = np.sum(NPART)

        print('Number of haloes after phase-space cleaning:', NHAL)
        print('Number of particles in haloes after cleaning:', NPARTHAL)
        print()
        print('Calculating properties')
        #CALCULATE HALO PROPERTIES
        RAD05 = np.zeros(NHAL)
        RAD05_x = np.zeros(NHAL)
        RAD05_y = np.zeros(NHAL)
        RAD05_z = np.zeros(NHAL)
        VX = np.zeros(NHAL)
        VY = np.zeros(NHAL)
        VZ = np.zeros(NHAL)
        JX = np.zeros(NHAL)
        JY = np.zeros(NHAL)
        JZ = np.zeros(NHAL)
        J = np.zeros(NHAL)
        PEAKX = np.zeros(NHAL)
        PEAKY = np.zeros(NHAL)
        PEAKZ = np.zeros(NHAL)
        MSFR = np.zeros(NHAL)
        SIG_3D = np.zeros(NHAL)
        SIG_1D_x = np.zeros(NHAL)
        SIG_1D_y = np.zeros(NHAL)
        SIG_1D_z = np.zeros(NHAL)
        EDAD = np.zeros(NHAL)
        EDAD_MASS = np.zeros(NHAL)
        MET = np.zeros(NHAL)
        MET_MASS = np.zeros(NHAL)
        VSIGMA = np.zeros(NHAL)
        LAMBDA = np.zeros(NHAL)
        V_TF = np.zeros(NHAL)
        SERSIC = np.zeros(NHAL)
        ELLIPTICITY = np.zeros(NHAL)
        for ihal in tqdm(range(NHAL)):
            part_list = new_groups[ihal]
            PEAKX[ihal], PEAKY[ihal], PEAKZ[ihal] = density_peak(part_list, st_x, st_y, st_z, st_mass)
            cx = PEAKX[ihal]
            cy = PEAKY[ihal]
            cz = PEAKZ[ihal]
            M = MM[ihal]
            RAD05[ihal] = half_mass_radius(cx, cy, cz, M, part_list, st_x, st_y, st_z, st_mass)
            RAD05_x[ihal], RAD05_y[ihal], RAD05_z[ihal] = half_mass_radius_proj(cx, cy, cz, M, part_list, st_x, st_y, st_z, st_mass)
            VX[ihal], VY[ihal], VZ[ihal] = CM_velocity(M, part_list, st_vx, st_vy, st_vz, st_mass)
            V_TF[ihal] = tully_fisher_velocity(part_list, cx, cy, cz, st_x, st_y, st_z, VX[ihal], VY[ihal], VZ[ihal], st_vx, st_vy, st_vz, st_mass, RAD05[ihal])
            JX[ihal], JY[ihal], JZ[ihal] = angular_momentum(M, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, cx, cy, cz, VX[ihal], VY[ihal], VZ[ihal])
            J[ihal] = (JX[ihal]**2 + JY[ihal]**2 + JZ[ihal]**2)**0.5
            #CARE HERE, RMAX IS CALCULATED WITH RESPECT TO THE CENTER OF MASS, NOT THE DENSITY PEAK
            # and RAD05 is calculated with respect to the density peak
            if sersic_flag:
                SERSIC[ihal], ELLIPTICITY[ihal] = sersic_index(part_list, st_x, st_y, st_z, CX[ihal], CY[ihal], CZ[ihal], RAD05[ihal], RAD05_x[ihal], RAD05_y[ihal], RAD05_z[ihal])
            if it_count > 0:
                MSFR[ihal] = star_formation(part_list, st_mass, st_age, cosmo_time*units.time_to_yr/1e9, dt)

            SIG_3D[ihal] = sigma_effective(part_list, RAD05[ihal], st_x, st_y, st_z, st_vx, st_vy, st_vz, cx, cy, cz, VX[ihal], VY[ihal], VZ[ihal])
            grid = np.arange(-(RMAX[ihal]+ll), RMAX[ihal]+ll, 2*ll) #centers of the cells
            n_cell = len(grid)
            SIG_1D_x[ihal], SIG_1D_y[ihal], SIG_1D_z[ihal], VSIGMA[ihal], LAMBDA[ihal] = sigma_projections(grid, n_cell, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, cx, cy, cz, RAD05_x[ihal], RAD05_y[ihal], RAD05_z[ihal])
            EDAD[ihal], EDAD_MASS[ihal], MET[ihal], MET_MASS[ihal] = avg_age_metallicity(part_list, st_age, st_met, st_mass, cosmo_time*units.time_to_yr/1e9)
        
        if len(new_groups)>0:
            print()
            print('CHECK min, max in R_05', np.min(RAD05)*rete*1e3, np.max(RAD05)*rete*1e3)
            print('CHECK min, max in RMAX', np.min(RMAX)*rete*1e3, np.max(RMAX)*rete*1e3)
            print('CHECK min, max in NPART', np.min(NPART), np.max(NPART))
            print('CHECK min, max in J', np.min(J)*rete*1e3, np.max(J)*rete*1e3)
            print()
        
        ##########################################
        ####### MERGER SECTION #####################
        ##########################################

        print('-------> MERGERS/PROGENITORS <-------')

        PRO1 = np.zeros((NHAL), dtype=np.int32)
        PRO2 = np.zeros((NHAL), dtype=np.int32)
        NMERG = np.zeros((NHAL), dtype=np.int32)
        MER_TYPE = np.zeros((NHAL), dtype=np.int32)
        for ih, halo in enumerate(new_groups): 
            oripas = st_oripa[halo]
            masses = st_mass[halo]
            mass_intersections = np.zeros(len(OMM)) # mass coming from haloes in the iteration before
            nmergs = 0
            for oih, ooripas in enumerate(oripas_before):
                intersection = np.in1d(oripas, ooripas, assume_unique = True)
                if np.count_nonzero(intersection) > 0:
                    mass_intersections[oih] = OMM[oih]
                    nmergs += 1

            NMERG[ih] = nmergs
            argsort_intersections = np.flip(np.argsort(mass_intersections)) #sort by mass (INDICES)
            sort_intersections = np.flip(np.sort(mass_intersections)) #sort by mass (MASSES)
            if NMERG[ih] > 0:
                PRO1[ih] = argsort_intersections[0] + 1 #ID in previous iteration of main progenitor
            
                #FIRST LOOK IF THIS HALO IS THE RESULT OF THE MAIN PROGENITOR BREAKING APPART
                if 1.2*MM[ih] < OMM[PRO1[ih]-1]:
                    MER_TYPE[ih] = -1 #Old HALO (OHALO) BREAKING APPART
            
                else:
                    if NMERG[ih] > 1:
                        PRO2[ih] = argsort_intersections[1] + 1 #ID in previous iteration of second progenitor

                        mer_frac = sort_intersections[1]/sort_intersections[0] #MERGER MASS FRACTION
                        if mer_frac > 1/3:
                            MER_TYPE[ih] = 1 #MAJOR MERGER

                        if 1/20 < mer_frac < 1/3:
                            MER_TYPE[ih] = 2 #MINOR MERGER

                        else:                #ACCRETION
                            MER_TYPE[ih] = 3 

        ###########################################################
        ####### SORTING BY NUMBER OF PARTICLES ##################
        ###########################################################

        argsort_part = np.flip(np.argsort(NPART)) #sorted descending

        NPART = NPART[argsort_part]
        MM = MM[argsort_part]
        MSFR = MSFR[argsort_part]
        RMAX = RMAX[argsort_part]
        RAD05 = RAD05[argsort_part]
        RAD05_x = RAD05_x[argsort_part]
        RAD05_y = RAD05_y[argsort_part]
        RAD05_z = RAD05_z[argsort_part]
        SIG_3D = SIG_3D[argsort_part]
        SIG_1D_x = SIG_1D_x[argsort_part]
        SIG_1D_y = SIG_1D_y[argsort_part]
        SIG_1D_z = SIG_1D_z[argsort_part]
        J = J[argsort_part]
        CX = CX[argsort_part]
        CY = CY[argsort_part]
        CZ = CZ[argsort_part]
        PEAKX = PEAKX[argsort_part]
        PEAKY = PEAKY[argsort_part]
        PEAKZ = PEAKZ[argsort_part]
        VX = VX[argsort_part]
        VY = VY[argsort_part]
        VZ = VZ[argsort_part]
        PRO1 = PRO1[argsort_part]
        PRO2 = PRO2[argsort_part]
        NMERG = NMERG[argsort_part]
        MER_TYPE = MER_TYPE[argsort_part]
        EDAD = EDAD[argsort_part]
        EDAD_MASS = EDAD_MASS[argsort_part]
        MET = MET[argsort_part]
        MET_MASS = MET_MASS[argsort_part]
        VSIGMA = VSIGMA[argsort_part]
        LAMBDA = LAMBDA[argsort_part]
        V_TF = V_TF[argsort_part]
        SERSIC = SERSIC[argsort_part]
        ELLIPTICITY = ELLIPTICITY[argsort_part]

        ##########################################
        ##########################################
        ##########################################

        ##########################################
        #oripas of particle in halos of iteration before
        # and masses before
        oripas_before = []
        OMM = np.copy(MM)
        for isort_part in range(len(argsort_part)): 
            halo = new_groups[argsort_part[isort_part]]
            oripas = st_oripa[halo]
            oripas_before.append(oripas)
        ##########################################



    else:
        print('No stars found!!')
        groups = []
        NHAL = 0
        NPARTHAL = 0


    ############################################################
    #################       SAVE DATA           ################
    ############################################################

    print()
    print('Saving data..')
    print()
    print()
    print()

    iteration_data = {}
    iteration_data['nhal'] = NHAL
    iteration_data['nparhal'] = NPARTHAL
    iteration_data['it_halma'] = it_count + 1
    iteration_data['it_masclet'] = iteration
    iteration_data['t'] = cosmo_time
    iteration_data['z'] = zeta

    haloes=[]
    for ih in range(NHAL):
        halo = {}
        halo['id'] = ih+1
        halo['partNum'] = NPART[ih]
        halo['M'] = MM[ih]
        halo['Mv'] = 0.
        halo['Mgas'] = 0.
        halo['fcold'] = 0.
        halo['Mhotgas'] = 0.
        halo['Mcoldgas'] = 0.
        halo['Msfr'] = MSFR[ih]
        halo['Rmax'] = RMAX[ih]*rete*1e3 #kpc
        halo['R'] = RAD05[ih]*rete*1e3
        halo['R_1d'] = (RAD05_x[ih] + RAD05_y[ih] + RAD05_z[ih])/3 * rete * 1e3
        halo['R_1dx'] = RAD05_x[ih]*rete*1e3
        halo['R_1dy'] = RAD05_y[ih]*rete*1e3
        halo['R_1dz'] = RAD05_z[ih]*rete*1e3
        halo['sigma_v'] = SIG_3D[ih]
        halo['sigma_v_1d'] = (SIG_1D_x[ih] + SIG_1D_y[ih] + SIG_1D_z[ih])/3
        halo['sigma_v_1dx'] = SIG_1D_x[ih]
        halo['sigma_v_1dy'] = SIG_1D_y[ih]
        halo['sigma_v_1dz'] = SIG_1D_z[ih]
        halo['L'] = J[ih]*rete*1e3 # kpc km/s
        halo['xcm'] = PEAKX[ih]*1e3 #kpc
        halo['ycm'] = PEAKY[ih]*1e3
        halo['zcm'] = PEAKZ[ih]*1e3
        halo['vx'] = VX[ih]
        halo['vy'] = VY[ih]
        halo['vz'] = VZ[ih]
        halo['father1'] = PRO1[ih]
        halo['father2'] = PRO2[ih]
        halo['nmerg'] = NMERG[ih]
        halo['mergType'] = MER_TYPE[ih]
        halo['age_m'] = EDAD_MASS[ih]
        halo['age'] = EDAD[ih]
        halo['Z_m'] = MET[ih]
        halo['Z'] = MET_MASS[ih]
        #NEW ADDED
        halo['Vsigma'] = VSIGMA[ih]
        halo['lambda'] = LAMBDA[ih]
        halo['v_TF'] = V_TF[ih]
        halo['sersic'] = SERSIC[ih]
        halo['ellipticity'] = ELLIPTICITY[ih]
        haloes.append(halo)

    total_iteration_data.append(iteration_data)
    total_halo_data.append(haloes)


    ###########################################
    ### SAVING PARTICLES in .npy (python friendly)
    if write_particles and len(data)>0:
        string_it = f'{iteration:05d}'
        new_groups = np.array(new_groups, dtype=object)
        new_groups = new_groups[argsort_part]
        all_particles_in_haloes = np.concatenate(new_groups)
        np.save(path_halo_particles+'/halotree'+string_it+'.npy', all_particles_in_haloes)
    ###########################################


write_to_HALMA_catalogue(total_iteration_data, total_halo_data, name = catalogue_name)


########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ##########    END     ########## ##########
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
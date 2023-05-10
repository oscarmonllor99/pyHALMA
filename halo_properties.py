import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
import sys
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
#Our things
sys.path.append('/home/monllor/projects/')
from masclet_framework import units, particles


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

                        # si la sigma de la particula es mayor que un factor
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
def density_peak(part_list, st_x, st_y, st_z, st_mass, ll):
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
def sigma_projections(grid, n_cell, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, cx, cy, cz, R05x, R05y, R05z, ll):
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


def halo_shape(part_list, st_x, st_y, st_z, st_mass, cx, cy, cz, RAD05):
    # CALLS particles module routine to calculate the shape of a given particle list
    x = st_x[part_list] - cx
    y = st_y[part_list] - cy
    z = st_z[part_list] - cz
    m = st_mass[part_list]
    r = RAD05
    semiaxis, _ = particles.ellipsoidal_shape_particles(x, y, z, m, r, tol=1e-3, maxiter=100, preserve='major', verbose=False)
    if semiaxis is None:
        return np.nan, np.nan, np.nan
    else:
        a = semiaxis[2]
        b = semiaxis[1]
        c = semiaxis[0]
        return a, b, c


@njit
def simple_fit(R_list_centers, dR, x_pos, y_pos, z_pos):
    Nc = len(R_list_centers)
    npart = len(x_pos)
    surface_density_xy = np.zeros(Nc)
    surface_density_xz = np.zeros(Nc)
    surface_density_yz = np.zeros(Nc)
    #Find surface number density profile in each projection XY, XZ, YZ
    for ip in range(npart):
        r_xy = np.sqrt(x_pos[ip]**2 + y_pos[ip]**2)
        r_xz = np.sqrt(x_pos[ip]**2 + z_pos[ip]**2)
        r_yz = np.sqrt(y_pos[ip]**2 + z_pos[ip]**2)
        
        #search radial bin
        iR_xy = int(r_xy/dR)
        iR_xz = int(r_xz/dR)
        iR_yz = int(r_yz/dR)

        #update surface density
        if iR_xy < Nc:
            surface_density_xy[iR_xy] += 1.
        if iR_xz < Nc:
            surface_density_xz[iR_xz] += 1.
        if iR_yz < Nc:
            surface_density_yz[iR_yz] += 1.

    #normalize
    surface_density_xy /= (2.*np.pi*R_list_centers*dR)
    surface_density_xz /= (2.*np.pi*R_list_centers*dR)
    surface_density_yz /= (2.*np.pi*R_list_centers*dR)

    return surface_density_xy, surface_density_xz, surface_density_yz



def simple_sersic_index(part_list, st_x, st_y, st_z, cx, cy, cz, R05):
    ##################################################################################################
    # THIS FUNCTION COMPUTES THE SERSIC INDEX OF THE STELLAR HALO ASSUMING CONTOURS ARE CIRCULAR.
    # IT COUNTS THE NUMBER OF PARTICLES IN RADIAL BINS AND FITS A SERSIC PROFILE TO THE SURFACE DENSITY
    # PROFILE. 
    #
    # n IS SENSITIVE TO Nc, THE NUMBER OF RADIAL BINS. IT SHOULD DEPEND ON RESOLUTION AND HALO SIZE.
    #
    # FUTURE IMPROVEMENTS: CONTOURS ARE NOT CIRCULAR, BUT ELLIPTICAL. THE SERSIC INDEX SHOULD BE
    # COMPUTED IN ELLIPTICAL BINS. THIS IS NOT IMPLEMENTED YET.
    ##################################################################################################

    #RADIAL LIMITS
    R_fit_min = 0.
    R_fit_max = 2.*R05

    #ONLY CONSIDER PARTICLES WITHIN R_fit_min and R_fit_max
    x_pos = st_x[part_list]-cx
    y_pos = st_y[part_list]-cy
    z_pos = st_z[part_list]-cz
    R_pos = ((x_pos)**2 + (y_pos)**2 + (z_pos)**2)**0.5

    part_list = part_list[R_pos < R_fit_max]
    x_pos = x_pos[R_pos < R_fit_max]
    y_pos = y_pos[R_pos < R_fit_max]
    z_pos = z_pos[R_pos < R_fit_max]
    R_pos = ((x_pos)**2 + (y_pos)**2 + (z_pos)**2)**0.5

    part_list = part_list[R_pos > R_fit_min]
    x_pos = x_pos[R_pos > R_fit_min]
    y_pos = y_pos[R_pos > R_fit_min]
    z_pos = z_pos[R_pos > R_fit_min]
    R_pos = ((x_pos)**2 + (y_pos)**2 + (z_pos)**2)**0.5

    #RADIAL BINNING
    Nc = 10 # number of "contours", radial bins
    R_list, dR = np.linspace(R_fit_min, R_fit_max, Nc + 1, retstep = True)
    R_list_centers = (R_list[1:] + R_list[:-1])/2.

    surface_density_xy, surface_density_xz, surface_density_yz = simple_fit(R_list_centers, dR, x_pos, y_pos, z_pos)

    # APPLYING A GAUSSIAN FILTER TO SMOOTH THE SURFACE DENSITY
    surface_density_xy = gaussian_filter(surface_density_xy, sigma = 0.5)
    surface_density_xz = gaussian_filter(surface_density_xz, sigma = 0.5)
    surface_density_yz = gaussian_filter(surface_density_yz, sigma = 0.5)

    # FIT THE TAIL: FROM SUPREME TO R_fit_max
    # FIND THE PEAK OF THE SURFACE DENSITY
    iRmax = np.argmax(surface_density_xy)
    R_list_centers = R_list_centers[iRmax:]
    surface_density_xy = surface_density_xy[iRmax:]
    surface_density_xz = surface_density_xz[iRmax:]
    surface_density_yz = surface_density_yz[iRmax:]

    #FITTING THE SERSIC PROFILE
    def sersic(R, Re, Ie, n):
        bn = 2*n - 1/3 + 4/(405*n)
        return Ie*np.exp( -bn*( (R/Re)**(1/n) - 1 ) )

    guess_xy = [R05, surface_density_xy[0], 1.]
    guess_xz = [R05, surface_density_xz[0], 1.]
    guess_yz = [R05, surface_density_yz[0], 1.]

    #FIT XY
    try:
        param_xy, _ = curve_fit(sersic, R_list_centers, surface_density_xy, p0 = guess_xy)
        n_xy = param_xy[2]
    except:
        n_xy = np.nan

    #FIT XZ
    try:
        param_xz, _ = curve_fit(sersic, R_list_centers, surface_density_xz, p0 = guess_xz)
        n_xz = param_xz[2]
    except:
        n_xz = np.nan

    #FIT YZ
    try:
        param_yz, _ = curve_fit(sersic, R_list_centers, surface_density_yz, p0 = guess_yz)
        n_yz = param_yz[2]
    except:
        n_yz = np.nan

    #CHECK UNUSUAL VALUES OF n
    if n_xy < 0.2 or n_xy > 10.:
        n_xy = np.nan
    if n_xz < 0.2 or n_xz > 10.:
        n_xz = np.nan
    if n_yz < 0.2 or n_yz > 10.:
        n_yz = np.nan

    # PLOT THE 3 FITS TO CHECK, IF THEY SUCCEDED
    # print('n_xy = ', n_xy)
    # print('n_xz = ', n_xz)
    # print('n_yz = ', n_yz)
    # fig, ax = plt.subplots(1, 3, figsize = (15, 5))
    # ax[0].plot(R_list_centers, surface_density_xy, 'k.')
    # if not np.isnan(n_xy):
    #     ax[0].plot(R_list_centers, sersic(R_list_centers, *param_xy), 'r--')
    # ax[0].set_title('XY')

    # ax[1].plot(R_list_centers, surface_density_xz, 'k.')
    # if not np.isnan(n_xz):
    #     ax[1].plot(R_list_centers, sersic(R_list_centers, *param_xz), 'r--')
    # ax[1].set_title('XZ')

    # ax[2].plot(R_list_centers, surface_density_yz, 'k.')
    # if not np.isnan(n_yz):
    #     ax[2].plot(R_list_centers, sersic(R_list_centers, *param_yz), 'r--')
    # ax[2].set_title('YZ')
    # plt.show()

    # NANMEAN
    n = np.nanmean([n_xy, n_xz, n_yz])
    return n
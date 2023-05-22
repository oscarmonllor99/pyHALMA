import numpy as np
import pyfof
import sys
from astropy import cosmology
import numba
from tqdm import tqdm
from math import sqrt, log10, acos
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

#Our things
sys.path.append('/home/monllor/projects/')
from masclet_framework import read_masclet, units, tools
import galaxy_image_fit, halo_properties, halo_gas, pycalipso

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
    f.readline() # MAIN BLOCK
    f.readline()
    FIRST,LAST,STEP = np.array(f.readline().split()[0].split(','), dtype = np.int32)
    f.readline()
    NX, NY, NZ = np.array(f.readline().split()[0].split(','), dtype = np.int32)
    f.readline()
    ACHE, OMEGA0, T0 =  np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    Z0, L = np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    LL, =  np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    MINP, =  np.array(f.readline().split()[0].split(','), dtype = np.int32)
    f.readline()
    SIG_FIL, Q_FIL = np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    OLD_MASCLET = bool(int(f.readline()))
    f.readline()
    WRITE_PARTICLES = bool(int(f.readline()))
    f.readline()
    RPS_FLAG = bool(int(f.readline()))
    f.readline()
    CATALOGUE_NAME = f.readline()[:-1]
    f.readline()
    NCORE = int(f.readline())
    f.readline()
    f.readline() #CALIPSO BLOCK
    f.readline()
    CALIPSO_FLAG = bool(int(f.readline()))
    f.readline()
    ZP_5556 = float(f.readline())
    f.readline()
    SB_LIM = float(f.readline())
    f.readline()
    CLIGHT = float(f.readline())
    f.readline() 
    USUN, GSUN, RSUN, ISUN = np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    METSUN = float(f.readline())
    f.readline()
    L_START, L_END = np.array(f.readline().split()[0].split(','), dtype = np.float64)


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
        if len(groups[ig]) >= MINP:
            good_index[ig] = True
    return good_index


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
        catalogue.write('===========================================================================================================')
        catalogue.write('\n')
        it_values = [*total_iteration_data[it_halma].values()]
        for value in it_values:
            catalogue.write(str(value)+'          ')

        
        first_strings = ['Halo','n','Mass','Mass','Mass','frac', 'm_rps','m_rps','m_SFR', 
                         ' R ','R_05','R_05','R_05','R_05','R_05', 'sigma','sigma','sig_x',
                         'sig_y','sig_z','j', 'c_x','c_y','c_z', 'V_x','V_y','V_z','Pro.','Pro.',
                         'n','type', 'age','age', 'Z','Z', 'V/Sigma', 'lambda', 'v_TF', 'a', 'b', 'c', 'sersic',
                         'lum_u','lum_g','lum_r','lum_i','sb_u','sb_g','sb_r','sb_i','ur_color','gr_color','sersic_lum']
        
        second_strings = ['ID',' part', ' * ','*_vis','gas','g_cold',  'cold','hot','  * ', 'max','3D',
                        '1D','1D_x','1D_y','1D_z', '05_3D','05_1D','05_1D','05_1D','05_1D',
                        '  ', 'kpc','kpc','kpc', 'km/s','km/s','km/s',
                        '(1)','(2)','merg','merg','m_weig','mean', 'm_weig','mean', '  ', '  ', 'km/s', 'kpc', 'kpc', 'kpc', '  ',
                        '  ','  ','  ','  ','  ','  ','  ','  ','  ','  ','  ']

        first_line = f'{first_strings[0]:6s}{first_strings[1]:10s}{first_strings[2]:15s}{first_strings[3]:15s}\
{first_strings[4]:15s}{first_strings[5]:8s}{first_strings[6]:15s}{first_strings[7]:15s}{first_strings[8]:15s}\
{first_strings[9]:10s}{first_strings[10]:10s}{first_strings[11]:10s}{first_strings[12]:10s}{first_strings[13]:10s}{first_strings[14]:10s}\
{first_strings[15]:10s}{first_strings[16]:10s}{first_strings[17]:10s}{first_strings[18]:10s}{first_strings[19]:10s}{first_strings[20]:10s}\
{first_strings[21]:10s}{first_strings[22]:10s}{first_strings[23]:10s}{first_strings[24]:10s}{first_strings[25]:10s}{first_strings[26]:10s}\
{first_strings[27]:6s}{first_strings[28]:6s}{first_strings[29]:6s}{first_strings[30]:6s}{first_strings[31]:9s}{first_strings[32]:9s}\
{first_strings[33]:11s}{first_strings[34]:11s}{first_strings[35]:11s}{first_strings[36]:11s}{first_strings[37]:11s}{first_strings[38]:11s}{first_strings[39]:11s}\
{first_strings[40]:11s}{first_strings[41]:11s}{first_strings[42]:11s}{first_strings[43]:11s}{first_strings[44]:11s}{first_strings[45]:11s}{first_strings[46]:11s}\
{first_strings[47]:11s}{first_strings[48]:11s}{first_strings[49]:11s}{first_strings[50]:11s}{first_strings[51]:11s}{first_strings[52]:11s}'
        
        second_line = f'{second_strings[0]:6s}{second_strings[1]:10s}{second_strings[2]:15s}{second_strings[3]:15s}\
{second_strings[4]:15s}{second_strings[5]:8s}{second_strings[6]:15s}{second_strings[7]:15s}{second_strings[8]:15s}\
{second_strings[9]:10s}{second_strings[10]:10s}{second_strings[11]:10s}{second_strings[12]:10s}{second_strings[13]:10s}{second_strings[14]:10s}\
{second_strings[15]:10s}{second_strings[16]:10s}{second_strings[17]:10s}{second_strings[18]:10s}{second_strings[19]:10s}{second_strings[20]:10s}\
{second_strings[21]:10s}{second_strings[22]:10s}{second_strings[23]:10s}{second_strings[24]:10s}{second_strings[25]:10s}{second_strings[26]:10s}\
{second_strings[27]:6s}{second_strings[28]:6s}{second_strings[29]:6s}{second_strings[30]:6s}{second_strings[31]:9s}{second_strings[32]:9s}\
{second_strings[33]:11s}{second_strings[34]:11s}{second_strings[35]:11s}{second_strings[36]:11s}{second_strings[37]:11s}{second_strings[38]:11s}{second_strings[39]:11s}\
{second_strings[40]:11s}{second_strings[41]:11s}{second_strings[42]:11s}{second_strings[43]:11s}{second_strings[44]:11s}{second_strings[45]:11s}{second_strings[46]:11s}\
{second_strings[47]:11s}{second_strings[48]:11s}{second_strings[49]:11s}{second_strings[50]:11s}{second_strings[51]:11s}{second_strings[52]:11s}'
        
        catalogue.write('\n')
        catalogue.write('------------------------------------------------------------------------------------------------------------')
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
{ih_values[35]:11.2f}{gap}{ih_values[36]:11.2f}{gap}{ih_values[37]:11.2f}{gap}{ih_values[38]:11.2f}{gap}{ih_values[39]:11.2f}{gap}\
{ih_values[40]:11.2f}{gap}{gap}{ih_values[41]:11.2f}{gap}{ih_values[42]:11.2e}{gap}{ih_values[43]:11.2e}{gap}{ih_values[44]:11.2e}{gap}\
{ih_values[45]:11.2e}{gap}{ih_values[46]:11.2f}{ih_values[47]:11.2f}{ih_values[48]:11.2f}{ih_values[49]:11.2f}{ih_values[50]:11.2f}\
{ih_values[51]:11.2f}{ih_values[52]:11.2f}'
            
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

DIRSSP = 'E-MILES'

########## ########## ########## ########## ##########
### CALIPSO BLOCK
########## ########## ########## ########## ##########
if CALIPSO_FLAG:
    print('*************************************')
    print('Before FoF, reading calipso files..')
    WAVELENGHTS, SSP, AGE_SPAN, Z_SPAN, MH_SPAN, N_AGES, N_Z, N_W = pycalipso.readSSPfiles(DIRSSP, L_START, L_END)
    N_F, N_LINES_FILTERS, W_FILTERS, RESPONSE_FILTERS = pycalipso.readFilters()
    W_VEGA, FLUX_VEGA, N_VEGA = pycalipso.readVega(ZP_5556)
    print('.. done')
    dlum = 1.e-5 #10 pc --> absolute mag
    magssp = np.zeros((N_AGES,N_Z,N_F))
    fluxssp = np.zeros((N_AGES,N_Z,N_F))
    LUMU = np.zeros((N_AGES,N_Z))
    LUMG = np.zeros((N_AGES,N_Z))
    LUMR = np.zeros((N_AGES,N_Z))
    LUMI = np.zeros((N_AGES,N_Z))
    # HERE WE CALCULATE ABSOLUTE MAGNITUDES (SINCE DLUM = 1 PC) AND THUS LUMINOSITIES
    # Calculating luminosity in u,g,r filters of each SSP
    for iage in range(N_AGES):
        for iZ in range(N_Z):
            pycalipso.mag_v1_0(WAVELENGHTS, SSP[iage, iZ, :], N_W, magssp[iage, iZ, :],
                                fluxssp[iage, iZ, :], N_F, N_LINES_FILTERS, W_FILTERS, RESPONSE_FILTERS, W_VEGA, FLUX_VEGA, N_VEGA, dlum)
            
            LUMU[iage,iZ]=10.**(-0.4*(magssp[iage,iZ, 0]-USUN)) #luminosity in u filter (U band more or less)
            LUMG[iage,iZ]=10.**(-0.4*(magssp[iage,iZ, 1]-GSUN)) #luminosity in g filter (B band more or less)
            LUMR[iage,iZ]=10.**(-0.4*(magssp[iage,iZ, 2]-RSUN)) #luminosity in r filter (R band more or less)
            LUMI[iage,iZ]=10.**(-0.4*(magssp[iage,iZ, 3]-ISUN)) #luminosity in i filter (I band more or less)

            
    CLOSEST_START = np.abs(L_START/WAVELENGHTS - 1)
    CLOSEST_END = np.abs(L_END/WAVELENGHTS - 1)
    I_START = np.argmin(CLOSEST_START) 
    I_END = np.argmin(CLOSEST_END) 

    print('Spectrum will be written in the range:', WAVELENGHTS[I_START], WAVELENGHTS[I_END], '(A)')
    DISP=WAVELENGHTS[2]-WAVELENGHTS[1] #resolución espectral

    print('*************************************')
    print()
    print()
    print()

########## ########## ########## ########## ##########
########## ########## ########## ########## ##########


PATH_RESULTS = 'simu_masclet'

RE0 = 1.0  #factor de escala a z = 0

print('****************************************************')
print('******************** MASCLET pyfof *****************')
print('****************************************************')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('CHECK!!!! --> LINKING LENGHT (kpc)', LL*1e3)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print()

###############################
if __name__ == '__main__':
    numba.set_num_threads(NCORE)
###############################

print('----> Available CPU cores:', numba.config.NUMBA_NUM_THREADS)
print('----> Using', numba.get_num_threads(), 'cores')

total_iteration_data = []
total_halo_data = []

#loop over iterations
oripas_before = []
omm = [] #HALOS PREVIOUS ITERATION
for it_count, iteration in enumerate(range(FIRST, LAST+STEP, STEP)):
    print()
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('             Iteration', iteration)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('Opening MASCLET files')
    print()
    #open MASCLET files
    
    #READ GRID
    grid_data = read_masclet.read_grids(iteration, path=PATH_RESULTS, parameters_path=PATH_RESULTS, digits=5, 
                                                read_general=True, read_patchnum=True, read_dmpartnum=False,
                                                read_patchcellextension=True, read_patchcellposition=True, read_patchposition=True,
                                                read_patchparent=False)
    cosmo_time = grid_data[1]
    zeta = grid_data[4]
    rete = RE0 / (1.0+zeta)
    
    dt = 0.
    if it_count>0:
        dt = (cosmo_time - total_iteration_data[it_count-1]['t'])*units.time_to_yr/1e9

    print('Cosmo time (Gyr):', cosmo_time*units.time_to_yr/1e9)
    print('Redshift (z):', zeta)

    #READ GAS IF RPS
    if RPS_FLAG:
        print('RPS == True !!!! ')
        print('     Opening grid, clus and DM files')

        #READ CLUS
        gas_data = read_masclet.read_clus(iteration, path=PATH_RESULTS, parameters_path=PATH_RESULTS, digits=5, max_refined_level=1000, output_delta=True, 
                                          output_v=True, output_pres=False, output_pot=False, output_opot=False, output_temp=True, output_metalicity=False,
                                          output_cr0amr=True, output_solapst=True, is_mascletB=False, output_B=False, is_cooling=True, verbose=False)


        #READ DM
        masclet_dm_data = read_masclet.read_cldm(iteration, path = PATH_RESULTS, parameters_path=PATH_RESULTS, 
                                                    digits=5, max_refined_level=1000, output_deltadm = False,
                                                    output_position=True, output_velocity=False, output_mass=True)


    masclet_st_data = read_masclet.read_clst(iteration, path = PATH_RESULTS, parameters_path=PATH_RESULTS, 
                                                    digits=5, max_refined_level=1000, 
                                                    output_deltastar=False, verbose=False, output_position=True, 
                                                    output_velocity=True, output_mass=True, output_time=True,
                                                    output_metalicity=True, output_id=True, are_BH = not OLD_MASCLET)

    st_x = masclet_st_data[0]
    st_y = masclet_st_data[1]
    st_z = masclet_st_data[2]
    st_vx = masclet_st_data[3]*CLIGHT #in km/s
    st_vy = masclet_st_data[4]*CLIGHT #in km/s
    st_vz = masclet_st_data[5]*CLIGHT #in km/s
    st_mass = masclet_st_data[6]*units.mass_to_sun #in Msun
    st_age = masclet_st_data[7]*units.time_to_yr/1e9 #in Gyr
    st_met = masclet_st_data[8]
    st_oripa = masclet_st_data[9] #necessary for mergers


    data = np.vstack((st_x, st_y, st_z)).T
    data = data.astype(np.float64)
    if len(data) > 0: #HAY ESTRELLAS
        #APPLY FOF
        print()
        print('----------> FoF begins <--------')
        print()
        groups = pyfof.friends_of_friends(data = data, linking_length = LL)
        groups = np.array(groups, dtype=object)
        #CLEAN THOSE HALOES with npart < minp
        groups = groups[good_groups(groups)]
        print(len(groups),'haloes found')
        print()
        #CALCULATE CM, CM_vel AND phase-space cleaning
        print('---> Phase-space cleaning begins <---')
        center_x = []
        center_y = []
        center_z = []
        masses = []
        rmax = []
        num_particles = []
        num_cells = []
        new_groups = []
        which_cell_list_x = [] 
        which_cell_list_y = []
        which_cell_list_z = []
        for ihal in tqdm(range(len(groups))):
            # Particle indices, center of mass and mass
            part_list = np.array(groups[ihal])
            cx, cy, cz, mass = halo_properties.center_of_mass(part_list, st_x, st_y, st_z, st_mass)

            #Create 3D grid with cellsize 2*LL for cleaning. Calculate most distant particle to center of mass
            most_distant_r = halo_properties.furthest_particle(cx, cy, cz, part_list, st_x, st_y, st_z)
            grid = np.arange(- ( most_distant_r + LL ), most_distant_r + LL, 2*LL)
            n_cell = len(grid)
            vcm_cell = np.zeros((n_cell, n_cell, n_cell, 3))
            mass_cell = np.zeros((n_cell, n_cell, n_cell))
            quantas_cell = np.zeros((n_cell, n_cell, n_cell))
            sig3D_cell = np.zeros((n_cell, n_cell, n_cell))

            #CALCULATE cell quantities
            (vcm_cell, mass_cell, quantas_cell, sig3D_cell, 
             which_cell_x, which_cell_y, which_cell_z) = halo_properties.calc_cell(cx, cy, cz, grid, part_list, 
                                                                                   st_x, st_y, st_z, st_vx, st_vy, st_vz, 
                                                                                   st_mass, vcm_cell, mass_cell, quantas_cell, sig3D_cell)
            
            #DO THE CLEANING: Right CM, Mass and Furthest particle
            (cx, cy, cz, mass, 
             most_distant_r, control,
             which_cell_x, which_cell_y, which_cell_z) = halo_properties.clean_cell(cx, cy, cz, mass, most_distant_r, grid, part_list, st_x, st_y, 
                                                                                    st_z, st_vx, st_vy, st_vz, st_mass, vcm_cell,
                                                                                    mass_cell, quantas_cell, sig3D_cell, LL, Q_FIL, SIG_FIL,
                                                                                    which_cell_x, which_cell_y, which_cell_z)

            #CLEANING DONE --> CALCULATE AGAIN CENTER OF MASS AND MASS
            control = control.astype(bool)
            npart = len(part_list[control])
            cx, cy, cz, mass = halo_properties.center_of_mass(part_list[control], st_x, st_y, st_z, st_mass)
            most_distant_r = halo_properties.furthest_particle(cx, cy, cz, part_list[control], st_x, st_y, st_z)

            #If after cleaning there are more than MINP particles, save the halo
            if npart > MINP:
                center_x.append(cx)
                center_y.append(cy)
                center_z.append(cz)
                masses.append(mass)
                rmax.append(most_distant_r)
                num_particles.append(npart)
                num_cells.append(n_cell)
                new_groups.append(part_list[control])
                which_cell_list_x.append(which_cell_x)
                which_cell_list_y.append(which_cell_y)
                which_cell_list_z.append(which_cell_z)

        #From list to arrays for convenience
        center_x = np.array(center_x)
        center_y = np.array(center_y)
        center_z = np.array(center_z)
        masses = np.array(masses)
        rmax = np.array(rmax)
        num_particles = np.array(num_particles).astype(np.int32)
        num_halos = len(new_groups)
        n_part_in_halos = np.sum(num_particles)
        num_cells = np.array(num_cells).astype(np.int32)
        which_cell_list_x = np.array(which_cell_list_x, dtype=object)
        which_cell_list_y = np.array(which_cell_list_y, dtype=object)
        which_cell_list_z = np.array(which_cell_list_z, dtype=object)

        print('Number of haloes after phase-space cleaning:', num_halos)
        print('Number of particles in haloes after cleaning:', n_part_in_halos)

        print()
        print('Calculating properties')

        #CALCULATE HALO PROPERTIES
        rad05 = np.zeros(num_halos)
        rad05_x = np.zeros(num_halos)
        rad05_y = np.zeros(num_halos)
        rad05_z = np.zeros(num_halos)
        velocities_x = np.zeros(num_halos)
        velocities_y = np.zeros(num_halos)
        velocities_z = np.zeros(num_halos)
        specific_angular_momentum_x = np.zeros(num_halos)
        specific_angular_momentum_y = np.zeros(num_halos)
        specific_angular_momentum_z = np.zeros(num_halos)
        specific_angular_momentum = np.zeros(num_halos)
        density_peak_x = np.zeros(num_halos)
        density_peak_y = np.zeros(num_halos)
        density_peak_z = np.zeros(num_halos)
        star_formation_masses = np.zeros(num_halos)
        sig_3D = np.zeros(num_halos)
        sig_1D_x = np.zeros(num_halos)
        sig_1D_y = np.zeros(num_halos)
        sig_1D_z = np.zeros(num_halos)
        stellar_ages = np.zeros(num_halos)
        stellar_age_mass = np.zeros(num_halos)
        metallicities = np.zeros(num_halos)
        metallicities_mass = np.zeros(num_halos)
        vsigma = np.zeros(num_halos)
        lambda_ensellem = np.zeros(num_halos)
        tully_fisher_velocities = np.zeros(num_halos)
        s_axis_major = np.zeros(num_halos)
        s_axis_intermediate = np.zeros(num_halos)
        s_axis_minor = np.zeros(num_halos)
        sersic_indices = np.zeros(num_halos)
        gas_masses = np.zeros(num_halos)
        cold_bound_gas_fractions = np.zeros(num_halos)
        cold_unbound_gas_masses = np.zeros(num_halos)
        hot_unbound_gas_masses = np.zeros(num_halos)

        for ihal in tqdm(range(num_halos)):
            #HALO PARTICLE INDICES
            part_list = new_groups[ihal]

            #DENSITY PEAK
            density_peak_x[ihal], density_peak_y[ihal], density_peak_z[ihal] = halo_properties.density_peak(part_list, st_x, st_y, st_z, st_mass, LL)

            #ASSUME SOME CENTERING, CENTER OF MASS OR DENSITY PEAK
            #CARE HERE, RMAX IS CALCULATED WITH RESPECT TO THE CENTER OF MASS, NOT THE DENSITY PEAK

            cx = center_x[ihal]
            cy = center_y[ihal]
            cz = center_z[ihal]
            mass = masses[ihal]

            #EFFECTIVE RADIUS
            rad05[ihal] = halo_properties.half_mass_radius(cx, cy, cz, mass, part_list, st_x, st_y, st_z, st_mass)

            (rad05_x[ihal], 
             rad05_y[ihal], 
             rad05_z[ihal]) = halo_properties.half_mass_radius_proj(cx, cy, cz, mass, part_list, st_x, st_y, st_z, st_mass)
            
            #BULK VELOCITY
            (velocities_x[ihal], 
             velocities_y[ihal], 
             velocities_z[ihal]) = halo_properties.CM_velocity(mass, part_list, st_vx, st_vy, st_vz, st_mass)
            
            #TULLY-FISHER
            tully_fisher_velocities[ihal] = halo_properties.tully_fisher_velocity(part_list, cx, cy, cz, st_x, st_y, st_z, 
                                                                                  velocities_x[ihal], velocities_y[ihal], velocities_z[ihal], 
                                                                                  st_vx, st_vy, st_vz, st_mass, rad05[ihal])
            #ANGULAR MOMENTUM
            (specific_angular_momentum_x[ihal], 
             specific_angular_momentum_y[ihal],
             specific_angular_momentum_z[ihal]) = halo_properties.angular_momentum(mass, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, 
                                                                                   cx, cy, cz, velocities_x[ihal], velocities_y[ihal], velocities_z[ihal])
            
            specific_angular_momentum[ihal] = ( specific_angular_momentum_x[ihal]**2 + 
                                                specific_angular_momentum_y[ihal]**2 + 
                                                specific_angular_momentum_z[ihal]**2 )**0.5
            
            #SHAPE
            (s_axis_major[ihal], 
             s_axis_intermediate[ihal], 
             s_axis_minor[ihal]) = halo_properties.halo_shape(part_list, st_x, st_y, st_z, st_mass, cx, cy, cz, rad05[ihal])
            
            #SÉRSIC INDEX
            sersic_indices[ihal] = halo_properties.simple_sersic_index(part_list, st_x, st_y, st_z, center_x[ihal], center_y[ihal], center_z[ihal], rad05[ihal])
            if it_count > 0:
                star_formation_masses[ihal] = halo_properties.star_formation(part_list, st_mass, st_age, cosmo_time*units.time_to_yr/1e9, dt)

            #SIGMA
            sig_3D[ihal] = halo_properties.sigma_effective(part_list, rad05[ihal], st_x, st_y, st_z, st_vx, st_vy, st_vz, cx, cy, cz, 
                                                           velocities_x[ihal], velocities_y[ihal], velocities_z[ihal])

            # FAST / SLOW ROTATOR 
            grid = np.arange(-(rmax[ihal]+LL), rmax[ihal]+LL, 2*LL) #centers of the cells
            n_cell = len(grid)

            (sig_1D_x[ihal], 
             sig_1D_y[ihal], 
             sig_1D_z[ihal], 
             vsigma[ihal], 
             lambda_ensellem[ihal]) = halo_properties.sigma_projections(grid, n_cell, part_list, st_x, st_y, st_z, 
                                                                        st_vx, st_vy, st_vz, st_mass, cx, cy, cz, 
                                                                        rad05_x[ihal], rad05_y[ihal], rad05_z[ihal], LL)
            
            # METALLICITY AND AGE
            (stellar_ages[ihal], 
             stellar_age_mass[ihal], 
             metallicities[ihal], 
             metallicities_mass[ihal]) = halo_properties.avg_age_metallicity(part_list, st_age, st_met, st_mass, cosmo_time*units.time_to_yr/1e9)

            # RPS EFFECTS
            if RPS_FLAG:
                rps_radius = 2*rad05[ihal] #RADIUS TO CALCULATE COLD/HOT BOUND/UNBOUND GAS
                (
                    gas_masses[ihal], 
                    cold_bound_gas_fractions[ihal], 
                    cold_unbound_gas_masses[ihal], 
                    hot_unbound_gas_masses[ihal]      ) = halo_gas.RPS(rete, L, NX, grid_data, gas_data, 
                                                                       masclet_dm_data, masclet_st_data, cx, cy, cz, 
                                                                       velocities_x[ihal], velocities_y[ihal], velocities_z[ihal], rps_radius)
                
        ############################################################################################################

        if len(new_groups)>0:
            print()
            print(f'CHECK min, max in R_05 {np.min(rad05)*rete*1e3:.2f} {np.max(rad05)*rete*1e3:.2f}')
            print(f'CHECK min, max in RMAX {np.min(rmax)*rete*1e3:.2f} {np.max(rmax)*rete*1e3:.2f}')
            print(f'CHECK min, max in stellar mass {np.min(masses):.2e} {np.max(masses):.2e}')
            print(f'CHECK min, max in gas mass {np.min(gas_masses):.2e} {np.max(gas_masses):.2e}')
            print(f'CHECK min, max in J {np.min(specific_angular_momentum)*rete*1e3:.2f} {np.max(specific_angular_momentum)*rete*1e3:.2f}')
            print()

        ##########################################
        ####### MERGER SECTION #####################
        ##########################################

        print('-------> MERGERS/PROGENITORS <-------')

        pro1 = np.zeros(num_halos, dtype=np.int32)
        pro2 = np.zeros(num_halos, dtype=np.int32)
        n_mergers = np.zeros(num_halos, dtype=np.int32)
        merger_type = np.zeros(num_halos, dtype=np.int32)
        for ih, halo in enumerate(new_groups): 
            oripas = st_oripa[halo]
            mass_intersections = np.zeros(len(omm)) # mass coming from haloes in the iteration before
            nmergs = 0
            for oih, ooripas in enumerate(oripas_before):
                intersection = np.in1d(oripas, ooripas, assume_unique = True)
                if np.count_nonzero(intersection) > 0:
                    mass_intersections[oih] = omm[oih]
                    nmergs += 1

            n_mergers[ih] = nmergs
            argsort_intersections = np.flip(np.argsort(mass_intersections)) #sort by mass (INDICES)
            sort_intersections = np.flip(np.sort(mass_intersections)) #sort by mass (MASSES)
            if n_mergers[ih] > 0:
                pro1[ih] = argsort_intersections[0] + 1 #ID in previous iteration of main progenitor
            
                #FIRST LOOK IF THIS HALO IS THE RESULT OF THE MAIN PROGENITOR BREAKING APPART
                if 1.2*masses[ih] < omm[pro1[ih]-1]:
                    merger_type[ih] = -1 #Old HALO (OHALO) BREAKING APPART
            
                else:
                    if n_mergers[ih] > 1:
                        pro2[ih] = argsort_intersections[1] + 1 #ID in previous iteration of second progenitor

                        mer_frac = sort_intersections[1]/sort_intersections[0] #MERGER MASS FRACTION
                        if mer_frac > 1/3:
                            merger_type[ih] = 1 #MAJOR MERGER

                        if 1/20 < mer_frac < 1/3:
                            merger_type[ih] = 2 #MINOR MERGER

                        else:                #ACCRETION
                            merger_type[ih] = 3 

        print('-------> DONE <-------')

        ###########################################################
        ####### SORTING BY NUMBER OF PARTICLES ##################
        ###########################################################

        argsort_part = np.flip(np.argsort(num_particles)) #sorted descending

        num_particles = num_particles[argsort_part]
        num_cells = num_cells[argsort_part]
        which_cell_list_x = which_cell_list_x[argsort_part]
        which_cell_list_y = which_cell_list_y[argsort_part]
        which_cell_list_z = which_cell_list_z[argsort_part]
        masses = masses[argsort_part]
        star_formation_masses = star_formation_masses[argsort_part]
        rmax = rmax[argsort_part]
        rad05 = rad05[argsort_part]
        rad05_x = rad05_x[argsort_part]
        rad05_y = rad05_y[argsort_part]
        rad05_z = rad05_z[argsort_part]
        sig_3D = sig_3D[argsort_part]
        sig_1D_x = sig_1D_x[argsort_part]
        sig_1D_y = sig_1D_y[argsort_part]
        sig_1D_z = sig_1D_z[argsort_part]
        specific_angular_momentum = specific_angular_momentum[argsort_part]
        center_x = center_x[argsort_part]
        center_y = center_y[argsort_part]
        center_z = center_z[argsort_part]
        density_peak_x = density_peak_x[argsort_part]
        density_peak_y = density_peak_y[argsort_part]
        density_peak_z = density_peak_z[argsort_part]
        velocities_x = velocities_x[argsort_part]
        velocities_y = velocities_y[argsort_part]
        velocities_z = velocities_z[argsort_part]
        pro1 = pro1[argsort_part]
        pro2 = pro2[argsort_part]
        n_mergers = n_mergers[argsort_part]
        merger_type = merger_type[argsort_part]
        stellar_ages = stellar_ages[argsort_part]
        stellar_age_mass = stellar_age_mass[argsort_part]
        metallicities = metallicities[argsort_part]
        metallicities_mass = metallicities_mass[argsort_part]
        vsigma = vsigma[argsort_part]
        lambda_ensellem = lambda_ensellem[argsort_part]
        tully_fisher_velocities = tully_fisher_velocities[argsort_part]
        s_axis_major = s_axis_major[argsort_part]
        s_axis_intermediate = s_axis_intermediate[argsort_part]
        s_axis_minor = s_axis_minor[argsort_part]
        sersic_indices = sersic_indices[argsort_part]
        gas_masses = gas_masses[argsort_part]
        cold_bound_gas_fractions = cold_bound_gas_fractions[argsort_part]
        cold_unbound_gas_masses = cold_unbound_gas_masses[argsort_part]
        hot_unbound_gas_masses = hot_unbound_gas_masses[argsort_part]

        ##########################################
        ##########################################
        ##########################################

        ##########################################
        #oripas of particle in halos of iteration before
        # and masses before
        oripas_before = []
        omm = np.copy(masses)
        for isort_part in range(len(argsort_part)): 
            halo = new_groups[argsort_part[isort_part]]
            oripas = st_oripa[halo]
            oripas_before.append(oripas)
        ##########################################

        ############################################################################################################
        # CALIPSO BLOCK
        ############################################################################################################

        #NOTES:
        # - SSP is in Lo A^-1 Mo^-1
        # - The luminosity weights are in g-band (see pycalipso.make_light)
        # - Care should be taken in the spectral range in which calculations are done. The filter ranges must be inside.
        # - Surface brightness can be: central surface brightness, or surface brightness within the effective radius


        # See that this arrays are already sorted by number of particles
        lum_u = np.zeros(num_halos)
        lum_g = np.zeros(num_halos)
        lum_r = np.zeros(num_halos)
        lum_i = np.zeros(num_halos)
        sb_u = np.zeros(num_halos)
        sb_g = np.zeros(num_halos)
        sb_r = np.zeros(num_halos)
        sb_i = np.zeros(num_halos)
        central_u = np.zeros(num_halos)
        central_g = np.zeros(num_halos)
        central_r = np.zeros(num_halos)
        central_i = np.zeros(num_halos)
        ur_color = np.zeros(num_halos)
        gr_color = np.zeros(num_halos)
        sersic_index_lum = np.zeros(num_halos)
        sersic_counter = np.zeros(num_halos)

        if CALIPSO_FLAG:
            
            print()
            print()
            print(' -----> CALIPSO BLOCK <-----')
            print()
            print('Establishing cosmology')
            print()

            #Sorting haloes
            new_groups_calipso = np.array(new_groups, dtype = object)[argsort_part]

            #Grid data to find resolution for calipso
            npatch = grid_data[5] #number of patches in each level, starting in l=0
            patchnx = grid_data[6] #patchnx (...): x-extension of each patch (in level l cells) (and Y and Z)
            patchny = grid_data[7]
            patchnz = grid_data[8]
            patchrx = grid_data[12] #patchrx (...): physical position of the center of each patch first ¡l-1! cell (and Y and Z)
            patchry = grid_data[13] # in Mpc
            patchrz = grid_data[14]

            #String to save calipso outputs
            string_it = f'{iteration:05d}'
            #Establishing FLAT LCDM cosmology 
            # NOTICE HERE DLUM IS NOT 1 PC, AND HENCE MAGNITUDES HERE ARE NOT ABSOLUTE
            cosmo = cosmology.FlatLambdaCDM(H0=ACHE*100, Om0=OMEGA0)
            dlum = abs(cosmo.luminosity_distance(zeta).value)
            daa = abs(cosmo.angular_diameter_distance(zeta).value) # if zeta < 0, not defined
            arcsec2kpc = daa*1e3*(acos(-1)/180.)/3600. #from angular size to kpc
            # Area of pixel in arcsec^2
            res = LL*1e3 # resolution of the grid in kpc
            area_com = res * res
            area_pys=area_com/((1.+zeta)*(1.+zeta))
            area_arc=area_pys/(arcsec2kpc*arcsec2kpc)
            print('zeta, dist_lum, arcsec2kpc:', zeta, dlum, arcsec2kpc)
            print()

            # List with calipso input to pass to pycalipso.main
            calipso_input = [ CLIGHT, WAVELENGHTS, SSP, 
                              AGE_SPAN, Z_SPAN, MH_SPAN,N_AGES, N_Z, N_W,
                              N_F, N_LINES_FILTERS, W_FILTERS, RESPONSE_FILTERS, 
                              W_VEGA, FLUX_VEGA, N_VEGA,
                              I_START, I_END, DISP, LUMG,
                              zeta, dlum, arcsec2kpc, area_arc ]

            print('Calculating SEDs of the galaxies')
            for ihal in tqdm(range(num_halos)):
                npart = num_particles[ihal] # number of particles in the halo
                halo = new_groups_calipso[ihal] # halo particle indices for array slicing
                halo = np.array(halo, dtype = int)
                
                mass = st_mass[halo] #mass of halo particles
                met = st_met[halo] #metallicity of halo particles
                age = st_age[halo] #age of halo particles
                x = (st_x[halo] - center_x[ihal])*1e3 #relative position of halo particles
                y = (st_y[halo] - center_y[ihal])*1e3 # in kpc
                z = (st_z[halo] - center_z[ihal])*1e3
                velx = st_vx[halo] - velocities_x[ihal] 
                vely = st_vy[halo] - velocities_y[ihal] #LOS velocity of halo particles
                velz = st_vz[halo] - velocities_z[ihal]

                # # CREATE THE GRID with maximum resolution applied to the halo in the simulation
                # box = [CX[ihal] - RMAX[ihal], CX[ihal] + RMAX[ihal], CY[ihal] - RMAX[ihal], 
                #        CY[ihal] + RMAX[ihal], CZ[ihal] - RMAX[ihal], CZ[ihal] + RMAX[ihal]]
                
                # which_patches = tools.which_patches_inside_box(box, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, L, nx)
                # patch_level = tools.create_vector_levels(npatch)
                # max_level = np.max(patch_level[which_patches])
                # res_minimum_cell = L / nx / 2**max_level * 1e3 # maximum resolution of the simulation in kpc

                grid_edges = np.arange(-rmax[ihal]*1e3 - res, rmax[ihal]*1e3 + res, res)
                grid_centers = (grid_edges[1:] + grid_edges[:-1]) / 2
                ncell = len(grid_centers)  # number of cells in each direction
                which_cell_x, which_cell_y, which_cell_z = pycalipso.put_particles_in_grid(grid_centers, x, y, z)

                # finner grid to interpolate
                res_interp =  L / NX / 2**9 * 1e3
                grid_interp_edges = np.arange(grid_centers[0], grid_centers[-1], res_interp)
                grid_interp = (grid_interp_edges[1:] + grid_interp_edges[:-1]) / 2
                x_meshgrid_interp, y_meshgrid_interp = np.meshgrid(grid_interp, grid_interp)

                
                # List with star particle data to pass to pycalipso.main
                star_particle_data = [npart, mass, met, age]


                ############################################################################################################
                ############################################################################################################
                ############################################################################################################
                ############################################################################################################

                #########################################################################################
                #########################################################################################
                # CALCULATION IN XY PLANE
                #########################################################################################
                #########################################################################################

                tam_i = which_cell_x
                tam_j = which_cell_y
                vel_LOS = velz
                effective_radius = rad05_z[ihal]*1e3

                (   fluxtot,
                    lumtotu, lumtotg, lumtotr, lumtoti, 
                    sb_u_tot, sb_g_tot, sb_r_tot, sb_i_tot, 
                    central_sb_u, central_sb_g, central_sb_r, central_sb_i,
                    gr, ur,
                    sbf, magf, fluxf    )  = pycalipso.main(calipso_input, star_particle_data, 
                                                        ncell, vel_LOS, tam_i, tam_j, effective_radius)

                # SÉRSIC INDEX WITH LIGHT (g filter SDSS)
                # FIRST: LINEAR INTERPOLATION OF THE FLUX IN THE GRID

                flux_interp = RegularGridInterpolator((grid_centers, grid_centers), fluxf[:,:, 1], method='linear')
                flux_interpolated = flux_interp((x_meshgrid_interp, y_meshgrid_interp))

                # COMPARE INTERPOLATED FLUX WITH THE ORIGINAL ONE
                n, eps = galaxy_image_fit.photutils_fit(effective_radius, 0., 2*effective_radius, res_interp, flux_2D = flux_interpolated)

                # SAVE DATA
                # FIRST, THE CATALOGUE
                lum_u[ihal] += lumtotu
                lum_g[ihal] += lumtotg
                lum_r[ihal] += lumtotr
                lum_i[ihal] += lumtoti
                sb_u[ihal] += sb_u_tot
                sb_g[ihal] += sb_g_tot
                sb_r[ihal] += sb_r_tot
                sb_i[ihal] += sb_i_tot
                central_u[ihal] += central_sb_u
                central_g[ihal] += central_sb_g
                central_r[ihal] += central_sb_r
                central_i[ihal] += central_sb_i
                ur_color[ihal] += ur
                gr_color[ihal] += gr
                if n>0.:
                    sersic_index_lum[ihal] += n
                    sersic_counter[ihal] += 1
                
                # SECOND IMAGES AND SPECTRA IN FILES
                # Saving spectra of the whole galaxy (not each cell):
                file_fluxtot_x = 'calipso_output/spectra_x/fluxtot'+string_it+'ih'+str(ihal+1)
                np.save(file_fluxtot_x, fluxtot)
                # Saving image (flux, mag and SB of each cell):
                file_image_x = 'calipso_output/images_x/image'+string_it+'ih'+str(ihal+1)
                file_image_array_x = np.zeros((ncell*ncell+1, N_F*3))
                file_image_array_x[0, 0] = ncell #HEADER
                file_image_array_x[0, 1] = ncell
                file_image_array_x[0, 2] = LL
                for i_f in range(N_F):
                    file_image_array_x[1:, i_f] = magf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_x[1:, N_F+i_f] = fluxf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_x[1:, 2*N_F+i_f] = sbf[:,:,i_f].reshape((ncell*ncell))
                np.save(file_image_x, file_image_array_x)

                #########################################################################################
                #########################################################################################
                # CALCULATION IN XZ PLANE
                #########################################################################################
                #########################################################################################

                tam_i = which_cell_x
                tam_j = which_cell_z
                vel_LOS = vely
                effective_radius = rad05_y[ihal]*1e3

                (   fluxtot,
                    lumtotu, lumtotg, lumtotr, lumtoti,
                    sb_u_tot, sb_g_tot, sb_r_tot, sb_i_tot,
                    central_sb_u, central_sb_g, central_sb_r, central_sb_i,
                    gr, ur,
                    sbf, magf, fluxf    )  = pycalipso.main(calipso_input, star_particle_data,
                                                        ncell, vel_LOS, tam_i, tam_j, effective_radius)
                

                # SÉRSIC INDEX WITH LIGHT (g filter SDSS)
                # FIRST: LINEAR INTERPOLATION OF THE FLUX IN THE GRID
                flux_interp = RegularGridInterpolator((grid_centers, grid_centers), fluxf[:,:, 1], method='linear')
                flux_interpolated = flux_interp((x_meshgrid_interp, y_meshgrid_interp))

                # COMPARE INTERPOLATED FLUX WITH THE ORIGINAL ONE
                n, eps = galaxy_image_fit.photutils_fit(effective_radius, 0., 2*effective_radius, res_interp, flux_2D = flux_interpolated) 

                # SAVE DATA
                # FIRST, THE CATALOGUE
                lum_u[ihal] += lumtotu
                lum_g[ihal] += lumtotg
                lum_r[ihal] += lumtotr
                lum_i[ihal] += lumtoti
                sb_u[ihal] += sb_u_tot
                sb_g[ihal] += sb_g_tot
                sb_r[ihal] += sb_r_tot
                sb_i[ihal] += sb_i_tot
                central_u[ihal] += central_sb_u
                central_g[ihal] += central_sb_g
                central_r[ihal] += central_sb_r
                central_i[ihal] += central_sb_i
                ur_color[ihal] += ur
                gr_color[ihal] += gr
                if n>0.:
                    sersic_index_lum[ihal] += n
                    sersic_counter[ihal] += 1
                
                # SECOND IMAGES AND SPECTRA IN FILES
                # Saving spectra of the whole galaxy (not each cell):
                file_fluxtot_y = 'calipso_output/spectra_y/fluxtot'+string_it+'ih'+str(ihal+1)
                np.save(file_fluxtot_y, fluxtot)
                # Saving image (flux, mag and SB of each cell):
                file_image_y = 'calipso_output/images_y/image'+string_it+'ih'+str(ihal+1)
                file_image_array_y = np.zeros((ncell*ncell+1, N_F*3))
                file_image_array_y[0, 0] = ncell #HEADER
                file_image_array_y[0, 1] = ncell
                file_image_array_y[0, 2] = LL
                for i_f in range(N_F):
                    file_image_array_y[1:, i_f] = magf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_y[1:, N_F+i_f] = fluxf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_y[1:, 2*N_F+i_f] = sbf[:,:,i_f].reshape((ncell*ncell))
                np.save(file_image_y, file_image_array_y)

                #########################################################################################
                #########################################################################################
                # CALCULATION IN YZ PLANE
                #########################################################################################
                #########################################################################################

                tam_i = which_cell_y
                tam_j = which_cell_z
                vel_LOS = velx
                effective_radius = rad05_x[ihal]*1e3

                (   fluxtot,
                    lumtotu, lumtotg, lumtotr, lumtoti,
                    sb_u_tot, sb_g_tot, sb_r_tot, sb_i_tot,
                    central_sb_u, central_sb_g, central_sb_r, central_sb_i,
                    gr, ur,
                    sbf, magf, fluxf    )  = pycalipso.main(calipso_input, star_particle_data,
                                                            ncell, vel_LOS, tam_i, tam_j, effective_radius)
                
                # SÉRSIC INDEX WITH LIGHT (g filter SDSS)
                # FIRST: LINEAR INTERPOLATION OF THE FLUX IN THE GRID

                flux_interp = RegularGridInterpolator((grid_centers, grid_centers), fluxf[:,:, 1], method='linear')
                flux_interpolated = flux_interp((x_meshgrid_interp, y_meshgrid_interp))

                # COMPARE INTERPOLATED FLUX WITH THE ORIGINAL ONE
                n, eps = galaxy_image_fit.photutils_fit(effective_radius, 0., 2*effective_radius, res_interp, flux_2D = flux_interpolated)

                # SAVE DATA
                # FIRST, THE CATALOGUE
                lum_u[ihal] += lumtotu
                lum_g[ihal] += lumtotg
                lum_r[ihal] += lumtotr
                lum_i[ihal] += lumtoti
                sb_u[ihal] += sb_u_tot
                sb_g[ihal] += sb_g_tot
                sb_r[ihal] += sb_r_tot
                sb_i[ihal] += sb_i_tot
                central_u[ihal] += central_sb_u
                central_g[ihal] += central_sb_g
                central_r[ihal] += central_sb_r
                central_i[ihal] += central_sb_i
                ur_color[ihal] += ur
                gr_color[ihal] += gr
                if n>0.:
                    sersic_index_lum[ihal] += n
                    sersic_counter[ihal] += 1

                # SECOND IMAGES AND SPECTRA IN FILES
                # Saving spectra of the whole galaxy (not each cell):
                file_fluxtot_z = 'calipso_output/spectra_z/fluxtot'+string_it+'ih'+str(ihal+1)
                np.save(file_fluxtot_z, fluxtot)
                # Saving image (flux, mag and SB of each cell):
                file_image_z = 'calipso_output/images_z/image'+string_it+'ih'+str(ihal+1)
                file_image_array_z = np.zeros((ncell*ncell+1, N_F*3))
                file_image_array_z[0, 0] = ncell #HEADER
                file_image_array_z[0, 1] = ncell
                file_image_array_z[0, 2] = LL
                for i_f in range(N_F):
                    file_image_array_z[1:, i_f] = magf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_z[1:, N_F+i_f] = fluxf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_z[1:, 2*N_F+i_f] = sbf[:,:,i_f].reshape((ncell*ncell))
                np.save(file_image_z, file_image_array_z)            

                ############################################################################################################
                ############################################################################################################
                ############################################################################################################
                ############################################################################################################

                # AVERAGE OVER 3 PROJECTIONS
                lum_u[ihal] /= 3.
                lum_g[ihal] /= 3.
                lum_r[ihal] /= 3.
                lum_i[ihal] /= 3.
                sb_u[ihal] /= 3.
                sb_g[ihal] /= 3.
                sb_r[ihal] /= 3.
                sb_i[ihal] /= 3.
                central_u[ihal] /= 3.
                central_g[ihal] /= 3.
                central_r[ihal] /= 3.
                central_i[ihal] /= 3.
                ur_color[ihal] /= 3.
                gr_color[ihal] /= 3.
                if sersic_counter[ihal]>0:
                    sersic_index_lum[ihal] /= sersic_counter[ihal]
                else:
                    sersic_index_lum[ihal] = np.nan

    else:
        print('No stars found!!')
        groups = []
        num_halos = 0
        n_part_in_halos = 0


    ############################################################
    #################       SAVE DATA           ################
    ############################################################

    print()
    print('Saving data..')
    print()
    print()
    print()

    iteration_data = {}
    iteration_data['nhal'] = num_halos
    iteration_data['nparhal'] = n_part_in_halos
    iteration_data['it_halma'] = it_count + 1
    iteration_data['it_masclet'] = iteration
    iteration_data['t'] = cosmo_time
    iteration_data['z'] = zeta

    haloes=[]
    for ih in range(num_halos):
        halo = {}
        halo['id'] = ih+1
        halo['partNum'] = num_particles[ih]
        halo['M'] = masses[ih]
        halo['Mv'] = 0.
        halo['Mgas'] = gas_masses[ih]
        halo['fcold'] = cold_bound_gas_fractions[ih]
        halo['Mcoldgas'] = cold_unbound_gas_masses[ih]
        halo['Mhotgas'] = hot_unbound_gas_masses[ih]
        halo['Msfr'] = star_formation_masses[ih]
        halo['Rmax'] = rmax[ih]*rete*1e3 #kpc
        halo['R'] = rad05[ih]*rete*1e3
        halo['R_1d'] = (rad05_x[ih] + rad05_y[ih] + rad05_z[ih])/3 * rete * 1e3
        halo['R_1dx'] = rad05_x[ih]*rete*1e3
        halo['R_1dy'] = rad05_y[ih]*rete*1e3
        halo['R_1dz'] = rad05_z[ih]*rete*1e3
        halo['sigma_v'] = sig_3D[ih]
        halo['sigma_v_1d'] = (sig_1D_x[ih] + sig_1D_y[ih] + sig_1D_z[ih])/3
        halo['sigma_v_1dx'] = sig_1D_x[ih]
        halo['sigma_v_1dy'] = sig_1D_y[ih]
        halo['sigma_v_1dz'] = sig_1D_z[ih]
        halo['L'] = specific_angular_momentum[ih]*rete*1e3 # kpc km/s
        halo['xcm'] = density_peak_x[ih]*1e3 #kpc
        halo['ycm'] = density_peak_y[ih]*1e3
        halo['zcm'] = density_peak_z[ih]*1e3
        halo['vx'] = velocities_x[ih]
        halo['vy'] = velocities_y[ih]
        halo['vz'] = velocities_z[ih]
        halo['father1'] = pro1[ih]
        halo['father2'] = pro2[ih]
        halo['nmerg'] = n_mergers[ih]
        halo['mergType'] = merger_type[ih]
        halo['age_m'] = stellar_age_mass[ih]
        halo['age'] = stellar_ages[ih]
        halo['Z_m'] = metallicities[ih]
        halo['Z'] = metallicities_mass[ih]
        #NEW ADDED
        halo['Vsigma'] = vsigma[ih]
        halo['lambda'] = lambda_ensellem[ih]
        halo['v_TF'] = tully_fisher_velocities[ih]
        halo['a'] = s_axis_major[ih]*rete*1e3
        halo['b'] = s_axis_intermediate[ih]*rete*1e3
        halo['c'] = s_axis_minor[ih]*rete*1e3
        halo['sersic'] = sersic_indices[ih]
        #CALIPSO
        halo['lum_u'] = lum_u[ih]
        halo['lum_g'] = lum_g[ih]
        halo['lum_r'] = lum_r[ih]
        halo['lum_i'] = lum_i[ih]
        halo['sb_u'] = central_u[ih]
        halo['sb_g'] = central_g[ih]
        halo['sb_r'] = central_r[ih]
        halo['sb_i'] = central_i[ih]
        halo['ur_color'] = ur_color[ih]
        halo['gr_color'] = gr_color[ih]
        halo['sersic_lum'] = sersic_index_lum[ih]
        haloes.append(halo)

    total_iteration_data.append(iteration_data)
    total_halo_data.append(haloes)


    ###########################################
    ### SAVING PARTICLES in .npy (python friendly)
    if WRITE_PARTICLES and len(data)>0:
        string_it = f'{iteration:05d}'
        new_groups = np.array(new_groups, dtype=object)
        new_groups = new_groups[argsort_part]
        all_particles_in_haloes = np.concatenate(new_groups)
        np.save('halo_particles/halotree'+string_it+'.npy', all_particles_in_haloes)
    ###########################################


write_to_HALMA_catalogue(total_iteration_data, total_halo_data, name = CATALOGUE_NAME)


########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ##########    END     ########## ##########
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
########## ########## ########## ########## ########## 
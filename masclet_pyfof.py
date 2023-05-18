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
    rps_flag = bool(int(f.readline()))
    f.readline()
    catalogue_name = f.readline()[:-1]
    f.readline()
    ncore = int(f.readline())
    f.readline()
    f.readline() #CALIPSO BLOCK
    f.readline()
    calipso_flag = bool(int(f.readline()))
    f.readline()
    zp_5556 = float(f.readline())
    f.readline()
    SB_lim = float(f.readline())
    f.readline()
    clight = float(f.readline())
    f.readline() 
    usun, gsun, rsun, isun = np.array(f.readline().split()[0].split(','), dtype = np.float64)
    f.readline()
    metsun = float(f.readline())
    f.readline()
    lstart, lend = np.array(f.readline().split()[0].split(','), dtype = np.float64)





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

dirssp = 'E-MILES'

########## ########## ########## ########## ##########
### CALIPSO BLOCK
########## ########## ########## ########## ##########
if calipso_flag:
    print('*************************************')
    print('Before FoF, reading calipso files..')
    wavelenghts, SSP, age_span, Z_span, MH_span, nages, nZ, nw = pycalipso.readSSPfiles(dirssp, lstart, lend)
    nf, nlf, wf, rf = pycalipso.readFilters()
    wv, fv, nv = pycalipso.readVega(zp_5556)
    print('.. done')
    dlum=1.e-5 #10 pc --> absolute mag
    magssp = np.zeros((nages,nZ,nf))
    fluxssp = np.zeros((nages,nZ,nf))
    lumu = np.zeros((nages,nZ))
    lumg = np.zeros((nages,nZ))
    lumr = np.zeros((nages,nZ))
    lumi = np.zeros((nages,nZ))
    # HERE WE CALCULATE ABSOLUTE MAGNITUDES (SINCE DLUM = 1 PC) AND THUS LUMINOSITIES
    # Calculating luminosity in u,g,r filters of each SSP
    for iage in range(nages):
        for iZ in range(nZ):
            pycalipso.mag_v1_0(wavelenghts, SSP[iage, iZ, :], nw, magssp[iage, iZ, :],
                                fluxssp[iage, iZ, :], nf, nlf, wf, rf, wv, fv, nv, dlum)
            
            lumu[iage,iZ]=10.**(-0.4*(magssp[iage,iZ, 0]-usun)) #luminosity in u filter (U band more or less)
            lumg[iage,iZ]=10.**(-0.4*(magssp[iage,iZ, 1]-gsun)) #luminosity in g filter (B band more or less)
            lumr[iage,iZ]=10.**(-0.4*(magssp[iage,iZ, 2]-rsun)) #luminosity in r filter (R band more or less)
            lumi[iage,iZ]=10.**(-0.4*(magssp[iage,iZ, 3]-isun)) #luminosity in i filter (I band more or less)

            
    closest_start = np.abs(lstart/wavelenghts - 1)
    closest_end = np.abs(lend/wavelenghts - 1)
    istart = np.argmin(closest_start) 
    iend = np.argmin(closest_end) 

    print('Spectrum will be written in the range:', wavelenghts[istart], wavelenghts[iend], '(A)')
    disp=wavelenghts[2]-wavelenghts[1] #resolución espectral
    print('*************************************')
    print()
    print()
    print()

########## ########## ########## ########## ##########
########## ########## ########## ########## ##########


path_results = 'simu_masclet'

re0 = 1.0  #factor de escala a z = 0

print('****************************************************')
print('******************** MASCLET pyfof *****************')
print('****************************************************')
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print('CHECK!!!! --> LINKING LENGHT (kpc)', ll*1e3)
print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
print()

###############################
if __name__ == '__main__':
    numba.set_num_threads(ncore)
###############################

print('----> Available CPU cores:', numba.config.NUMBA_NUM_THREADS)
print('----> Using', numba.get_num_threads(), 'cores')

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
    
    #READ GRID
    grid_data = read_masclet.read_grids(iteration, path=path_results, parameters_path=path_results, digits=5, 
                                                read_general=True, read_patchnum=True, read_dmpartnum=False,
                                                read_patchcellextension=True, read_patchcellposition=True, read_patchposition=True,
                                                read_patchparent=False)
    cosmo_time = grid_data[1]
    zeta = grid_data[4]
    rete = re0 / (1.0+zeta)
    
    dt = 0.
    if it_count>0:
        dt = (cosmo_time - total_iteration_data[it_count-1]['t'])*units.time_to_yr/1e9

    print('Cosmo time (Gyr):', cosmo_time*units.time_to_yr/1e9)
    print('Redshift (z):', zeta)

    #READ GAS IF RPS
    if rps_flag:
        print('RPS == True !!!! ')
        print('     Opening grid, clus and DM files')

        #READ CLUS
        gas_data = read_masclet.read_clus(iteration, path=path_results, parameters_path=path_results, digits=5, max_refined_level=1000, output_delta=True, 
                                          output_v=True, output_pres=False, output_pot=False, output_opot=False, output_temp=True, output_metalicity=False,
                                          output_cr0amr=True, output_solapst=True, is_mascletB=False, output_B=False, is_cooling=True, verbose=False)


        #READ DM
        masclet_dm_data = read_masclet.read_cldm(iteration, path = path_results, parameters_path=path_results, 
                                                    digits=5, max_refined_level=1000, output_deltadm = False,
                                                    output_position=True, output_velocity=False, output_mass=True)


    masclet_st_data = read_masclet.read_clst(iteration, path = path_results, parameters_path=path_results, 
                                                    digits=5, max_refined_level=1000, 
                                                    output_deltastar=False, verbose=False, output_position=True, 
                                                    output_velocity=True, output_mass=True, output_time=True,
                                                    output_metalicity=True, output_id=True, are_BH = not old_masclet)

    st_x = masclet_st_data[0]
    st_y = masclet_st_data[1]
    st_z = masclet_st_data[2]
    st_vx = masclet_st_data[3]*clight #in km/s
    st_vy = masclet_st_data[4]*clight #in km/s
    st_vz = masclet_st_data[5]*clight #in km/s
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
        NCELL = []
        new_groups = []
        which_cell_list_x = [] 
        which_cell_list_y = []
        which_cell_list_z = []
        for ihal in tqdm(range(len(groups))):
            part_list = np.array(groups[ihal])
            cx, cy, cz, M = halo_properties.center_of_mass(part_list, st_x, st_y, st_z, st_mass)
            M0 = M
            #Create 3D grid with cellsize 2*LL for cleaning
            RRHH = halo_properties.furthest_particle(cx, cy, cz, part_list, st_x, st_y, st_z)
            grid = np.arange(-(RRHH+ll), RRHH+ll, 2*ll)
            n_cell = len(grid)
            vcm_cell = np.zeros((n_cell, n_cell, n_cell, 3))
            mass_cell = np.zeros((n_cell, n_cell, n_cell))
            quantas_cell = np.zeros((n_cell, n_cell, n_cell))
            sig3D_cell = np.zeros((n_cell, n_cell, n_cell))
            #CALCULATE cell quantities
            vcm_cell, mass_cell, quantas_cell, sig3D_cell, which_cell_x, which_cell_y, which_cell_z = halo_properties.calc_cell(cx, cy, cz, grid, part_list, st_x, 
                                                                                                                                st_y, st_z, st_vx, st_vy, 
                                                                                                                                st_vz, st_mass, vcm_cell, 
                                                                                                                                mass_cell, quantas_cell, 
                                                                                                                                sig3D_cell)
            
            #DO THE CLEANING: Right CM, Mass and Furthest particle
            cx, cy, cz, M, RRHH, control, which_cell_x, which_cell_y, which_cell_z = halo_properties.clean_cell(cx, cy, cz, M, RRHH, grid, part_list, st_x, st_y, 
                                                                                                                st_z, st_vx, st_vy, st_vz, st_mass, vcm_cell,
                                                                                                                mass_cell, quantas_cell, sig3D_cell, ll, q_fil, sig_fil,
                                                                                                                which_cell_x, which_cell_y, which_cell_z)
            #vx, vy, vz = CM_velocity(M, part_list[control.astype(bool)], st_vx, st_vy, st_vz, st_mass)
            #FASTER CLEANING --> ESCAPE VELOCITY IN ITS POSITION, CONSIDERING MASS WITHOUT CLEANING M0 and SPHERICAL SIMMETRY
            # factor_v = 4
            # control = escape_velocity_cleaning(cx, cy, cz, vx, vy, vz, M0, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, control, factor_v)
            #CLEANING DONE
            control = control.astype(bool)
            npart = len(part_list[control])
            cx, cy, cz, M = halo_properties.center_of_mass(part_list[control], st_x, st_y, st_z, st_mass)
            RRHH = halo_properties.furthest_particle(cx, cy, cz, part_list[control], st_x, st_y, st_z)
            if npart > minp:
                CX.append(cx)
                CY.append(cy)
                CZ.append(cz)
                MM.append(M)
                RMAX.append(RRHH)
                NPART.append(npart)
                NCELL.append(n_cell)
                new_groups.append(part_list[control])
                which_cell_list_x.append(which_cell_x)
                which_cell_list_y.append(which_cell_y)
                which_cell_list_z.append(which_cell_z)

        CX = np.array(CX)
        CY = np.array(CY)
        CZ = np.array(CZ)
        MM = np.array(MM)
        RMAX = np.array(RMAX)
        NPART = np.array(NPART).astype(np.int32)
        NHAL = len(new_groups)
        NPARTHAL = np.sum(NPART)
        NCELL = np.array(NCELL).astype(np.int32)
        which_cell_list_x = np.array(which_cell_list_x, dtype=object)
        which_cell_list_y = np.array(which_cell_list_y, dtype=object)
        which_cell_list_z = np.array(which_cell_list_z, dtype=object)

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
        SAXISMAJOR = np.zeros(NHAL)
        SAXISINTERMEDIATE = np.zeros(NHAL)
        SAXISMINOR = np.zeros(NHAL)
        SERSIC = np.zeros(NHAL)
        MGAS = np.zeros(NHAL)
        FRACCOLD = np.zeros(NHAL)
        MRPS_COLD = np.zeros(NHAL)
        MRPS_HOT = np.zeros(NHAL)
        for ihal in tqdm(range(NHAL)):
            part_list = new_groups[ihal]
            PEAKX[ihal], PEAKY[ihal], PEAKZ[ihal] = halo_properties.density_peak(part_list, st_x, st_y, st_z, st_mass, ll)
            cx = CX[ihal]
            cy = CY[ihal]
            cz = CZ[ihal]
            M = MM[ihal]
            RAD05[ihal] = halo_properties.half_mass_radius(cx, cy, cz, M, part_list, st_x, st_y, st_z, st_mass)
            RAD05_x[ihal], RAD05_y[ihal], RAD05_z[ihal] = halo_properties.half_mass_radius_proj(cx, cy, cz, M, part_list, st_x, st_y, st_z, st_mass)
            VX[ihal], VY[ihal], VZ[ihal] = halo_properties.CM_velocity(M, part_list, st_vx, st_vy, st_vz, st_mass)
            V_TF[ihal] = halo_properties.tully_fisher_velocity(part_list, cx, cy, cz, st_x, st_y, st_z, VX[ihal], VY[ihal], VZ[ihal], st_vx, st_vy, st_vz, st_mass, RAD05[ihal])
            JX[ihal], JY[ihal], JZ[ihal] = halo_properties.angular_momentum(M, part_list, st_x, st_y, st_z, st_vx, st_vy, st_vz, st_mass, cx, cy, cz, VX[ihal], VY[ihal], VZ[ihal])
            J[ihal] = (JX[ihal]**2 + JY[ihal]**2 + JZ[ihal]**2)**0.5
            SAXISMAJOR[ihal], SAXISINTERMEDIATE[ihal], SAXISMINOR[ihal] = halo_properties.halo_shape(part_list, st_x, st_y, st_z, st_mass, cx, cy, cz, RAD05[ihal])
            #CARE HERE, RMAX IS CALCULATED WITH RESPECT TO THE CENTER OF MASS, NOT THE DENSITY PEAK
            # and RAD05 is calculated with respect to the density peak
            SERSIC[ihal] = halo_properties.simple_sersic_index(part_list, st_x, st_y, st_z, CX[ihal], CY[ihal], CZ[ihal], RAD05[ihal])
            if it_count > 0:
                MSFR[ihal] = halo_properties.star_formation(part_list, st_mass, st_age, cosmo_time*units.time_to_yr/1e9, dt)

            SIG_3D[ihal] = halo_properties.sigma_effective(part_list, RAD05[ihal], st_x, st_y, st_z, st_vx, st_vy, st_vz, cx, cy, cz, VX[ihal], VY[ihal], VZ[ihal])
            grid = np.arange(-(RMAX[ihal]+ll), RMAX[ihal]+ll, 2*ll) #centers of the cells
            n_cell = len(grid)
            SIG_1D_x[ihal], SIG_1D_y[ihal], SIG_1D_z[ihal], VSIGMA[ihal], LAMBDA[ihal] = halo_properties.sigma_projections(grid, n_cell, part_list, st_x, st_y, st_z, 
                                                                                                                           st_vx, st_vy, st_vz, st_mass, cx, cy, cz, 
                                                                                                                           RAD05_x[ihal], RAD05_y[ihal], RAD05_z[ihal], ll)
            EDAD[ihal], EDAD_MASS[ihal], MET[ihal], MET_MASS[ihal] = halo_properties.avg_age_metallicity(part_list, st_age, st_met, st_mass, cosmo_time*units.time_to_yr/1e9)

            # RPS EFFECTS
            if rps_flag:
                Rrps = 2*RAD05[ihal]
                MGAS[ihal], FRACCOLD[ihal], MRPS_COLD[ihal], MRPS_HOT[ihal] = halo_gas.RPS(rete, L, nx, grid_data, gas_data, 
                                                                                     masclet_dm_data, masclet_st_data, cx, cy, cz, 
                                                                                     VX[ihal], VY[ihal], VZ[ihal], Rrps)
                
        ############################################################################################################

        if len(new_groups)>0:
            print()
            print(f'CHECK min, max in R_05 {np.min(RAD05)*rete*1e3:.2f} {np.max(RAD05)*rete*1e3:.2f}')
            print(f'CHECK min, max in RMAX {np.min(RMAX)*rete*1e3:.2f} {np.max(RMAX)*rete*1e3:.2f}')
            print(f'CHECK min, max in stellar mass {np.min(MM):.2e} {np.max(MM):.2e}')
            print(f'CHECK min, max in gas mass {np.min(MGAS):.2e} {np.max(MGAS):.2e}')
            print(f'CHECK min, max in J {np.min(J)*rete*1e3:.2f} {np.max(J)*rete*1e3:.2f}')
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

        print('-------> DONE <-------')

        ###########################################################
        ####### SORTING BY NUMBER OF PARTICLES ##################
        ###########################################################

        argsort_part = np.flip(np.argsort(NPART)) #sorted descending

        NPART = NPART[argsort_part]
        NCELL = NCELL[argsort_part]
        which_cell_list_x = which_cell_list_x[argsort_part]
        which_cell_list_y = which_cell_list_y[argsort_part]
        which_cell_list_z = which_cell_list_z[argsort_part]
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
        SAXISMAJOR = SAXISMAJOR[argsort_part]
        SAXISINTERMEDIATE = SAXISINTERMEDIATE[argsort_part]
        SAXISMINOR = SAXISMINOR[argsort_part]
        SERSIC = SERSIC[argsort_part]
        MGAS = MGAS[argsort_part]
        FRACCOLD = FRACCOLD[argsort_part]
        MRPS_COLD = MRPS_COLD[argsort_part]
        MRPS_HOT = MRPS_HOT[argsort_part]

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

        ############################################################################################################
        # CALIPSO BLOCK
        ############################################################################################################

        #NOTES:
        # - SSP is in Lo A^-1 Mo^-1
        # - The luminosity weights are in g-band (see pycalipso.make_light)
        # - Care should be taken in the spectral range in which calculations are done. The filter ranges must be inside.

        LUMU = np.zeros(NHAL)
        LUMG = np.zeros(NHAL)
        LUMR = np.zeros(NHAL)
        LUMI = np.zeros(NHAL)
        SBU = np.zeros(NHAL)
        SBG = np.zeros(NHAL)
        SBR = np.zeros(NHAL)
        SBI = np.zeros(NHAL)
        UR_COLOR = np.zeros(NHAL)
        GR_COLOR = np.zeros(NHAL)
        SERSIC_LUM = np.zeros(NHAL)
        sersic_counter = np.zeros(NHAL)

        if calipso_flag:
            
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
            cosmo = cosmology.FlatLambdaCDM(H0=ache*100, Om0=omega0)
            dlum = abs(cosmo.luminosity_distance(zeta).value)
            daa = abs(cosmo.angular_diameter_distance(zeta).value) # if zeta < 0, not defined
            arcsec2kpc=daa*1.e3*(acos(-1)/180.)/3600.
            # Area of pixel in arcsec^2
            area_com=(ll*ll)
            area_pys=area_com/((1.+zeta)*(1.+zeta))
            area_arc=area_pys/(arcsec2kpc*arcsec2kpc)
            print('zeta, dist_lum, arcsec2kpc:', zeta, dlum, arcsec2kpc)
            print()
            print('Calculating SEDs of the galaxies')
            for ihal in tqdm(range(NHAL)):
                npart = NPART[ihal] # number of particles in the halo
                halo = new_groups_calipso[ihal] # halo particle indices for array slicing

                mass = st_mass[halo] #mass of halo particles
                met = st_met[halo] #metallicity of halo particles
                age = st_age[halo] #age of halo particles
                x = (st_x[halo] - CX[ihal])*1e3 #relative position of halo particles
                y = (st_y[halo] - CY[ihal])*1e3 # in kpc
                z = (st_z[halo] - CZ[ihal])*1e3
                velx = st_vx[halo] - VX[ihal] 
                vely = st_vy[halo] - VY[ihal] #LOS velocity of halo particles
                velz = st_vz[halo] - VZ[ihal]

                # # CREATE THE GRID with maximum resolution applied to the halo in the simulation
                # box = [CX[ihal] - RMAX[ihal], CX[ihal] + RMAX[ihal], CY[ihal] - RMAX[ihal], 
                #        CY[ihal] + RMAX[ihal], CZ[ihal] - RMAX[ihal], CZ[ihal] + RMAX[ihal]]
                
                # which_patches = tools.which_patches_inside_box(box, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, L, nx)
                # patch_level = tools.create_vector_levels(npatch)
                # max_level = np.max(patch_level[which_patches])
                # res_minimum_cell = L / nx / 2**max_level * 1e3 # maximum resolution of the simulation in kpc

                res = ll*1e3 #2*res_minimum_cell # resolution of the grid in kpc
                grid_edges = np.arange(-RMAX[ihal]*1e3 - res, RMAX[ihal]*1e3 + res, res)
                grid_centers = (grid_edges[1:] + grid_edges[:-1]) / 2
                ncell = len(grid_centers)  # number of cells in each direction
                which_cell_x, which_cell_y, which_cell_z = pycalipso.put_particles_in_grid(grid_centers, x, y, z)

                # finner grid to interpolate
                res_interp =  L / nx / 2**9 * 1e3
                grid_interp_edges = np.arange(grid_centers[0], grid_centers[-1], res_interp)
                grid_interp = (grid_interp_edges[1:] + grid_interp_edges[:-1]) / 2
                X_grid_interp, Y_grid_interp = np.meshgrid(grid_interp, grid_interp)

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
                # Mean cell properties in each projection
                num_cell, mass_cell, vel_malla, sig_cell, twm, tmed, Zwm, Zmed = pycalipso.mean_mesh2D(ncell, ncell, npart, 
                                                                                                       mass, met, age, 
                                                                                                       vel_LOS, tam_i, tam_j)
                # Finding fluxes in each cell and luminosity weighted quantities
                (flux_cell, flux_cell_sig, fluxtot, vell_malla, 
                sigl_malla, lum_malla, twl, Zwl, vell2_malla) = pycalipso.make_light(npart, mass, age, 
                                                                                    met, wavelenghts, 
                                                                                    SSP, age_span, Z_span,
                                                                                    nw, istart, iend, disp, 
                                                                                    lumg, tam_i, tam_j, 
                                                                                    vel_LOS, ncell, ncell, 
                                                                                    mass_cell, clight)
                
                # Magnitudes and fluxes through filters in each cell. Images.
                # First: total quantities
                fmag_tot = np.zeros(nf)
                flux_tot = np.zeros(nf)
                pycalipso.mag_v1_0(wavelenghts, fluxtot, nw, fmag_tot, flux_tot, nf, nlf, wf, rf, wv, fv, nv, dlum) 
                
                lumtotu = flux_tot[0]
                lumtotg = flux_tot[1]
                lumtotr = flux_tot[2]
                lumtoti = flux_tot[3]

                area_halo_com = np.pi*RAD05[ihal]**2 # comoving area of the halo in kpc^2
                area_halo_pys = area_halo_com/((1.+zeta)*(1.+zeta)) # physical area of the halo in kpc^2
                area_halo_arc = area_halo_pys/(arcsec2kpc*arcsec2kpc) # physical area of the halo in arcsec^2

                SBu_tot = fmag_tot[0]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 u filter
                SBg_tot = fmag_tot[1]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 g filter
                SBr_tot = fmag_tot[2]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 r filter
                SBi_tot = fmag_tot[3]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 i filter

                gr = fmag_tot[1]-fmag_tot[2] # color g-r
                ur = fmag_tot[0]-fmag_tot[2] # color u-r

                #Second: same quantities but for each cell and "visibility"
                SBf, magf, fluxf = pycalipso.magANDfluxes(wavelenghts, nw, nf, nlf, wf, rf, wv, fv, nv,
                                                           dlum, ncell, ncell, flux_cell, area_arc)
                SBf[fluxf==0.] = np.nan 
                magf[fluxf==0.] = np.nan

                # Third: same quantities but for each cell and "visibility" CONSIDERING DOPPLER SHIFT
                # SBfdoppler, magfdoppler, fluxfdoppler = pycalipso.magANDfluxes(wavelenghts, nw, nf, nlf, wf, rf, wv, fv, nv,
                #                                                                 dlum, ncell, ncell, flux_cell_sig, area_arc)
                # SBfdoppler[fluxfdoppler==0.] = np.nan
                # magfdoppler[fluxfdoppler==0.] = np.nan

                # SÉRSIC INDEX WITH LIGHT (g filter SDSS)
                # FIRST: LINEAR INTERPOLATION OF THE FLUX IN THE GRID

                flux_interp = RegularGridInterpolator((grid_centers, grid_centers), fluxf[:,:, 1], method='linear')
                flux_interpolated = flux_interp((X_grid_interp, Y_grid_interp))

                # COMPARE INTERPOLATED FLUX WITH THE ORIGINAL ONE
                n, eps = galaxy_image_fit.photutils_fit(RAD05[ihal]*1e3, 0., 2*RAD05[ihal]*1e3, res_interp, flux_2D = flux_interpolated)

                # SAVE DATA
                # FIRST, THE CATALOGUE
                LUMU[ihal] += lumtotu
                LUMG[ihal] += lumtotg
                LUMR[ihal] += lumtotr
                LUMI[ihal] += lumtoti
                SBU[ihal] += SBu_tot
                SBG[ihal] += SBg_tot
                SBR[ihal] += SBr_tot
                SBI[ihal] += SBi_tot
                UR_COLOR[ihal] += ur
                GR_COLOR[ihal] += gr
                if n>0.:
                    SERSIC_LUM[ihal] += n
                    sersic_counter[ihal] += 1
                
                # SECOND IMAGES AND SPECTRA IN FILES
                # Saving spectra of the whole galaxy (not each cell):
                file_fluxtot_x = 'calipso_output/spectra_x/fluxtot'+string_it+'ih'+str(ihal+1)
                np.save(file_fluxtot_x, fluxtot)
                # Saving image (flux, mag and SB of each cell):
                file_image_x = 'calipso_output/images_x/image'+string_it+'ih'+str(ihal+1)
                file_image_array_x = np.zeros((ncell*ncell+1, nf*3))
                file_image_array_x[0, 0] = ncell #HEADER
                file_image_array_x[0, 1] = ncell
                file_image_array_x[0, 2] = ll
                for i_f in range(nf):
                    file_image_array_x[1:, i_f] = magf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_x[1:, nf+i_f] = fluxf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_x[1:, 2*nf+i_f] = SBf[:,:,i_f].reshape((ncell*ncell))
                np.save(file_image_x, file_image_array_x)

                #########################################################################################
                #########################################################################################
                # CALCULATION IN XZ PLANE
                #########################################################################################
                #########################################################################################

                tam_i = which_cell_x
                tam_j = which_cell_z
                vel_LOS = vely
                # Mean cell properties in each projection
                num_cell, mass_cell, vel_malla, sig_cell, twm, tmed, Zwm, Zmed = pycalipso.mean_mesh2D(ncell, ncell, npart, 
                                                                                                       mass, met, age, 
                                                                                                       vel_LOS, tam_i, tam_j)
                # Finding fluxes in each cell and luminosity weighted quantities
                (flux_cell, flux_cell_sig, fluxtot, vell_malla, 
                sigl_malla, lum_malla, twl, Zwl, vell2_malla) = pycalipso.make_light(npart, mass, age, 
                                                                                    met, wavelenghts, 
                                                                                    SSP, age_span, Z_span,
                                                                                    nw, istart, iend, disp, 
                                                                                    lumg, tam_i, tam_j, 
                                                                                    vel_LOS, ncell, ncell, 
                                                                                    mass_cell, clight)                
                # Magnitudes and fluxes through filters in each cell. Images.
                # First: total quantities
                fmag_tot = np.zeros(nf)
                flux_tot = np.zeros(nf)
                pycalipso.mag_v1_0(wavelenghts, fluxtot, nw, fmag_tot, flux_tot, nf, nlf, wf, rf, wv, fv, nv, dlum) 
                
                lumtotu = flux_tot[0]
                lumtotg = flux_tot[1]
                lumtotr = flux_tot[2]
                lumtoti = flux_tot[3]

                area_halo_com = np.pi*RAD05[ihal]**2 # comoving area of the halo in kpc^2
                area_halo_pys = area_halo_com/((1.+zeta)*(1.+zeta)) # physical area of the halo in kpc^2
                area_halo_arc = area_halo_pys/(arcsec2kpc*arcsec2kpc) # physical area of the halo in arcsec^2

                SBu_tot = fmag_tot[0]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 u filter
                SBg_tot = fmag_tot[1]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 g filter
                SBr_tot = fmag_tot[2]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 r filter
                SBi_tot = fmag_tot[3]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 i filter

                gr = fmag_tot[1]-fmag_tot[2] # color g-r
                ur = fmag_tot[0]-fmag_tot[2] # color u-r

                #Second: same quantities but for each cell and "visibility"
                SBf, magf, fluxf = pycalipso.magANDfluxes(wavelenghts, nw, nf, nlf, wf, rf, wv, fv, nv,
                                                           dlum, ncell, ncell, flux_cell, area_arc)
                SBf[fluxf==0.] = np.nan 
                magf[fluxf==0.] = np.nan

                # Third: same quantities but for each cell and "visibility" CONSIDERING DOPPLER SHIFT
                # SBfdoppler, magfdoppler, fluxfdoppler = pycalipso.magANDfluxes(wavelenghts, nw, nf, nlf, wf, rf, wv, fv, nv,
                #                                                                 dlum, ncell, ncell, flux_cell_sig, area_arc)
                # SBfdoppler[fluxfdoppler==0.] = np.nan
                # magfdoppler[fluxfdoppler==0.] = np.nan

                # SÉRSIC INDEX WITH LIGHT (g filter SDSS)
                # FIRST: LINEAR INTERPOLATION OF THE FLUX IN THE GRID

                flux_interp = RegularGridInterpolator((grid_centers, grid_centers), fluxf[:,:, 1], method='linear')
                flux_interpolated = flux_interp((X_grid_interp, Y_grid_interp))

                # COMPARE INTERPOLATED FLUX WITH THE ORIGINAL ONE
                n, eps = galaxy_image_fit.photutils_fit(RAD05[ihal]*1e3, 0., 2*RAD05[ihal]*1e3, res_interp, flux_2D = flux_interpolated) 

                # SAVE DATA
                # FIRST, THE CATALOGUE
                LUMU[ihal] += lumtotu
                LUMG[ihal] += lumtotg
                LUMR[ihal] += lumtotr
                LUMI[ihal] += lumtoti
                SBU[ihal] += SBu_tot
                SBG[ihal] += SBg_tot
                SBR[ihal] += SBr_tot
                SBI[ihal] += SBi_tot
                UR_COLOR[ihal] += ur
                GR_COLOR[ihal] += gr
                if n>0.:
                    SERSIC_LUM[ihal] += n
                    sersic_counter[ihal] += 1
                
                # SECOND IMAGES AND SPECTRA IN FILES
                # Saving spectra of the whole galaxy (not each cell):
                file_fluxtot_y = 'calipso_output/spectra_y/fluxtot'+string_it+'ih'+str(ihal+1)
                np.save(file_fluxtot_y, fluxtot)
                # Saving image (flux, mag and SB of each cell):
                file_image_y = 'calipso_output/images_y/image'+string_it+'ih'+str(ihal+1)
                file_image_array_y = np.zeros((ncell*ncell+1, nf*3))
                file_image_array_y[0, 0] = ncell #HEADER
                file_image_array_y[0, 1] = ncell
                file_image_array_y[0, 2] = ll
                for i_f in range(nf):
                    file_image_array_y[1:, i_f] = magf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_y[1:, nf+i_f] = fluxf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_y[1:, 2*nf+i_f] = SBf[:,:,i_f].reshape((ncell*ncell))
                np.save(file_image_y, file_image_array_y)

                #########################################################################################
                #########################################################################################
                # CALCULATION IN YZ PLANE
                #########################################################################################
                #########################################################################################

                tam_i = which_cell_y
                tam_j = which_cell_z
                vel_LOS = velx
                # Mean cell properties in each projection
                num_cell, mass_cell, vel_malla, sig_cell, twm, tmed, Zwm, Zmed = pycalipso.mean_mesh2D(ncell, ncell, npart, 
                                                                                                       mass, met, age, 
                                                                                                       vel_LOS, tam_i, tam_j)
                # Finding fluxes in each cell and luminosity weighted quantities
                (flux_cell, flux_cell_sig, fluxtot, vell_malla, 
                sigl_malla, lum_malla, twl, Zwl, vell2_malla) = pycalipso.make_light(npart, mass, age, 
                                                                                    met, wavelenghts, 
                                                                                    SSP, age_span, Z_span,
                                                                                    nw, istart, iend, disp, 
                                                                                    lumg, tam_i, tam_j, 
                                                                                    vel_LOS, ncell, ncell, 
                                                                                    mass_cell, clight)                
                # Magnitudes and fluxes through filters in each cell. Images.
                # First: total quantities
                fmag_tot = np.zeros(nf)
                flux_tot = np.zeros(nf)
                pycalipso.mag_v1_0(wavelenghts, fluxtot, nw, fmag_tot, flux_tot, nf, nlf, wf, rf, wv, fv, nv, dlum) 
                
                lumtotu = flux_tot[0]
                lumtotg = flux_tot[1]
                lumtotr = flux_tot[2]
                lumtoti = flux_tot[3]
                
                area_halo_com = np.pi*RAD05[ihal]**2 # comoving area of the halo in kpc^2
                area_halo_pys = area_halo_com/((1.+zeta)*(1.+zeta)) # physical area of the halo in kpc^2
                area_halo_arc = area_halo_pys/(arcsec2kpc*arcsec2kpc) # physical area of the halo in arcsec^2

                SBu_tot = fmag_tot[0]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 u filter
                SBg_tot = fmag_tot[1]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 g filter
                SBr_tot = fmag_tot[2]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 r filter
                SBi_tot = fmag_tot[3]+2.5*log10(area_halo_arc) # total surface brightness in mag/arcsec^2 i filter

                gr = fmag_tot[1]-fmag_tot[2] # color g-r
                ur = fmag_tot[0]-fmag_tot[2] # color u-r

                #Second: same quantities but for each cell and "visibility"
                SBf, magf, fluxf = pycalipso.magANDfluxes(wavelenghts, nw, nf, nlf, wf, rf, wv, fv, nv,
                                                           dlum, ncell, ncell, flux_cell, area_arc)
                SBf[fluxf==0.] = np.nan 
                magf[fluxf==0.] = np.nan

                # Third: same quantities but for each cell and "visibility" CONSIDERING DOPPLER SHIFT
                # SBfdoppler, magfdoppler, fluxfdoppler = pycalipso.magANDfluxes(wavelenghts, nw, nf, nlf, wf, rf, wv, fv, nv,
                #                                                                 dlum, ncell, ncell, flux_cell_sig, area_arc)
                # SBfdoppler[fluxfdoppler==0.] = np.nan
                # magfdoppler[fluxfdoppler==0.] = np.nan

                # SÉRSIC INDEX WITH LIGHT (g filter SDSS)
                # FIRST: LINEAR INTERPOLATION OF THE FLUX IN THE GRID

                flux_interp = RegularGridInterpolator((grid_centers, grid_centers), fluxf[:,:, 1], method='linear')
                flux_interpolated = flux_interp((X_grid_interp, Y_grid_interp))

                # COMPARE INTERPOLATED FLUX WITH THE ORIGINAL ONE
                n, eps = galaxy_image_fit.photutils_fit(RAD05[ihal]*1e3, 0., 2*RAD05[ihal]*1e3, res_interp, flux_2D = flux_interpolated)

                # SAVE DATA
                # FIRST, THE CATALOGUE
                LUMU[ihal] += lumtotu
                LUMG[ihal] += lumtotg
                LUMR[ihal] += lumtotr
                LUMI[ihal] += lumtoti
                SBU[ihal] += SBu_tot
                SBG[ihal] += SBg_tot
                SBR[ihal] += SBr_tot
                SBI[ihal] += SBi_tot
                UR_COLOR[ihal] += ur
                GR_COLOR[ihal] += gr
                if n>0.:
                    SERSIC_LUM[ihal] += n
                    sersic_counter[ihal] += 1

                # SECOND IMAGES AND SPECTRA IN FILES
                # Saving spectra of the whole galaxy (not each cell):
                file_fluxtot_z = 'calipso_output/spectra_z/fluxtot'+string_it+'ih'+str(ihal+1)
                np.save(file_fluxtot_z, fluxtot)
                # Saving image (flux, mag and SB of each cell):
                file_image_z = 'calipso_output/images_z/image'+string_it+'ih'+str(ihal+1)
                file_image_array_z = np.zeros((ncell*ncell+1, nf*3))
                file_image_array_z[0, 0] = ncell #HEADER
                file_image_array_z[0, 1] = ncell
                file_image_array_z[0, 2] = ll
                for i_f in range(nf):
                    file_image_array_z[1:, i_f] = magf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_z[1:, nf+i_f] = fluxf[:,:,i_f].reshape((ncell*ncell))
                    file_image_array_z[1:, 2*nf+i_f] = SBf[:,:,i_f].reshape((ncell*ncell))
                np.save(file_image_z, file_image_array_z)            

                ############################################################################################################
                ############################################################################################################
                ############################################################################################################
                ############################################################################################################

                # AVERAGE OVER THE 3 PROJECTIONS
                LUMU[ihal] /= 3.
                LUMG[ihal] /= 3.
                LUMR[ihal] /= 3.
                LUMI[ihal] /= 3.
                SBU[ihal] /= 3.
                SBG[ihal] /= 3.
                SBR[ihal] /= 3.
                SBI[ihal] /= 3.
                UR_COLOR[ihal] /= 3.
                GR_COLOR[ihal] /= 3.
                if sersic_counter[ihal]>0:
                    SERSIC_LUM[ihal] /= sersic_counter[ihal]
                else:
                    SERSIC_LUM[ihal] = np.nan



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
        halo['Mgas'] = MGAS[ih]
        halo['fcold'] = FRACCOLD[ih]
        halo['Mcoldgas'] = MRPS_COLD[ih]
        halo['Mhotgas'] = MRPS_HOT[ih]
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
        halo['a'] = SAXISMAJOR[ih]*rete*1e3
        halo['b'] = SAXISINTERMEDIATE[ih]*rete*1e3
        halo['c'] = SAXISMINOR[ih]*rete*1e3
        halo['sersic'] = SERSIC[ih]
        #CALIPSO
        halo['lum_u'] = LUMU[ih]
        halo['lum_g'] = LUMG[ih]
        halo['lum_r'] = LUMR[ih]
        halo['lum_i'] = LUMI[ih]
        halo['sb_u'] = SBU[ih]
        halo['sb_g'] = SBG[ih]
        halo['sb_r'] = SBR[ih]
        halo['sb_i'] = SBI[ih]
        halo['ur_color'] = UR_COLOR[ih]
        halo['gr_color'] = GR_COLOR[ih]
        halo['sersic_lum'] = SERSIC_LUM[ih]
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
        np.save('halo_particles/halotree'+string_it+'.npy', all_particles_in_haloes)
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
'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used test the interpolation script nsm_rho_Ye_T_interpolator.py
The data files dump_00001000.h5 and grid.h5 should be in the same directory as this script.
'''

import numpy as np
import h5py
import nsm_grid_generator
import nsm_rho_Ye_T_interpolator
import time

# importing Jonah NSM data
dump_file = h5py.File('dump_00001000.h5','r')
grid_file = h5py.File('grid.h5','r')

# r_min_bin = 0
# r_max_bin = 6 
# theta_min_bin = 0
# theta_max_bin = 6
# phi_min_bin = 0
# phi_max_bin = 6

a = 0
b = 4
r_min_bin = a
r_max_bin = b
theta_min_bin = a
theta_max_bin = b
phi_min_bin = a
phi_max_bin = b

# getting NSM simulation grid in cartesian coordinates system
grid_cartesian = grid_file['Xcart'] * np.array(dump_file['L_unit']) # cm
NSM_grid_cartesian = grid_cartesian[ r_min_bin : r_max_bin , theta_min_bin : theta_max_bin , phi_min_bin : phi_max_bin , 1 : 4 ]

start = time.time()
indices, T_rho_Ye = nsm_rho_Ye_T_interpolator.interpolate_Ye_rho_T(NSM_grid_cartesian)
end = time.time()
print(f'Interpolation time = {end - start} s')

T = np.full( ( r_max_bin - r_min_bin , theta_max_bin - theta_min_bin , phi_max_bin - phi_min_bin ), 18081999 ) # array of size (ncellsx, ncellsy, ncellsz)
rho = np.full( ( r_max_bin - r_min_bin , theta_max_bin - theta_min_bin , phi_max_bin - phi_min_bin ), 18081999 ) # array of size (ncellsx, ncellsy, ncellsz)
Ye = np.full( ( r_max_bin - r_min_bin , theta_max_bin - theta_min_bin , phi_max_bin - phi_min_bin ), 18081999 ) # array of size (ncellsx, ncellsy, ncellsz)

T[indices] = T_rho_Ye[:,0]
rho[indices] = T_rho_Ye[:,1]
Ye[indices] = T_rho_Ye[:,2]

indices_no_interpolation_T = np.where( T == 18081999 )
indices_no_interpolation_rho = np.where( rho == 18081999 ) 
indices_no_interpolation_Ye = np.where( Ye == 18081999 )

print(f'{len(indices_no_interpolation_T[0])} T points were not interpolated')
print(f'{len(indices_no_interpolation_rho[0])} rho points were not interpolated')
print(f'{len(indices_no_interpolation_Ye[0])} Ye points were not interpolated')

# Getting index of Ye and rho in NSM data
index_Ye = np.where(dump_file['P'].attrs['vnams'] == 'Ye')[0][0]
index_rho = np.where(dump_file['P'].attrs['vnams'] == 'RHO')[0][0]

# Getting Ye, rho, T in NSM data
Ye_NSM = dump_file['P'][:,:,:,index_Ye] # no units
Ye_NSM = Ye_NSM[ r_min_bin : r_max_bin , theta_min_bin : theta_max_bin , phi_min_bin : phi_max_bin ]
rho_NSM = dump_file['P'][:,:,:,index_rho] * dump_file['RHO_unit'] # g / ccm
rho_NSM = rho_NSM[ r_min_bin : r_max_bin , theta_min_bin : theta_max_bin , phi_min_bin : phi_max_bin ]
T_NSM = dump_file['TEMP'] # MeV
T_NSM = T_NSM[ r_min_bin : r_max_bin , theta_min_bin : theta_max_bin , phi_min_bin : phi_max_bin ]

T_NSM  [indices_no_interpolation_T] = 18081999
rho_NSM[indices_no_interpolation_rho] = 18081999
Ye_NSM [indices_no_interpolation_Ye] = 18081999

print(f'rho test ---> Average relative error = { np.average( np.abs( rho - rho_NSM ) / rho_NSM ) }')
print(f'T test   ---> Average relative error = { np.average( np.abs( T   - T_NSM   ) / T_NSM   )}')
print(f'Ye test  ---> Average relative error = { np.average( np.abs( Ye  - Ye_NSM  ) / Ye_NSM  )}')

print(f'rho test ---> Maximum relative error = { np.max( np.abs( rho - rho_NSM ) / rho_NSM ) }')
print(f'T test   ---> Maximum relative error = { np.max( np.abs( T   - T_NSM   ) / T_NSM   )}')
print(f'Ye test  ---> Maximum relative error = { np.max( np.abs( Ye  - Ye_NSM  ) / Ye_NSM  )}')
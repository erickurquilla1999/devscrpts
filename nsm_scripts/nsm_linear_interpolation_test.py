'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used test the interpolation script nsm_rho_Ye_T_linear_interpolator.py
The data files dump_00001000.h5 and grid.h5 should be in the same directory as this script.
'''

import numpy as np
import h5py
import nsm_grid_generator
import nsm_rho_Ye_T_linear_interpolator
import time

# importing Jonah NSM data
dump_file = h5py.File('dump_00001000.h5','r')
grid_file = h5py.File('grid.h5','r')

# NSM grid will be split in radial, theta and phi bins accordingly to this values
min_bin = 0 # Start taking grid points in this bin
max_bin = 4 # Stop taking grid points in this bin

# getting NSM simulation grid in cartesian coordinates system
grid_cartesian = grid_file['Xcart'] * np.array(dump_file['L_unit']) # cm
NSM_grid_cartesian = grid_cartesian[ min_bin : max_bin , min_bin : max_bin , min_bin : max_bin , 1 : 4 ]

start = time.time()
# Perform interpolation
indices, T_rho_Ye = nsm_rho_Ye_T_linear_interpolator.interpolate_Ye_rho_T(NSM_grid_cartesian)
end = time.time()
print(f'Interpolation time = {end - start} s')

# Create arrays to store the interpolated values of T, rho, and Ye.
T   = np.full( ( max_bin - min_bin , max_bin - min_bin , max_bin - min_bin ) , 18081999.0 ) # array of size (ncellsx, ncellsy, ncellsz)
rho = np.full( ( max_bin - min_bin , max_bin - min_bin , max_bin - min_bin ) , 18081999.0 ) # array of size (ncellsx, ncellsy, ncellsz)
Ye  = np.full( ( max_bin - min_bin , max_bin - min_bin , max_bin - min_bin ) , 18081999.0 ) # array of size (ncellsx, ncellsy, ncellsz)

# Store the interpolated values of T, rho, and Ye.
T[indices]   = T_rho_Ye[:,0]
rho[indices] = T_rho_Ye[:,1]
Ye[indices]  = T_rho_Ye[:,2]

# Find if some points were not interpolated.
indices_no_interpolation_T = np.where( T == 18081999.0 )
indices_no_interpolation_rho = np.where( rho == 18081999.0 ) 
indices_no_interpolation_Ye = np.where( Ye == 18081999.0 )

# Print warning if some points were not interpolated.
if ( (len(indices_no_interpolation_T[0]) != 0 ) or (len(indices_no_interpolation_rho[0]) != 0 ) or (len(indices_no_interpolation_Ye[0]) != 0) ):
    print(f'Warnings: {len(indices_no_interpolation_T[0])} T points were not interpolated')
    print(f'Warnings: {len(indices_no_interpolation_rho[0])} rho points were not interpolated')
    print(f'Warnings: {len(indices_no_interpolation_Ye[0])} Ye points were not interpolated')

# Getting index of Ye and rho in NSM data
index_Ye = np.where(dump_file['P'].attrs['vnams'] == 'Ye')[0][0]
index_rho = np.where(dump_file['P'].attrs['vnams'] == 'RHO')[0][0]

# Getting Ye, rho, T in NSM data
Ye_NSM = dump_file['P'][:,:,:,index_Ye] # no units
Ye_NSM = Ye_NSM[ min_bin : max_bin , min_bin : max_bin , min_bin : max_bin ]
rho_NSM = dump_file['P'][:,:,:,index_rho] * dump_file['RHO_unit'] # g / ccm
rho_NSM = rho_NSM[ min_bin : max_bin , min_bin : max_bin , min_bin : max_bin ]
T_NSM = dump_file['TEMP'] # MeV
T_NSM = T_NSM[ min_bin : max_bin , min_bin : max_bin , min_bin : max_bin ]

# Filling non-interpolated data with 18081999.0 to only compute the error of interpolated points.
T_NSM  [indices_no_interpolation_T] = 18081999.0
rho_NSM[indices_no_interpolation_rho] = 18081999.0
Ye_NSM [indices_no_interpolation_Ye] = 18081999.0

# Print test results
print(f'rho test ---> Average relative error = { np.average( np.abs( rho - rho_NSM ) / rho_NSM ) }')
print(f'T test   ---> Average relative error = { np.average( np.abs( T   - T_NSM   ) / T_NSM   )}')
print(f'Ye test  ---> Average relative error = { np.average( np.abs( Ye  - Ye_NSM  ) / Ye_NSM  )}')
print(f'rho test ---> Maximum relative error = { np.max( np.abs( rho - rho_NSM ) / rho_NSM ) }')
print(f'T test   ---> Maximum relative error = { np.max( np.abs( T   - T_NSM   ) / T_NSM   )}')
print(f'Ye test  ---> Maximum relative error = { np.max( np.abs( Ye  - Ye_NSM  ) / Ye_NSM  )}')
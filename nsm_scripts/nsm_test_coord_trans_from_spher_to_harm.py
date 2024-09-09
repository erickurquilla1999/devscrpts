'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to test the scripts that transform a point in Spherical coordinates to the Harm cordinates used in Jonah's NSM simulation.
The data files dump_00001000.h5 and grid.h5 should be in the same directory as this script.
'''

import numpy as np
import h5py    
import nsm_coord_trans_from_spher_to_harm
import time

# importing Jonah NSM data
dump_file = h5py.File('dump_00001000.h5','r')
grid_file = h5py.File('grid.h5','r')

# Getting index of Ye and rho in NSM data
index_Ye = np.where(dump_file['P'].attrs['vnams'] == 'Ye')[0][0]
index_rho = np.where(dump_file['P'].attrs['vnams'] == 'RHO')[0][0]

# Getting Ye, rho, T in NSM data
Ye = dump_file['P'][:,:,:,index_Ye] # no units
rho = dump_file['P'][:,:,:,index_rho] * dump_file['RHO_unit'] # g / ccm
T = dump_file['TEMP'] # MeV

# getting NSM simulation grid in harm coordinates system
grid_harm  = grid_file['Xharm']
harm_points = grid_harm[:,:,:,1:4]
X1_harm_coordinates = grid_harm[...,1]
X2_harm_coordinates = grid_harm[...,2]
X3_harm_coordinates = grid_harm[...,3]

# getting NSM simulation grid in cartesian coordinates system
grid_cartesian = grid_file['Xcart'] * np.array(dump_file['L_unit']) # cm
x_cartesian_coordinates = grid_cartesian[...,1] # cm
y_cartesian_coordinates = grid_cartesian[...,2] # cm
z_cartesian_coordinates = grid_cartesian[...,3] # cm

# Transform NSM simulation grid to spherical coordinates system
r_spherical_coordinates     = np.sqrt( x_cartesian_coordinates**2 + y_cartesian_coordinates**2 + z_cartesian_coordinates**2 ) # cm
theta_spherical_coordinates = np.arccos( z_cartesian_coordinates / r_spherical_coordinates ) # radians from 0 to pi
phi_spherical_coordinates   = np.arctan2( y_cartesian_coordinates, x_cartesian_coordinates ) # radians from -pi to pi
phi_spherical_coordinates = np.where(phi_spherical_coordinates < 0, phi_spherical_coordinates + 2 * np.pi, phi_spherical_coordinates) # Adjusting the angle to be between 0 and 2pi.

# This list saves the difference between the interpolated point and the real value.
interpolation_difference = []

# Data cut control
a = 10

# Cutting data to loop
r_1 = r_spherical_coordinates[0:a,0:a,0:a]

# Looping over data
for i in range(r_1.shape[0]):
    for j in range(r_1.shape[1]):
        for k in range(r_1.shape[2]):
            #Perform interpolation
            interpolation_result = nsm_coord_trans_from_spher_to_harm.from_cart_to_harm_tranformation( ( r_spherical_coordinates[i,j,k] , theta_spherical_coordinates[i,j,k] , phi_spherical_coordinates[i,j,k] ) )
            interpolation_difference.append(np.abs(interpolation_result - harm_points[i,j,k]))

# Cutting data to loop
r_1 = r_spherical_coordinates[-a:,-a:,-a:]

# Looping over data
for i in range(r_1.shape[0]):
    for j in range(r_1.shape[1]):
        for k in range(r_1.shape[2]):
            #Perform interpolation
            interpolation_result = nsm_coord_trans_from_spher_to_harm.from_cart_to_harm_tranformation( ( r_spherical_coordinates[i,j,k] , theta_spherical_coordinates[i,j,k] , phi_spherical_coordinates[i,j,k] ) )
            interpolation_difference.append(np.abs(interpolation_result - harm_points[i,j,k]))

# Convert to np array
interpolation_difference = np.array(interpolation_difference)

#Print test result
print(f'\nMaximum difference in X1 = {np.max(interpolation_difference[:,0])}')
print(f'Minimum difference in X1 = {np.min(interpolation_difference[:,0])}')
print(f'\nMaximum difference in X2 = {np.max(interpolation_difference[:,1])}')
print(f'Minimum difference in X2 = {np.min(interpolation_difference[:,1])}')
print(f'\nMaximum difference in X3 = {np.max(interpolation_difference[:,2])}')
print(f'Minimum difference in X3 = {np.min(interpolation_difference[:,2])}')
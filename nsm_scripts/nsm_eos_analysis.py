'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to generate data (rho, Ye, T) to do an EOS analysis of the neutrino collisional flavor instability with Zidu.
The data files dump_00001000.h5 and grid.h5 should be in the same directory as this script.
'''

import numpy as np
import h5py

# importing Jonah NSM data
dump_file = h5py.File('dump_00001000.h5','r')
grid_file = h5py.File('grid.h5','r')

# getting grid in cartesian coordinates system
grid_cartesian = grid_file['Xcart'] * np.array(dump_file['L_unit']) # cm
x_cartesian_coordinates = grid_cartesian[...,1] # cm
y_cartesian_coordinates = grid_cartesian[...,2] # cm
z_cartesian_coordinates = grid_cartesian[...,3] # cm

# Getting index of Ye and rho
index_Ye = np.where(dump_file['P'].attrs['vnams'] == 'Ye')[0][0]
index_rho = np.where(dump_file['P'].attrs['vnams'] == 'RHO')[0][0]

# Getting Ye, rho, T
Ye = dump_file['P'][:,:,:,index_Ye] # no units
rho = dump_file['P'][:,:,:,index_rho] * dump_file['RHO_unit'] # g / ccm
T = dump_file['TEMP'] # MeV

#
r_idx = 8 # radial index
phi_idx = 0 # azimutal index

# Maximum electron fraction
Max_Ye_index = Ye[r_idx,:,phi_idx].argmax()
Max_Ye_point = [Ye[r_idx,Max_Ye_index,phi_idx], T[r_idx,Max_Ye_index,phi_idx] , rho[r_idx,Max_Ye_index,phi_idx]]
Max_Ye_coord = grid_cartesian[r_idx,Max_Ye_index,phi_idx,1:4]

# Maximum density
Max_rho_index = rho[r_idx,:,phi_idx].argmax()
Max_rho_point = [Ye[r_idx,Max_rho_index,phi_idx], T[r_idx,Max_rho_index,phi_idx] , rho[r_idx,Max_rho_index,phi_idx]]
Max_rho_coord = grid_cartesian[r_idx,Max_rho_index,phi_idx,1:4]

# Maximum temperature
Max_T_index = T[r_idx,:,phi_idx].argmax()
Max_T_point = [Ye[r_idx,Max_T_index,phi_idx], T[r_idx,Max_T_index,phi_idx] , rho[r_idx,Max_T_index,phi_idx]]
Max_T_coord = grid_cartesian[r_idx,Max_T_index,phi_idx,1:4]

print(f'Points        = [ Ye , T_Mev , rho_g/ccm ]')
print(f'Max_Ye_point  = {Max_Ye_point}')
print(f'Max_T_point   = {Max_T_point}')
print(f'Max_rho_point = {Max_rho_point}')
print(f'Max_Ye_coord = {Max_Ye_coord}')
print(f'Max_rho_coord = {Max_rho_coord}')
print(f'Max_T_coord = {Max_T_coord}')
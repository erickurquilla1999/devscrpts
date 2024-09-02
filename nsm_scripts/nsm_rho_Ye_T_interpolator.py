'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to interpolate fluid data (rho, Ye, T) from Jonah's NSM simulation to the EMU Cartesian grid.
The data files dump_00001000.h5 and grid.h5 should be in the same directory as this script.
'''

import numpy as np
import h5py
from scipy.interpolate import griddata

def interpolate(grid, T, rho, Ye, interpolation_point):
    """
    Interpolates values of temperature (T), density (rho), and electron fraction (Ye)
    at a given interpolation point within a non-structured grid.
    
    Parameters:
    - grid: A (N,3) array-like of points where the N data points are known. 
            This is a non-structured grid with known coordinates.
    - T: A 1D array-like of temperature values corresponding to each point in the grid.
    - rho: A 1D array-like of density values corresponding to each point in the grid.
    - Ye: A 1D array-like of electron fraction values corresponding to each point in the grid.
    - interpolation_point: A tuple or list representing the coordinates where interpolation is desired.

    Returns:
    - A list containing the interpolated values of T, rho, and Ye at the interpolation point.
      [T_interpolated, rho_interpolated, Ye_interpolated]
    """
    
    # Perform linear interpolation for the temperature (T) at the given interpolation point
    T_interpolated   = griddata( grid, T,   interpolation_point, method='linear', rescale=False)[0]

    # Perform linear interpolation for the density (rho) at the given interpolation point
    rho_interpolated = griddata( grid, rho, interpolation_point, method='linear', rescale=False)[0]
    
    # Perform linear interpolation for the electron fraction (Ye) at the given interpolation point
    Ye_interpolated  = griddata( grid, Ye,  interpolation_point, method='linear', rescale=False)[0]
    
    # Return the interpolated values as a list
    return [T_interpolated, rho_interpolated, Ye_interpolated]


def interpolate_Ye_rho_T(emu_mesh):
    """
    Interpolates the electron fraction (Ye), density (rho), and temperature (T)
    from a neutron star merger (NSM) simulation onto an EMU mesh grid.
    
    Parameters:
    - emu_mesh: A 3D array representing the EMU mesh grid points in Cartesian coordinates.
    
    Returns:
    - indices_r_emu_grid_in_nsm_simulation_domain: The indices of the EMU grid points that lie within the NSM simulation domain.
    - rho_Ye_T_interpolated: The interpolated values of rho, Ye, and T at the selected EMU grid points.
    """
    
    # importing Jonah NSM data
    dump_file = h5py.File('dump_00001000.h5','r')
    grid_file = h5py.File('grid.h5','r')

    # getting NSM simulation grid in cartesian coordinates system
    grid_cartesian = grid_file['Xcart'] * np.array(dump_file['L_unit']) # cm
    x_cartesian_coordinates = grid_cartesian[...,1] # cm
    y_cartesian_coordinates = grid_cartesian[...,2] # cm
    z_cartesian_coordinates = grid_cartesian[...,3] # cm

    # Getting index of Ye and rho in NSM data
    index_Ye = np.where(dump_file['P'].attrs['vnams'] == 'Ye')[0][0]
    index_rho = np.where(dump_file['P'].attrs['vnams'] == 'RHO')[0][0]

    # Getting Ye, rho, T in NSM data
    Ye = dump_file['P'][:,:,:,index_Ye] # no units
    rho = dump_file['P'][:,:,:,index_rho] * dump_file['RHO_unit'] # g / ccm
    T = dump_file['TEMP'] # MeV

    # Computing delta_ln_r in NSM grid points. This quantity remains constant in the data.
    r_points   = np.sqrt( x_cartesian_coordinates[:,0,0]**2 + y_cartesian_coordinates[:,0,0]**2 + z_cartesian_coordinates[:,0,0]**2 )
    r_start    = r_points[0]
    r_end      = r_points[-1]
    delta_ln_r = np.log(r_points[1]) - np.log(r_points[0]) 
    lnr_start  = np.log(r_points[0])

    print(f'NSM r_start = {r_start:.3e} cm')
    print(f'NSM r_end = {r_end:.3e} cm')
    print(f'NSM min_delta_r = {( r_points[1] - r_points[0] ):.3e} cm')

    # Computing delta_phi in NSM grid points. This quantity remains constant in the data.
    phi_points = np.arctan2( y_cartesian_coordinates[0,0,:] , x_cartesian_coordinates[0,0,:] ) 
    phi_start  = phi_points[0]
    phi_end    = phi_points[-1]
    delta_phi  = phi_points[1] - phi_points[0]
    
    # Compute spherical coordinates \( r \) and \( \phi \) of EMU mesh points.
    r_emu_grid = np.sqrt(emu_mesh[:,:,:,0]**2 + emu_mesh[:,:,:,1]**2 + emu_mesh[:,:,:,2]**2) # Computing r in spherical coordinates for the EMU grid points.
    phi_emu_grid = np.arctan2( emu_mesh[:,:,:,1] , emu_mesh[:,:,:,0] )  # Computing the phi in spherical coordinates for the EMU grid points.

    # Extracting indices of the EMU grid points that are within the NSM simulation domain.
    indices_r_emu_grid_in_nsm_simulation_domain = np.where( (r_emu_grid > r_start) & (r_emu_grid < r_end) ) # Do not allow cell centers on the NSM domain boundaries.
    
    # Extracting the EMU grid points that are within the NSM simulation domain.
    r_emu_grid_for_interpolation   = r_emu_grid       [indices_r_emu_grid_in_nsm_simulation_domain]
    phi_emu_grid_for_interpolation = phi_emu_grid     [indices_r_emu_grid_in_nsm_simulation_domain]
    phi_emu_grid_for_interpolation = np.where(phi_emu_grid_for_interpolation < 0, phi_emu_grid_for_interpolation + 2 * np.pi, phi_emu_grid_for_interpolation) # Adjusting the angle to be between 0 and 2pi.
    x_emu_grid_for_interpolation   = emu_mesh[:,:,:,0][indices_r_emu_grid_in_nsm_simulation_domain]
    y_emu_grid_for_interpolation   = emu_mesh[:,:,:,1][indices_r_emu_grid_in_nsm_simulation_domain]
    z_emu_grid_for_interpolation   = emu_mesh[:,:,:,2][indices_r_emu_grid_in_nsm_simulation_domain]

    # EMU grid points in Cartesian coordinates where interpolation will be performed.
    emu_grid_for_interpolation = np.stack( ( x_emu_grid_for_interpolation , y_emu_grid_for_interpolation , z_emu_grid_for_interpolation ), axis=-1)

    # Compute the radial bin in NSM data of EMU grid points.
    ln_r_emu_grid_for_interpolation  = np.log( r_emu_grid_for_interpolation )
    ln_r_emu_grid_float_index = ( ln_r_emu_grid_for_interpolation - lnr_start ) / delta_ln_r
    ln_r_emu_grid_ceil_index = np.ceil(ln_r_emu_grid_float_index)   # The ceil of the scalar x is the smallest integer i, such that i >= x. 
    ln_r_emu_grid_floor_index = np.floor(ln_r_emu_grid_float_index) # The floor of the scalar x is the largest integer i, such that i <= x.
    ln_r_emu_grid_indices = np.stack( ( ln_r_emu_grid_floor_index , ln_r_emu_grid_ceil_index ), axis=-1)
    ln_r_emu_grid_indices = ln_r_emu_grid_indices.astype(int)
    
    # Compute the azimultal bin in NSM data of EMU grid points.
    phi_emu_grid_float_index = ( phi_emu_grid_for_interpolation - phi_start ) / delta_phi
    phi_emu_grid_ceil_index = np.ceil(phi_emu_grid_float_index)   # The ceil of the scalar x is the smallest integer i, such that i >= x. 
    phi_emu_grid_floor_index = np.floor(phi_emu_grid_float_index) # The floor of the scalar x is the largest integer i, such that i <= x.
    phi_emu_grid_indices = np.stack( ( phi_emu_grid_floor_index , phi_emu_grid_ceil_index ), axis=-1)
    phi_emu_grid_indices = phi_emu_grid_indices.astype(int)

    # This array will save the interpolated values of rho, Ye and T
    rho_Ye_T_interpolated = np.zeros( ( len(emu_grid_for_interpolation) , 3 ) )
    
    # Looping over all EMU grid point and perform interpolation
    for i in range(len(emu_grid_for_interpolation)):
        
        minus_r_idx = ln_r_emu_grid_indices[i][0]
        plus_r_idx  = ln_r_emu_grid_indices[i][1]
        minus_phi_idx = phi_emu_grid_indices[i][0]
        plus_phi_idx  = phi_emu_grid_indices[i][1]

        if ( minus_r_idx == plus_r_idx ):
            # If an EMU grid point is exactly between two radial bins, consider both bins for interpolation.
            minus_r_idx -= 1
            plus_r_idx  += 1 

        if ( minus_phi_idx == plus_phi_idx ):
            # If an EMU grid point is exactly between two azimultal bins, consider both bins for interpolation.
            minus_phi_idx -= 1
            plus_phi_idx
        
        # To perform faster interpolation, reduce the NSM data set to the radial and azimuthal bin where the EMU grid point is located.
        x_for_interpolation   = x_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , : , minus_phi_idx : plus_phi_idx + 1 ].flatten()
        y_for_interpolation   = y_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , : , minus_phi_idx : plus_phi_idx + 1 ].flatten()
        z_for_interpolation   = z_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , : , minus_phi_idx : plus_phi_idx + 1 ].flatten()
        Ye_for_interpolation  = Ye                     [ minus_r_idx : plus_r_idx + 1 , : , minus_phi_idx : plus_phi_idx + 1 ].flatten()
        T_for_interpolation   = T                      [ minus_r_idx : plus_r_idx + 1 , : , minus_phi_idx : plus_phi_idx + 1 ].flatten()
        rho_for_interpolation = rho                    [ minus_r_idx : plus_r_idx + 1 , : , minus_phi_idx : plus_phi_idx + 1 ].flatten()

        # Unstructured grid points extracted from the NSM simulation data set closest to where the EMU grid point is located.
        grid_for_interpolation = np.stack( ( x_for_interpolation , y_for_interpolation , z_for_interpolation ) , axis=-1 )
        
        # Perform interpolation
        rho_Ye_T_interpolated[i] = interpolate(grid_for_interpolation, T_for_interpolation, rho_for_interpolation, Ye_for_interpolation, emu_grid_for_interpolation[i])

    return indices_r_emu_grid_in_nsm_simulation_domain , rho_Ye_T_interpolated
'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to interpolate fluid data (rho, Ye, T) from Jonah's NSM simulation to the EMU Cartesian grid.
This script use a linear interpolation in a regular grid
The data files dump_00001000.h5 and grid.h5 should be in the same directory as this script.
'''

import numpy as np
import h5py
from scipy.interpolate import RegularGridInterpolator
import multiprocessing
import nsm_coord_trans_from_spher_to_harm
import time

def interpolate(data):
    """
    Performs linear interpolation for temperature (T), density (rho), and electron fraction (Ye)
    at a given interpolation point using RegularGridInterpolator.

    Parameters:
    data (list): A list containing the following elements:
        - data[0]: The coordinates of the interpolation point (list or array-like).
        - data[1]: The x-coordinates for the regular grid (1D array).
        - data[2]: The y-coordinates for the regular grid (1D array).
        - data[3]: The z-coordinates for the regular grid (1D array).
        - data[4]: The temperature values defined on the (x, y, z) grid (3D array).
        - data[5]: The density values defined on the (x, y, z) grid (3D array).
        - data[6]: The electron fraction (Ye) values defined on the (x, y, z) grid (3D array).

    Returns:
    list: A list containing the interpolated values at the given interpolation point:
        - T_interpolated: Interpolated temperature value at the interpolation point.
        - rho_interpolated: Interpolated density value at the interpolation point.
        - Ye_interpolated: Interpolated electron fraction (Ye) value at the interpolation point.

    Example:
    Given data for the coordinates and grid values, this function will interpolate 
    the temperature, density, and electron fraction at the specified interpolation point.
    """
  
    interpolation_point = data[0]
    x = data[1]
    y = data[2]
    z = data[3]
    T = data[4]
    rho = data[5]
    Ye = data[6]

    try:
      
      # Perform linear interpolation for the temperature (T) at the given interpolation point
      T_interpolated = RegularGridInterpolator((x, y, z), T)(interpolation_point)[0]
      
      # Perform linear interpolation for the density (rho) at the given interpolation point
      rho_interpolated = RegularGridInterpolator((x, y, z), rho)(interpolation_point)[0]
      
      # Perform linear interpolation for the electron fraction (Ye) at the given interpolation point
      Ye_interpolated = RegularGridInterpolator((x, y, z), Ye)(interpolation_point)[0]
      
      # Return the interpolated values as a list
      return [T_interpolated, rho_interpolated, Ye_interpolated]

    except Exception as e:

      # Print error if interpolation fails
      print(f"Interpolation failed: {e}. \n{interpolation_point} = point \n{x} = x \n{y} = y \n{z} = z")
      
      # Return the interpolated values as a list
      return [ 18081999.0 , 18081999.0 , 18081999.0 ]

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

    # Getting index of Ye and rho in NSM data
    index_Ye = np.where(dump_file['P'].attrs['vnams'] == 'Ye')[0][0]
    index_rho = np.where(dump_file['P'].attrs['vnams'] == 'RHO')[0][0]

    # Getting Ye, rho, T in NSM data
    Ye = dump_file['P'][:,:,:,index_Ye] # no units
    rho = dump_file['P'][:,:,:,index_rho] * dump_file['RHO_unit'] # g / ccm
    T = dump_file['TEMP'] # MeV

    # getting NSM simulation grid in harm coordinates system
    grid_harm  = grid_file['Xharm']
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

    # Getting maximum r, theta and phi of NSM simulation grid
    r_start     = np.min(r_spherical_coordinates)
    r_end       = np.max(r_spherical_coordinates)
    theta_start = np.min(theta_spherical_coordinates)
    theta_end   = np.max(theta_spherical_coordinates)
    phi_start   = np.min(phi_spherical_coordinates)
    phi_end     = np.max(phi_spherical_coordinates)

    print(f'NSM r_start = {r_start:.3e} cm')
    print(f'NSM r_end = {r_end:.3e} cm')
    print(f'NSM min_delta_r = { np.min( r_spherical_coordinates[1:,:,:] - r_spherical_coordinates[0:-1,:,:] ) :.3e} cm')
    print(f'NSM theta_start = {theta_start:.3e} rad')
    print(f'NSM theta_end = {theta_end:.3e} rad')
    print(f'NSM phi_start = {phi_start:.3e} rad')
    print(f'NSM phi_end = {phi_end:.3e} rad')

    # Getting maximum X1, X2 and X3 of NSM simulation grid
    X1_start = np.min(X1_harm_coordinates)
    X1_end   = np.max(X1_harm_coordinates)
    X2_start = np.min(X2_harm_coordinates)
    X2_end   = np.max(X2_harm_coordinates)
    X3_start = np.min(X3_harm_coordinates)
    X3_end   = np.max(X3_harm_coordinates)

    print(f'NSM X1_start = {X1_start}')
    print(f'NSM X1_end = {X1_end}')
    print(f'NSM X2_start = {X2_start}')
    print(f'NSM X2_end = {X2_end}')
    print(f'NSM X3_start = {X3_start}')
    print(f'NSM X3_end = {X3_end}')

    print(f'NSM min_delta_X1 = { np.min( X1_harm_coordinates[1:,:,:] - X1_harm_coordinates[0:-1,:,:] ) }')
    print(f'NSM max_delta_X1 = { np.max( X1_harm_coordinates[1:,:,:] - X1_harm_coordinates[0:-1,:,:] ) }')
    print(f'NSM min_delta_X2 = { np.min( X2_harm_coordinates[:,1:,:] - X2_harm_coordinates[:,0:-1,:] ) }')
    print(f'NSM max_delta_X2 = { np.max( X2_harm_coordinates[:,1:,:] - X2_harm_coordinates[:,0:-1,:] ) }')
    print(f'NSM min_delta_X3 = { np.min( X3_harm_coordinates[:,:,1:] - X3_harm_coordinates[:,:,0:-1] ) }')
    print(f'NSM max_delta_X3 = { np.max( X3_harm_coordinates[:,:,1:] - X3_harm_coordinates[:,:,0:-1] ) }')

    # Compute spherical coordinates \( r \) and \( \phi \) of EMU mesh points.
    r_emu_grid     = np.sqrt(emu_mesh[:,:,:,0]**2 + emu_mesh[:,:,:,1]**2 + emu_mesh[:,:,:,2]**2) # Computing r in spherical coordinates for the EMU grid points.
    theta_emu_grid = np.arccos( emu_mesh[:,:,:,2] / r_emu_grid ) # Computing theta in spherical coordinates for the EMU grid points in radians from 0 to pi
    phi_emu_grid   = np.arctan2( emu_mesh[:,:,:,1] , emu_mesh[:,:,:,0] )  # Computing the phi in spherical coordinates for the EMU grid points in radians from -pi to pi
    phi_emu_grid   = np.where(phi_emu_grid < 0, phi_emu_grid + 2 * np.pi, phi_emu_grid) # Adjusting the angle to be between 0 and 2pi.

    # Extracting indices of the EMU grid points that are within the NSM simulation domain.
    indices_r_emu_grid_in_nsm_simulation_domain = np.where( (r_emu_grid >= r_start) & (r_emu_grid <= r_end) ) # Allow cell centers on the NSM domain boundaries.
    
    # Extracting the EMU grid points that are within the NSM simulation domain.
    r_emu_grid_for_interpolation     = r_emu_grid         [indices_r_emu_grid_in_nsm_simulation_domain]
    theta_emu_grid_for_interpolation = theta_emu_grid     [indices_r_emu_grid_in_nsm_simulation_domain]
    phi_emu_grid_for_interpolation   = phi_emu_grid       [indices_r_emu_grid_in_nsm_simulation_domain]

    print(f' ---> {len(r_emu_grid_for_interpolation)} / {len(r_emu_grid.flatten())} points were taken for interpolation')

    # EMU grid points in Cartesian coordinates where interpolation will be performed.
    emu_grid_for_interpolation = np.stack( ( r_emu_grid_for_interpolation , theta_emu_grid_for_interpolation , phi_emu_grid_for_interpolation ), axis=-1)

    start = time.time()
    # Create a Pool with the number of processes equal to the number of CPU cores
    pool = multiprocessing.Pool()
    # EMU grid points in Harm coordinates where interpolation will be performed.
    emu_grid_for_interpolation_harm = np.array(pool.map(nsm_coord_trans_from_spher_to_harm.from_cart_to_harm_tranformation, emu_grid_for_interpolation))
    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()
    end = time.time()
    print(f'Coordinate transformation time = {end - start} s')

    # Coordinates of the interpolation points in the EMU mesh
    X1_emu = emu_grid_for_interpolation_harm[:,0]
    X2_emu = emu_grid_for_interpolation_harm[:,1]
    X3_emu = emu_grid_for_interpolation_harm[:,2]

    # Number of interpolation points
    number_interpolation_points = len(X1_emu)

    # NSM mesh coordinates in one dimensional single arrays
    X1_NSM = X1_harm_coordinates[:,0,0]
    X2_NSM = X2_harm_coordinates[0,:,0]
    X3_NSM = X3_harm_coordinates[0,0,:]

    # Compute the two polar indices of the NSM mesh where the EMU grid points are located.
    X1_closer_index = [ np.abs(  X1_emu[i] - X1_NSM ).argmin() for i in range(number_interpolation_points) ] 
    X2_closer_index = [ np.abs(  X2_emu[i] - X2_NSM ).argmin() for i in range(number_interpolation_points) ] 
    X3_closer_index = [ np.abs(  X3_emu[i] - X3_NSM ).argmin() for i in range(number_interpolation_points) ] 

    # List to store interpolation data for each point in the EMU mesh to be interpolated.
    inderpolation_data = [None] * number_interpolation_points

    def get_indices(closer_index, i):
        """
        Determines the neighboring indices for interpolation based on the value 
        of the current index `i` in the `closer_index` array.

        Parameters:
        closer_index (list or array): A list or array containing indices.
        i (int): The current index in the `closer_index` array for which neighboring
                indices need to be determined.

        Returns:
        list: A list of two values representing the neighboring indices:
              - If the current index is the first element (0), returns [0, 3].
              - If the current index is the last element, returns [-3, None].
              - Otherwise, returns the indices just before and after the current index.
        """
        
        # Check if the current index is the first element in the array
        if closer_index[i] == 0:
            # If the first element, return [0, 3] as the neighboring indices
            return [0, 3]
        
        # Check if the current index is the last element in the array
        elif closer_index[i] == len(closer_index) - 1:
            # If the last element, return [-3, None] as the neighboring indices
            return [-3, None]
        
        # For any other case, return the previous and next indices
        else:
            return [closer_index[i] - 1, closer_index[i] + 2]

    # Looping over X1_closer_index 
    for i in range( number_interpolation_points ):

      # Initialize neighboring indices as None
      X1_indices = [None,None]
      X2_indices = [None,None]
      X3_indices = [None,None]

      # Get neighboring indices
      X1_indices = get_indices(X1_closer_index, i)
      X2_indices = get_indices(X2_closer_index, i)
      X3_indices = get_indices(X3_closer_index, i)

      # Trim data arrays to get only points close to the interpolation point 
      X1_interpolation = X1_NSM[X1_indices[0]:X1_indices[1]]
      X2_interpolation = X2_NSM[X2_indices[0]:X2_indices[1]]
      X3_interpolation = X3_NSM[X3_indices[0]:X3_indices[1]]
      T_interpolation   = T  [ X1_indices[0]:X1_indices[1] , X2_indices[0]:X2_indices[1] , X3_indices[0]:X3_indices[1] ]
      rho_interpolation = rho[ X1_indices[0]:X1_indices[1] , X2_indices[0]:X2_indices[1] , X3_indices[0]:X3_indices[1] ]
      Ye_interpolation  = Ye [ X1_indices[0]:X1_indices[1] , X2_indices[0]:X2_indices[1] , X3_indices[0]:X3_indices[1] ]

      # Save interpolation data
      inderpolation_data[i] = [ [ X1_emu[i], X2_emu[i], X3_emu[i] ] , X1_interpolation , X2_interpolation , X3_interpolation , T_interpolation , rho_interpolation , Ye_interpolation ]

    pool = multiprocessing.Pool()
    # Perform interpolation
    T_rho_Ye_interpolated = np.array( pool.map( interpolate , inderpolation_data ) )
    pool.close()
    pool.join()

    return indices_r_emu_grid_in_nsm_simulation_domain , T_rho_Ye_interpolated

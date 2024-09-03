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

    # Getting index of Ye and rho in NSM data
    index_Ye = np.where(dump_file['P'].attrs['vnams'] == 'Ye')[0][0]
    index_rho = np.where(dump_file['P'].attrs['vnams'] == 'RHO')[0][0]

    # Getting Ye, rho, T in NSM data
    Ye = dump_file['P'][:,:,:,index_Ye] # no units
    rho = dump_file['P'][:,:,:,index_rho] * dump_file['RHO_unit'] # g / ccm
    T = dump_file['TEMP'] # MeV

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

    # Computing delta_ln_r in NSM grid points. This quantity remains constant in the data.
    r_points   = r_spherical_coordinates[:,0,0]
    delta_ln_r = np.log(r_points[1]) - np.log(r_points[0])
    lnr_start  = np.log(r_start)

    print(f'NSM r_start = {r_start:.3e} cm')
    print(f'NSM r_end = {r_end:.3e} cm')
    print(f'NSM min_delta_r = {( r_points[1] - r_points[0] ):.3e} cm')
    print(f'NSM theta_start = {theta_start:.3e} cm')
    print(f'NSM theta_end = {theta_end:.3e} cm')
    print(f'NSM phi_start = {phi_start:.3e} cm')
    print(f'NSM phi_end = {phi_end:.3e} cm')

    # Computing delta_phi in NSM grid points. This quantity remains constant in the data.
    phi_points = phi_spherical_coordinates[0,0,:]
    delta_phi  = phi_points[1] - phi_points[0]

    # Compute spherical coordinates \( r \) and \( \phi \) of EMU mesh points.
    r_emu_grid     = np.sqrt(emu_mesh[:,:,:,0]**2 + emu_mesh[:,:,:,1]**2 + emu_mesh[:,:,:,2]**2) # Computing r in spherical coordinates for the EMU grid points.
    theta_emu_grid = np.arccos( emu_mesh[:,:,:,2] / r_emu_grid ) # Computing theta in spherical coordinates for the EMU grid points in radians from 0 to pi
    phi_emu_grid   = np.arctan2( emu_mesh[:,:,:,1] , emu_mesh[:,:,:,0] )  # Computing the phi in spherical coordinates for the EMU grid points in radians from -pi to pi
    phi_emu_grid   = np.where(phi_emu_grid < 0, phi_emu_grid + 2 * np.pi, phi_emu_grid) # Adjusting the angle to be between 0 and 2pi.

    # Extracting indices of the EMU grid points that are within the NSM simulation domain.
    # indices_r_emu_grid_in_nsm_simulation_domain = np.where( (r_emu_grid > r_start) & (r_emu_grid < r_end) ) # Do not allow cell centers on the NSM domain boundaries.
    indices_r_emu_grid_in_nsm_simulation_domain = np.where( (r_emu_grid >= r_start) & (r_emu_grid <= r_end) ) # Allow cell centers on the NSM domain boundaries.
    
    # Extracting the EMU grid points that are within the NSM simulation domain.
    r_emu_grid_for_interpolation     = r_emu_grid         [indices_r_emu_grid_in_nsm_simulation_domain]
    theta_emu_grid_for_interpolation = theta_emu_grid     [indices_r_emu_grid_in_nsm_simulation_domain]
    phi_emu_grid_for_interpolation   = phi_emu_grid       [indices_r_emu_grid_in_nsm_simulation_domain]
    x_emu_grid_for_interpolation     = emu_mesh[:,:,:,0][indices_r_emu_grid_in_nsm_simulation_domain]
    y_emu_grid_for_interpolation     = emu_mesh[:,:,:,1][indices_r_emu_grid_in_nsm_simulation_domain]
    z_emu_grid_for_interpolation     = emu_mesh[:,:,:,2][indices_r_emu_grid_in_nsm_simulation_domain]

    print(f'There are {len(r_emu_grid.flatten())} points in EMU grid. {len(r_emu_grid_for_interpolation)} points where taken for interpolation')

    # EMU grid points in Cartesian coordinates where interpolation will be performed.
    emu_grid_for_interpolation = np.stack( ( x_emu_grid_for_interpolation , y_emu_grid_for_interpolation , z_emu_grid_for_interpolation ), axis=-1)

    # Compute the radial bin in NSM data of EMU grid points.
    ln_r_emu_grid_for_interpolation  = np.log( r_emu_grid_for_interpolation )
    ln_r_emu_grid_float_index = ( ln_r_emu_grid_for_interpolation - lnr_start ) / delta_ln_r
    ln_r_emu_grid_ceil_index = np.ceil(ln_r_emu_grid_float_index)   # The ceil of the scalar x is the smallest integer i, such that i >= x. 
    ln_r_emu_grid_floor_index = np.floor(ln_r_emu_grid_float_index) # The floor of the scalar x is the largest integer i, such that i <= x.
    ln_r_emu_grid_near_index = np.rint(ln_r_emu_grid_float_index) # Round each element to the nearest integer 
    ln_r_emu_grid_near_index = ln_r_emu_grid_near_index.astype(int)
    ln_r_emu_grid_indices = np.stack( ( ln_r_emu_grid_floor_index , ln_r_emu_grid_ceil_index ), axis=-1)
    ln_r_emu_grid_indices = ln_r_emu_grid_indices.astype(int)

    # Compute the azimultal bin in NSM data of EMU grid points.
    phi_emu_grid_float_index = ( phi_emu_grid_for_interpolation - phi_start ) / delta_phi
    phi_emu_grid_ceil_index = np.ceil(phi_emu_grid_float_index)   # The ceil of the scalar x is the smallest integer i, such that i >= x. 
    phi_emu_grid_floor_index = np.floor(phi_emu_grid_float_index) # The floor of the scalar x is the largest integer i, such that i <= x.
    phi_emu_grid_near_index = np.rint(phi_emu_grid_float_index) # Round each element to the nearest integer 
    phi_emu_grid_near_index = phi_emu_grid_near_index.astype(int)
    phi_emu_grid_indices = np.stack( ( phi_emu_grid_floor_index , phi_emu_grid_ceil_index ), axis=-1)
    phi_emu_grid_indices = phi_emu_grid_indices.astype(int)

    # Compute the polar bin in NSM data of EMU grid points.
    theta_emu_grid_closer_index      = [ np.abs(  theta_emu_grid_for_interpolation[i] - theta_spherical_coordinates[ln_r_emu_grid_near_index[i],:,phi_emu_grid_near_index[i]] ).argmin() for i in range(len(theta_emu_grid_for_interpolation)) ] 
    theta_emu_grid_closer_index_sing = [ np.sign( theta_emu_grid_for_interpolation[i] - theta_spherical_coordinates[ln_r_emu_grid_near_index[i],:,phi_emu_grid_near_index[i]] )[theta_emu_grid_closer_index[i]] for i in range(len(theta_emu_grid_for_interpolation)) ] 
    theta_emu_index_up = np.where(theta_emu_grid_closer_index_sing == 1)
    theta_emu_index_down = np.where(theta_emu_grid_closer_index_sing == -1)
    theta_emu_grid_up = np.array(theta_emu_grid_closer_index)
    theta_emu_grid_up[theta_emu_index_up] += 1
    theta_emu_grid_down = np.array(theta_emu_grid_closer_index)
    theta_emu_grid_down[theta_emu_index_down] -= 1
    theta_emu_grid_indices = np.stack( ( theta_emu_grid_down , theta_emu_grid_up ) , axis=-1 )

    # This array will save the interpolated values of rho, Ye and T
    T_rho_Ye_interpolated = np.zeros( ( len(emu_grid_for_interpolation) , 3 ) )
    
    # Looping over all EMU grid point and perform interpolation
    for i in range(len(emu_grid_for_interpolation)):
        
        minus_r_idx = ln_r_emu_grid_indices[i][0]
        plus_r_idx  = ln_r_emu_grid_indices[i][1]
        minus_theta_idx = theta_emu_grid_indices[i][0]
        plus_theta_idx  = theta_emu_grid_indices[i][1]
        minus_phi_idx = phi_emu_grid_indices[i][0]
        plus_phi_idx  = phi_emu_grid_indices[i][1]

        if ( minus_r_idx == plus_r_idx ):
            # If an EMU grid point is exactly between two radial bins, consider the following two option:
            if ( minus_r_idx == len( grid_cartesian[:,0,0,0] ) ):
                # If the grid point is at the outer boundary, consider the last bin.
                minus_r_idx += -1
                plus_r_idx  +=  0
            else :
                # Otherwise consider the bin in increasing radial direction.
                minus_r_idx += 0
                plus_r_idx  += 1

        # If theta is smaller than theta_start and bigger than theta_end
        theta_last_beam = False
        if ( ( minus_theta_idx == -1 ) or plus_theta_idx == len( grid_cartesian[0,:,0,0] ) ) :
            theta_last_beam = True            
        elif ( minus_theta_idx == plus_theta_idx ):
            # If an EMU grid point is exactly between two polar bins, consider the bin in increasing polar direction.
            minus_theta_idx -= 0
            plus_theta_idx  += 1 

        # If phi is smaller than phi_start and bigger than phi_end
        phi_last_beam = False
        if ( ( minus_phi_idx == -1 ) or plus_phi_idx == len( grid_cartesian[0,0,:,0] ) ) :
            phi_last_beam = True            
        elif ( minus_phi_idx == plus_phi_idx ):
            # If an EMU grid point is exactly between two azimultal bins, consider the bin in increasing azimultal direction.
            minus_phi_idx -= 0
            plus_phi_idx  += 1 

        # To perform faster interpolation, reduce the NSM data set to the radial, polar and azimuthal bin where the EMU grid point is located.
        if ( phi_last_beam and theta_last_beam ) : 

            # If phi and theta are in the last bin, concatenate the last phi and theta coordinates with the initial ones.
            x_for_interpolation   = np.concatenate( ( x_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , -1 : , -1 : ].flatten() , x_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , 0 : 1 , 0 : 1 ].flatten() ) )
            y_for_interpolation   = np.concatenate( ( y_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , -1 : , -1 : ].flatten() , y_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , 0 : 1 , 0 : 1 ].flatten() ) )
            z_for_interpolation   = np.concatenate( ( z_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , -1 : , -1 : ].flatten() , z_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , 0 : 1 , 0 : 1 ].flatten() ) )
            Ye_for_interpolation  = np.concatenate( ( Ye                     [ minus_r_idx : plus_r_idx + 1 , -1 : , -1 : ].flatten() , Ye                     [ minus_r_idx : plus_r_idx + 1 , 0 : 1 , 0 : 1 ].flatten() ) )
            T_for_interpolation   = np.concatenate( ( T                      [ minus_r_idx : plus_r_idx + 1 , -1 : , -1 : ].flatten() , T                      [ minus_r_idx : plus_r_idx + 1 , 0 : 1 , 0 : 1 ].flatten() ) )
            rho_for_interpolation = np.concatenate( ( rho                    [ minus_r_idx : plus_r_idx + 1 , -1 : , -1 : ].flatten() , rho                    [ minus_r_idx : plus_r_idx + 1 , 0 : 1 , 0 : 1 ].flatten() ) )

        elif theta_last_beam : 

            # If theta is in the last bin, concatenate the last theta coordinates with the initial ones.
            x_for_interpolation   = np.concatenate( ( x_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , -1 : , minus_phi_idx : plus_phi_idx + 1 ].flatten() , x_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , 0 : 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten() ) )
            y_for_interpolation   = np.concatenate( ( y_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , -1 : , minus_phi_idx : plus_phi_idx + 1 ].flatten() , y_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , 0 : 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten() ) )
            z_for_interpolation   = np.concatenate( ( z_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , -1 : , minus_phi_idx : plus_phi_idx + 1 ].flatten() , z_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , 0 : 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten() ) )
            Ye_for_interpolation  = np.concatenate( ( Ye                     [ minus_r_idx : plus_r_idx + 1 , -1 : , minus_phi_idx : plus_phi_idx + 1 ].flatten() , Ye                     [ minus_r_idx : plus_r_idx + 1 , 0 : 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten() ) )
            T_for_interpolation   = np.concatenate( ( T                      [ minus_r_idx : plus_r_idx + 1 , -1 : , minus_phi_idx : plus_phi_idx + 1 ].flatten() , T                      [ minus_r_idx : plus_r_idx + 1 , 0 : 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten() ) )
            rho_for_interpolation = np.concatenate( ( rho                    [ minus_r_idx : plus_r_idx + 1 , -1 : , minus_phi_idx : plus_phi_idx + 1 ].flatten() , rho                    [ minus_r_idx : plus_r_idx + 1 , 0 : 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten() ) )

        elif phi_last_beam :

            # If phi is in the last bin, concatenate the last phi coordinates with the initial ones.
            x_for_interpolation   = np.concatenate( ( x_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , -1 :   ].flatten() , x_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 ,  0 : 1 ].flatten() ) )
            y_for_interpolation   = np.concatenate( ( y_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , -1 :   ].flatten() , y_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 ,  0 : 1 ].flatten() ) )
            z_for_interpolation   = np.concatenate( ( z_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , -1 :   ].flatten() , z_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 ,  0 : 1 ].flatten() ) )
            Ye_for_interpolation  = np.concatenate( ( Ye                     [ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , -1 :   ].flatten() , Ye                     [ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 ,  0 : 1 ].flatten() ) )
            T_for_interpolation   = np.concatenate( ( T                      [ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , -1 :   ].flatten() , T                      [ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 ,  0 : 1 ].flatten() ) )
            rho_for_interpolation = np.concatenate( ( rho                    [ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , -1 :   ].flatten() , rho                    [ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 ,  0 : 1 ].flatten() ) )

        else :

            # If phi is not in the last bin, concatenate contiguous phi coordinates using a slicing operation.
            x_for_interpolation   = x_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten()
            y_for_interpolation   = y_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten()
            z_for_interpolation   = z_cartesian_coordinates[ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten()
            Ye_for_interpolation  = Ye                     [ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten()
            T_for_interpolation   = T                      [ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten()
            rho_for_interpolation = rho                    [ minus_r_idx : plus_r_idx + 1 , minus_theta_idx : plus_theta_idx + 1 , minus_phi_idx : plus_phi_idx + 1 ].flatten()

        # Unstructured grid points extracted from the NSM simulation data set closest to where the EMU grid point is located.
        grid_for_interpolation = np.stack( ( x_for_interpolation , y_for_interpolation , z_for_interpolation ) , axis=-1 )
        
        # Perform interpolation
        T_rho_Ye_interpolated[i] = interpolate(grid_for_interpolation, T_for_interpolation, rho_for_interpolation, Ye_for_interpolation, emu_grid_for_interpolation[i])

        # Print status
        if i % ( len(emu_grid_for_interpolation) // 100 ) == 0:
            percentage = (i * 100) // len(emu_grid_for_interpolation)
            print(f"{percentage}% of the mesh points interpolated")

    return indices_r_emu_grid_in_nsm_simulation_domain , T_rho_Ye_interpolated
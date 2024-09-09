'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to test the coordinates transformation from harm to cartesian coordinates in Jonah's NSM simulation data.
The data files dump_00001000.h5 and grid.h5 should be in the same directory as this script.
'''

import numpy as np
import h5py

# importing Jonah NSM data
dump_file = h5py.File('dump_00001000.h5','r')
grid_file = h5py.File('grid.h5','r')

# getting grid in harm coordinates system
grid_harm  = grid_file['Xharm']
X1_harm_coordinates = grid_harm[...,1]
X2_harm_coordinates = grid_harm[...,2]
X3_harm_coordinates = grid_harm[...,3]

# getting grid in cartesian coordinates system
grid_cartesian = grid_file['Xcart'] * np.array(dump_file['L_unit']) # cm
x_cartesian_coordinates = grid_cartesian[...,1] # cm
y_cartesian_coordinates = grid_cartesian[...,2] # cm
z_cartesian_coordinates = grid_cartesian[...,3] # cm

# pull out variables required for transformation from harm to spherical coordinates 
# harm coordinate system information can be found in https://github.com/lanl/nubhlight/wiki

derefine_poles = dump_file['derefine_poles'][0]
mks_smooth = dump_file['mks_smooth'][0]
hslope = dump_file['hslope'][0]
poly_xt = dump_file['poly_xt'][0]
poly_alpha = dump_file['poly_alpha'][0]
min_X1_harm_coordinates = dump_file['startx[1]'][0]

# poly_norm     = 0.5 *  M_PI * 1. / ( 1. + 1. / ( poly_alpha + 1. ) * 1. / pow(poly_xt, poly_alpha)); https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L439
poly_norm       = 0.5 * np.pi * 1. / ( 1. + 1. / ( poly_alpha + 1. ) * 1. /     poly_xt**poly_alpha )
#G = M_PI  *        X[2]         + ( ( 1. - hslope ) / 2.) *    sin(2. * M_PI *         X[2]         ); https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L80C10-L81C1 
G  = np.pi * X2_harm_coordinates + ( ( 1.  - hslope ) / 2. ) * np.sin( 2. * np.pi * X2_harm_coordinates )
#  *y   = 2. *        X[2]         - 1.; https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L85C1-L86C1
y_tilde = 2. * X2_harm_coordinates - 1.
# *thJ = poly_norm *   (*y)  * ( 1. + pow(  (*y)  / poly_xt  , poly_alpha ) / ( poly_alpha + 1.)) + 0.5 * M_PI; https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L86C3-L88C21 
J      = poly_norm * y_tilde * ( 1.0  + (   y_tilde / poly_xt )**poly_alpha / ( poly_alpha + 1 ) ) + 0.5 * np.pi 

# Do transformation from harm to spherical cordinates
#                r                       =    exp(X[1])                                                https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L92 
r_spherical_coordinates_mytransformation = np.exp(X1_harm_coordinates) * np.array(dump_file['L_unit']) # cm
#                theta                       = thG +                     exp( mks_smooth * (        startx[1]        -        X[1]         ) ) * (thJ - thG); https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L100C10-L100C67
theta_spherical_coordinates_mytransformation = G   + derefine_poles * np.exp( mks_smooth * ( min_X1_harm_coordinates - X1_harm_coordinates ) ) * ( J  -  G ) # polar angle in radians from 0 to pi
#                phi                       =          X3
phi_spherical_coordinates_mytransformation = X3_harm_coordinates # azimutal angle in radians from 0 to 2 pi

# Do transformation from cartesian to spherical cordinates
r_from_cartesians = np.sqrt( x_cartesian_coordinates**2 + y_cartesian_coordinates**2 + z_cartesian_coordinates**2 )
theta_from_cartesians = np.arccos( z_cartesian_coordinates / r_from_cartesians )
phi_from_cartesians = np.arctan2( y_cartesian_coordinates , x_cartesian_coordinates ) 
phi_from_cartesians = np.where(phi_from_cartesians < 0, phi_from_cartesians + 2 * np.pi, phi_from_cartesians) # Adjusting the angle to be between 0 and 2pi.

# Test if transformation is correct
# This compare the transformation from harm to spherical coordinates
# With the transformation from cartesian to spherical coordinates
# Both transformation should lead to the same result

diff_r = np.abs( r_from_cartesians - r_spherical_coordinates_mytransformation )
print(f"\nnp.max(diff_r) = {np.max(diff_r)}")
print(f"np.min(diff_r) = {np.min(diff_r)}")
print(f"np.average(diff_r) = {np.average(diff_r)}")

diff_theta = np.abs( theta_from_cartesians - theta_spherical_coordinates_mytransformation )
print(f"\nnp.max(diff_theta) = {np.max(diff_theta)}")
print(f"np.min(diff_theta) = {np.min(diff_theta)}")
print(f"np.average(diff_theta) = {np.average(diff_theta)}")

diff_phi = np.abs( phi_from_cartesians - phi_spherical_coordinates_mytransformation )
print(f"\nnp.max(diff_phi) = {np.max(diff_phi)}")
print(f"np.min(diff_phi) = {np.min(diff_phi)}")
print(f"np.average(diff_phi) = {np.average(diff_phi)}")

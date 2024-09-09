'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to transform a point in Spherical coordinates to Harm cordinates used in Jonah's NSM simulation.
The data files dump_00001000.h5 should be in the same directory as this script.
'''

import numpy as np
import h5py
from scipy.optimize import bisect

# importing Jonah NSM dump_file
dump_file = h5py.File('dump_00001000.h5','r')

# pull out variables required for transformation from harm to spherical coordinates 
# harm coordinate system information can be found in https://github.com/lanl/nubhlight/wiki
derefine_poles = dump_file['derefine_poles'][0]
mks_smooth = dump_file['mks_smooth'][0]
hslope = dump_file['hslope'][0]
poly_xt = dump_file['poly_xt'][0]
poly_alpha = dump_file['poly_alpha'][0]
min_X1_harm_coordinates = dump_file['startx[1]'][0]
L_unit = dump_file['L_unit'][0]

# poly_norm     = 0.5 *  M_PI * 1. / ( 1. + 1. / ( poly_alpha + 1. ) * 1. / pow(poly_xt, poly_alpha)); https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L439
poly_norm       = 0.5 * np.pi * 1. / ( 1. + 1. / ( poly_alpha + 1. ) * 1. /     poly_xt**poly_alpha )

def from_cart_to_harm_tranformation(coords):
    '''
    This function converts a point from spherical coordinates (r, theta, phi)
    to harmonic coordinates (X1, X2, X3).
    
    Arguments:
    coords -- A tuple or list containing the spherical coordinates (r, theta, phi):
              r     -- Radial distance in spherical coordinates.
              theta -- Polar angle in spherical coordinates (in radians, from 0 to pi).
              phi   -- Azimuthal angle in spherical coordinates (in radians, from 0 to 2*pi).

    Returns:
    X1, X2, X3 -- Harmonic coordinates corresponding to the input spherical coordinates.
    '''

    # Unpack the input tuple/list to get the individual spherical coordinates
    r = coords[0]  # Radial distance
    theta = coords[1]  # Polar angle in radians
    phi = coords[2]  # Azimuthal angle in radians

    # X1 is the radial component in harmonic coordinates, obtained by normalizing
    # the spherical radius (r) with a unit scale (L_unit) and applying a logarithmic transformation.
    X1 = np.log( r / L_unit )

    # X3 is directly the azimuthal angle (phi) from spherical coordinates.
    X3 = phi

    # Define a helper function to find the root of the equation needed to compute X2.
    # X2 is related to the polar angle in harmonic coordinates
    def f(X2_guess):
        '''
        Helper function to compute the difference between the guessed polar angle in harmonic coordinates (X2_guess)
        and the actual spherical polar angle (theta). We aim to find the root of this function using Brent's method.
        
        X2_guess -- Current guess for the polar angle in harmonic coordinates (between 0 and 1).
        '''

        #G = M_PI  *        X[2]         + ( ( 1. - hslope ) / 2.) *    sin(2. * M_PI *         X[2]         ); https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L80C10-L81C1 
        G  = np.pi * X2_guess + ( ( 1.  - hslope ) / 2. ) * np.sin( 2. * np.pi * X2_guess )

        #  *y   = 2. *        X[2]         - 1.; https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L85C1-L86C1
        y_tilde = 2. * X2_guess - 1.

        # *thJ = poly_norm *   (*y)  * ( 1. + pow(  (*y)  / poly_xt  , poly_alpha ) / ( poly_alpha + 1.)) + 0.5 * M_PI; https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L86C3-L88C21 
        J      = poly_norm * y_tilde * ( 1.0  + (   y_tilde / poly_xt )**poly_alpha / ( poly_alpha + 1 ) ) + 0.5 * np.pi 
        
        # 0. =          - theta + thG +                     exp( mks_smooth * (        startx[1]        -        X[1]         ) ) * (thJ - thG); https://github.com/lanl/nubhlight/blob/1f751d36a49b40e80c35f11fc7a4386d7f895d55/core/coord.c#L100C10-L100C67
        fx   =      -1. * theta + G   + derefine_poles * np.exp( mks_smooth * ( min_X1_harm_coordinates - X1 ) ) * ( J  -  G ) # polar angle in radians from 0 to pi
        
        # Return fx to find the root (we want fx = 0).
        return fx

    try:
        # Find the root of the helper function f which corresponds to X2.
        X2 = bisect(f, -0.99, 1.01)
    except Exception as e:
        # Print error if interpolation fails
        print(f"Interpolation failed: {e}. Point = ( {r:.3e} , {theta:.3e} , {phi:.3e} ) ")
        # Set a default value if solver fails
        X2 = 18081999.0

    # Return the harmonic coordinates (X1, X2, X3).
    return [ X1 , X2 , X3 ]
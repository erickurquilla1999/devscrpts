'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is reads 'particle_input.dat' file and apply a perturbations in N_eu and Nbar_eu.
The present algorithm applies a positive random perturbation to particles moving in the x direction. 
The same perturbation, but with a negative sign, is applied to particles moving in the oposite momentum direction.
The data file 'particle_input.dat' should be in the same directory as this script.
'''

import numpy as np
import random
import matplotlib.pyplot as plt                                                                                                                  

# Reading initial condition data
n_flavors = np.loadtxt('particle_input.dat', max_rows=1) # read number of flavor of particles in the initial condition script
data = np.array(np.loadtxt('particle_input.dat', skiprows=1)) # read particles initial condition

# This script works only for 2-flavor simulations.
assert n_flavors == 2, 'This script works only for 2-flavor simulations'

E = data[...,3][0] # energy
p = data[...,0:3] / E # unit momentum vector

minus_p_indices = [] # indices of the particle with oposite momentum

# Loop over all particles
for i in range(len(p)):
    pi = np.tile(np.array(p[i]), (len(p),1)) # Create an array with a length equal to the number of particles, where all elements correspond to the momentum of the i-th particle.
    delta_p = p - pi # Compute the difference between the momentum of the i-th particle and the momentum of all other particles.
    delta_p_mag = np.sqrt(delta_p[...,0]**2+delta_p[...,1]**2+delta_p[...,2]**2) # Compute the magnitude of the vector \(\Delta p\).
    minus_p_indices.append(np.argmax(delta_p_mag)) # The index of the maximum value of \(\Delta p\) corresponds to the particle with momentum opposite to particle \(i\).

# Check that there are no repeated indices in the minus_p_indices list.
# This checks that the initial conditions do not have repeated momentum beams or beams in close directions.
counts = np.bincount(minus_p_indices)
has_repeats = np.any(counts > 1)
assert has_repeats == False, 'The initial conditions have repeated momentum beams or beams in close directions'

# Perturbation amplitud
perturbation_amplitud = 1.0e-6

#Loop over half of the particles
for i in range(int(len(p)/2)):
    if p[i][0] > 0 :
        # Applied a positive perturbation to the particles with momentum components in the positive x direction.
        data[i][5]  += 0.5 * ( data[i][4] + data[i][7] )  * perturbation_amplitud * random.random()
        data[i][6]  += 0.5 * ( data[i][4] + data[i][7] )  * perturbation_amplitud * random.random()
        data[i][9]  += 0.5 * ( data[i][8] + data[i][11] ) * perturbation_amplitud * random.random()
        data[i][10] += 0.5 * ( data[i][8] + data[i][11] ) * perturbation_amplitud * random.random()
        # Applied a negative perturbation to the particles with momentum components in the positive x direction.
        data[minus_p_indices[i]][5] += -1.0 * data[i][5]
        data[minus_p_indices[i]][6] += -1.0 * data[i][6]
        data[minus_p_indices[i]][9] += -1.0 * data[i][9]
        data[minus_p_indices[i]][10] += -1.0 * data[i][10]
    elif p[i][0] < 0:
        # Applied a positive perturbation to the particles with momentum components in the positive x direction.
        data[minus_p_indices[i]][5]  += 0.5 * ( data[i][4] + data[i][7] )  * perturbation_amplitud * random.random()
        data[minus_p_indices[i]][6]  += 0.5 * ( data[i][4] + data[i][7] )  * perturbation_amplitud * random.random()
        data[minus_p_indices[i]][9]  += 0.5 * ( data[i][8] + data[i][11] ) * perturbation_amplitud * random.random()
        data[minus_p_indices[i]][10] += 0.5 * ( data[i][8] + data[i][11] ) * perturbation_amplitud * random.random()
        # Applied a negative perturbation to the particles with momentum components in the positive x direction.
        data[i][5] += -1.0 * data[i][5]
        data[i][6] += -1.0 * data[i][6]
        data[i][9] += -1.0 * data[i][9]
        data[i][10] += -1.0 * data[i][10]        

# Save new initial conditions
np.savetxt('particle_input.dat', data, delimiter=' ', fmt='%.16e', header = str(int(n_flavors)), comments='')

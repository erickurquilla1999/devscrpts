'''
This script is used to interpolate fluid data (rho, Ye, T) from Jonah's NSM simulation to the EMU Cartesian grid.
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
The data files dump_00001000.h5 and grid.h5 should be in the same directory as this script.
'''

import numpy as np
import h5py

# importing Jonah NSM data
dump_file = h5py.File('dump_00001000.h5','r')
grid_file = h5py.File('grid.h5','r')

# getting grid in harm coordinates system
r_harm  = grid_file['Xharm']
X1_harm_coordinates = r_harm[...,1]
X2_harm_coordinates = r_harm[...,2]
X3_harm_coordinates = r_harm[...,3]

# getting grid in cartesian coordinates system
r_cartesian = grid_file['Xcart'] * np.array(dump_file['L_unit']) # cm
x_cartesian_coordinates = r_cartesian[...,1]
y_cartesian_coordinates = r_cartesian[...,2]
z_cartesian_coordinates = r_cartesian[...,3]

# pull out variables required for transformation from harm to spherical coordinates 
# harm coordinate system information can be found in https://github.com/lanl/nubhlight/wiki
derefine_poles = dump_file['derefine_poles'][0]
mks_smooth = dump_file['mks_smooth'][0]
hslope = dump_file['hslope'][0]
poly_xt = dump_file['poly_xt'][0]
poly_alpha = dump_file['poly_alpha'][0]
poly_norm = 1. / ( 1. + 1. / ( poly_alpha + 1. ) * 1. / poly_xt**poly_alpha )
G = np.pi * X2_harm_coordinates + ( ( 1 - hslope ) / 2 ) * np.sin( 2 * np.pi * X2_harm_coordinates )
y_tilde = 2 * X2_harm_coordinates - 1
J = poly_norm * y_tilde * ( 1 + ( y_tilde / poly_xt )**poly_alpha ) / ( poly_alpha + 1 ) + np.pi / 2
min_X1_harm_coordinates = np.min( X1_harm_coordinates )

# Do transformation from harm to spherical cordinates
r_spherical_coordinates_mytransformation = np.exp(X1_harm_coordinates) * np.array(dump_file['L_unit']) # cm
theta_spherical_coordinates_mytransformation = G + derefine_poles * np.exp( mks_smooth * ( min_X1_harm_coordinates - X1_harm_coordinates ) ) * ( J - G ) # polar angle in radians from 0 to pi
phi_spherical_coordinates_mytransformation = X3_harm_coordinates # azimutal angle in radians from 0 to 2 pi

# Do transformation from spherical to cartesian cordinates
x_cartesian_coordinates_mytransformation = r_spherical_coordinates_mytransformation * np.sin( theta_spherical_coordinates_mytransformation ) * np.cos( phi_spherical_coordinates_mytransformation )
y_cartesian_coordinates_mytransformation = r_spherical_coordinates_mytransformation * np.sin( theta_spherical_coordinates_mytransformation ) * np.sin( phi_spherical_coordinates_mytransformation )
z_cartesian_coordinates_mytransformation = r_spherical_coordinates_mytransformation * np.cos( theta_spherical_coordinates_mytransformation )

# Compare my transformation with cartisian grid data already in grid.h5 file

print(f'x_cartesian_coordinates = {x_cartesian_coordinates} ')
print(f'x_cartesian_coordinates_mytransformation = {x_cartesian_coordinates_mytransformation} ')

print(f'x_cartesian_coordinates_mytransformation == x_cartesian_coordinates = {x_cartesian_coordinates_mytransformation == x_cartesian_coordinates}')
print(f'np.sort(x_cartesian_coordinates_mytransformation.flatten()) == np.sort(x_cartesian_coordinates.flatten()) = {np.sort(x_cartesian_coordinates_mytransformation.flatten()) == np.sort(x_cartesian_coordinates.flatten())}')

firts_cell_harm_coordinate_system = [ X1_harm_coordinates[0][0][0] , X2_harm_coordinates[0][0][0] , X3_harm_coordinates[0][0][0] ]
firts_cell_sphercial_coordinate_system = [ r_spherical_coordinates[0][0][0] , theta_spherical_coordinates[0][0][0] , phi_spherical_coordinates[0][0][0] ]
firts_cell_cartesian_coordinate_system = [ 0 , 0 , 0 ]
firts_cell_cartesian_coordinate_system[0] = firts_cell_sphercial_coordinate_system[0] * np.sin(firts_cell_sphercial_coordinate_system[1]) * np.cos(firts_cell_sphercial_coordinate_system[2])
firts_cell_cartesian_coordinate_system[1] = firts_cell_sphercial_coordinate_system[0] * np.sin(firts_cell_sphercial_coordinate_system[1]) * np.sin(firts_cell_sphercial_coordinate_system[2])
firts_cell_cartesian_coordinate_system[2] = firts_cell_sphercial_coordinate_system[0] * np.cos(firts_cell_sphercial_coordinate_system[1])

print(f'firts_cell_harm_coordinate_system = {firts_cell_harm_coordinate_system}')
print(f'firts_cell_sphercial_coordinate_system = {firts_cell_sphercial_coordinate_system}')
print(f'r = {np.sqrt(firts_cell_cartesian_coordinate_system[0]**2 + firts_cell_cartesian_coordinate_system[1]**2 + firts_cell_cartesian_coordinate_system[2]**2)}')
print(f'firts_cell_cartesian_coordinate_system = {firts_cell_cartesian_coordinate_system}')
r_cart = grid_file['Xcart'] * np.array(dump_file['L_unit']) # cm
print(f'r_cart.shape = {r_cart.shape}')
print(f'r_cart[0][0][0] = {r_cart[0][0][0]}')
print(f'r = { np.sqrt( r_cart[0][0][0][1]**2 + r_cart[0][0][0][2]**2 + r_cart[0][0][0][3]**2 )}')


quit()


print(f'r_spherical_coordinates.shape = {r_spherical_coordinates.shape}')
print(f'theta_spherical_coordinates.shape = {theta_spherical_coordinates.shape}')
print(f'phi_spherical_coordinates.shape = {phi_spherical_coordinates.shape}')

print(f'np.min(r_spherical_coordinates) = {np.min(r_spherical_coordinates)}')
print(f'np.max(r_spherical_coordinates) = {np.max(r_spherical_coordinates)}')

print(f'np.min(theta_spherical_coordinates) = {np.min(theta_spherical_coordinates)}')
print(f'np.max(theta_spherical_coordinates) = {np.max(theta_spherical_coordinates)}')

print(f'np.min(phi_spherical_coordinates) = {np.min(phi_spherical_coordinates)}')
print(f'np.max(phi_spherical_coordinates) = {np.max(phi_spherical_coordinates)}')

print(f'X1.shape = {X1.shape}')


print(f'np.min(X1) = {np.min(X1)}')
print(f'np.max(X1) = {np.max(X1)}')
print(f'np.min(X3) = {np.min(X3)}')
print(f'np.max(X3) = {np.max(X1)}')

r = np.sqrt(X1**2 + X2**2 + X3**2)
r_min = np.min(r) # this black hole radius
mask = r < 7

X1 = X1[mask]
X2 = X2[mask]
X3 = X3[mask]

print(f'X1.shape = {X1.shape}')
print(f'X2.shape = {X2.shape}')

mask_x0 = X3 < 0.05
X1 = X1[mask_x0]
X2 = X2[mask_x0]



print(f'X1.shape = {X1.shape}')
print(f'X2.shape = {X2.shape}')
print(f'r_harm.shape = {r_harm.shape}')




import matplotlib.pyplot as plt

# plot x=0: Grid points
fig,ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(X1,X2,s=1)
ax.tick_params(axis='both',which='both',direction='in',right=True,top=True)
ax.set_xlabel(r"$X_1$")
ax.set_ylabel(r"$X_2$")
ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
fig.savefig('Harm_grid_X2X3.pdf',bbox_inches='tight')
plt.close(fig)









r_code = np.exp(grid_file['Xharm'][:,0,0,1])

quit()





dr_code = r_code[1:] - r_code[:-1]
min_dr = np.min(dr_code)
print("min dr in [code,cm] =",min_dr,min_dr*dump_file['L_unit'][0])
print("Rin, Reh, Risco =", dump_file['Rin'][0],dump_file['Reh'][0],dump_file['Risco'][0])










index_Ye = np.where(dump_file['P'].attrs['vnams'] == 'Ye')[0][0]
index_rho = np.where(dump_file['P'].attrs['vnams'] == 'RHO')[0][0]

# get Ye, rho, T and cartesian coordinates
Ye = dump_file['P'][:,:,:,index_Ye] # no units
rho = dump_file['P'][:,:,:,index_rho] * dump_file['RHO_unit'] # g / ccm
T = dump_file['TEMP'] # MeV
r_cart = grid_file['Xcart'] * np.array(dump_file['L_unit']) # cm









print(f'Ye.shape = {Ye.shape}')
print(f'r_cart.shape = {r_cart.shape}')

par_1 = r_cart[0:5, 0:5, 0:5]
x1 = par_1[...,1] # cm
y1 = par_1[...,2] # cm
z1 = par_1[...,3] # cm

par_2 = r_cart[0:5, 5:10, 0:5]
x2 = par_2[...,1] # cm
y2 = par_2[...,2] # cm
z2 = par_2[...,3] # cm


import matplotlib.pyplot as plt

# plot x=0: Grid points
fig,ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(x1,y1,s=1,color = 'C0')
ax.scatter(x2,y2,s=1,color = 'C1')
ax.tick_params(axis='both',which='both',direction='in',right=True,top=True)
ax.set_xlabel(r"$x$ (cm)")
ax.set_ylabel(r"$y$ (cm)")
ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
fig.savefig('xy.pdf',bbox_inches='tight')
plt.close(fig)

fig,ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(x1,z1,s=1,color = 'C0')
ax.scatter(x2,z2,s=1,color = 'C1')
ax.tick_params(axis='both',which='both',direction='in',right=True,top=True)
ax.set_xlabel(r"$x$ (cm)")
ax.set_ylabel(r"$z$ (cm)")
ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
fig.savefig('xz.pdf',bbox_inches='tight')
plt.close(fig)

fig,ax = plt.subplots(1,1, figsize=(8,6))
ax.scatter(y1,z1,s=1,color = 'C0')
ax.scatter(y2,z2,s=1,color = 'C1')
ax.tick_params(axis='both',which='both',direction='in',right=True,top=True)
ax.set_xlabel(r"$y$ (cm)")
ax.set_ylabel(r"$z$ (cm)")
ax.xaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
ax.yaxis.grid(True, linestyle='--', linewidth=0.5, color='gray')
fig.savefig('yz.pdf',bbox_inches='tight')
plt.close(fig)








###########################################
###########################################
# selecting data closer to the black hole

R_from_bh_center = 1e6 # cm : data out of this distance from black hole center will not be consider
x = r_cart[...,1] # cm
y = r_cart[...,2] # cm
z = r_cart[...,3] # cm
r = np.sqrt(x**2 + y**2 + z**2)
r_black_hole = np.min(r) # cm , this black hole radius
mask = r < R_from_bh_center

# print(f'x = {x}')
print(f'len(x) = {len(x)}')
print(f'len(x[0]) = {len(x[0])}')
print(f'len(x[0][0]) = {len(x[0][0])}')

print(f'\nx[0][0] = {x[0][0]}')
print(f'\nx[0][1] = {x[0][1]}')

print(f'x.shape = {x.shape}')


# data to be used in the interpolation
Ye = Ye[mask] # no units
print(f'Ye.shape = {Ye.shape}')
T = T[mask] # MeV
rho = rho[mask] # g / ccm
xsim = x[mask] # cm
ysim = y[mask] # cm
zsim = z[mask] # cm

###########################################
###########################################
# Creating a EMU grid 

def create_grid(cell_numbers, dimensions):
    faces = [np.linspace(-d, d, n+1) for d,n in zip(dimensions, cell_numbers)]
    centers = [0.5*(f[1:] + f[:-1]) for f in faces]
    X,Y,Z = np.meshgrid(*centers, indexing='ij')
    
    # X1 = np.log(np.sqrt(X**2+y**2+z**2))
    # X3 = np.arta
    centers = np.vstack((X.flatten(),Y.flatten(),Z.flatten())).transpose()
    mesh = np.array([X,Y,Z])
    return centers, mesh

# Define cell numbers and dimensions
# n_cell_x = round( R_from_bh_center / min_dr )
# n_cell_y = round( R_from_bh_center / min_dr )
# n_cell_z = round( R_from_bh_center / min_dr )
n_cell_x = 31
n_cell_y = 31
n_cell_z = 31
cell_numbers = [n_cell_x, n_cell_y, n_cell_z]
dimensions   = [R_from_bh_center,R_from_bh_center,R_from_bh_center]

# Generate mesh
centers, mesh = create_grid(cell_numbers, dimensions) # cm
T_interpolation = np.zeros((n_cell_x,n_cell_y,n_cell_z)) # MeV
Ye_interpolation = np.zeros((n_cell_x,n_cell_y,n_cell_z)) # No units
rho_interpolation = np.zeros((n_cell_x,n_cell_y,n_cell_z)) # g / ccm

# Getting the inidices of the cell center's position to be interpolated
r = np.sqrt(mesh[0]**2+mesh[1]**2+mesh[2]**2)
mask_bh_domain = r > r_black_hole
mask_R_max     = r < R_from_bh_center
mask_combined = mask_bh_domain & mask_R_max
indices = np.where(mask_combined)

###########################################
###########################################
# Perform interpolation

from scipy.interpolate import griddata, RegularGridInterpolator
import multiprocessing

points = np.stack((xsim, ysim, zsim), axis=-1)
data = T

def interpolation_function(point_to_interpolate, index):
    interpolation_value = griddata(points, data, point_to_interpolate)
    return interpolation_value

class Simulation_Data:

    def __init__(self, T, rho, Ye, coordinates):
        self.temperature = T
        self.electron_fraction = Ye
        self.density = rho
        self.coordinates = coordinates
    
    def interpolate(self, point_to_interpolate, index):
        T_in_interpolation_point = griddata(self.coordinates, self.temperature, point_to_interpolate)
        Ye_in_interpolation_point = griddata(self.coordinates, self.electron_fraction, point_to_interpolate)
        rho_in_interpolation_point = griddata(self.coordinates, self.density, point_to_interpolate)
        return np.array([T_in_interpolation_point,Ye_in_interpolation_point,rho_in_interpolation_point]), index

JonahSim = Simulation_Data(T, rho, Ye, points)
print(JonahSim.interpolate([0,0.7e6,0],[0,0,0]))    

quit()

for idx in zip(*indices):
    
    start = time.time()
    interpolation_point = np.array([mesh[0][idx],mesh[1][idx],mesh[2][idx]])
    r_interpolation_point = np.sqrt(interpolation_point[0]**2+interpolation_point[1]**2+interpolation_point[2]**2)
    T_interpolation[idx] = griddata(points, T, interpolation_point)
    # T_interpolation[idx] = interpolation_function(interpolation_point, idx)
    end = time.time()

    print(f'\nidx = {idx}')
    print(f'interpolation_point = {interpolation_point}')
    print(f'r_black_hole = {r_black_hole}')
    print(f'R_from_bh_center = {R_from_bh_center}')
    print(f'r_interpolation_point = {r_interpolation_point}')
    print(f'Time for {idx} = {end - start}')
    print(f'T_interpolation = {T_interpolation[idx]}')
    
T_interpolation[np.isnan(T_interpolation)] = 0.0

quit()
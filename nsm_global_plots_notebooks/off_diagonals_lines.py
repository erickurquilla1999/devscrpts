import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator
import glob
import cv2
import os
import re

data_directory = '/pscratch/sd/b/bigqke/gw170817_global_simulations/qke_res_cla_sim/92ppEb/att_01.00e-05_cdt_100m_dom_96-96-32km_ncells_192-192-64'

mesh_files = glob.glob(os.path.join(data_directory, 'mesh_plt*.h5'))
mesh_files_nums = [filenames.split('mesh_plt')[1].split('.h5')[0] for filenames in mesh_files]
mesh_files_sorted = [f for _, f in sorted(zip(mesh_files_nums, mesh_files))]
mesh_files_sorted = np.array(mesh_files_sorted)
nfiles = len(mesh_files_sorted)
print(f'Found {len(mesh_files_sorted)} mesh files.')

rho_ye_T_h5py = h5py.File(data_directory+'/rho_Ye_T.hdf5', 'r')

# number of cells
Nx = np.array(rho_ye_T_h5py['/ncellsx'])
Ny = np.array(rho_ye_T_h5py['/ncellsy'])
Nz = np.array(rho_ye_T_h5py['/ncellsz'])
print(f'Nx = {Nx}, Ny = {Ny}, Nz = {Nz}')

# cell size
dx = ( np.array(rho_ye_T_h5py['/xmax_cm']) - np.array(rho_ye_T_h5py['/xmin_cm']) ) / np.array(rho_ye_T_h5py['/ncellsx']) # cm
dy = ( np.array(rho_ye_T_h5py['/ymax_cm']) - np.array(rho_ye_T_h5py['/ymin_cm']) ) / np.array(rho_ye_T_h5py['/ncellsy']) # cm
dz = ( np.array(rho_ye_T_h5py['/zmax_cm']) - np.array(rho_ye_T_h5py['/zmin_cm']) ) / np.array(rho_ye_T_h5py['/ncellsz']) # cm
print(f'dx = {dx} cm, dy = {dy} cm, dz = {dz} cm')

# cell centers
xc = np.linspace(dx / 2, dx * (Nx - 0.5), Nx) # cm
yc = np.linspace(dy / 2, dy * (Ny - 0.5), Ny) # cm
zc = np.linspace(dz / 2, dz * (Nz - 0.5), Nz) # cm

# cell centers mesh
Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij') # cm

bh_radius = 5.43e+05 # cm
bh_center_x = 48.0e+5 # cm
bh_center_y = 48.0e+5 # cm
bh_center_z = 16.0e+5 # cm

distance_from_bh = np.sqrt( (Xc - bh_center_x)**2 + (Yc - bh_center_y)**2 + (Zc - bh_center_z)**2 )
mask_bh = (distance_from_bh < bh_radius + 1*np.max([dx,dy,dz]))
mask_x = (Xc < dx) | (Xc > (Nx - 1) * dx)
mask_y = (Yc < dy) | (Yc > (Ny - 1) * dy)
mask_z = (Zc < dz) | (Zc > (Nz - 1) * dz)
mask = mask_bh | mask_x | mask_y | mask_z

rho_ye_T_h5py.close()

x_slice_indx = 96 # cell number
y_slice_indx = 96 # cell number
z_slice_indx = 32 # cell number
time_indx    = -1 # last time step

print(f'mesh_files_sorted[-1]={mesh_files_sorted[-1]}')
alldatafin_h5 = h5py.File(mesh_files_sorted[-1], 'r')
alldatafin_dict = {key: np.array(alldatafin_h5[key]) for key in alldatafin_h5.keys()}
alldatafin_h5.close()
alldatafin = alldatafin_dict.copy()
zup_prev_fin  = np.sqrt(np.array(alldatafin['N01_Re(1|ccm)'])[1:-1, 1:-1, z_slice_indx]**2+np.array(alldatafin['N01_Im(1|ccm)'])[1:-1, 1:-1, z_slice_indx]**2) / ( np.array(alldatafin['N00_Re(1|ccm)'])[1:-1, 1:-1, z_slice_indx] + np.array(alldatafin['N11_Re(1|ccm)'])[1:-1, 1:-1, z_slice_indx] + np.array(alldatafin['N22_Re(1|ccm)'])[1:-1, 1:-1, z_slice_indx] )
zlow_prev_fin = np.sqrt(np.array(alldatafin['N01_Re(1|ccm)'])[1:-1, y_slice_indx, 1:-1]**2+np.array(alldatafin['N01_Im(1|ccm)'])[1:-1, y_slice_indx, 1:-1]**2) / ( np.array(alldatafin['N00_Re(1|ccm)'])[1:-1, y_slice_indx, 1:-1] + np.array(alldatafin['N11_Re(1|ccm)'])[1:-1, y_slice_indx, 1:-1] + np.array(alldatafin['N22_Re(1|ccm)'])[1:-1, y_slice_indx, 1:-1] )

# for filepath in mesh_files_sorted[::100]:
for filepath in [mesh_files_sorted[-1]]:
# for filepath in mesh_files_sorted:
# for filepath in mesh_files_sorted[279:]:

    print(f'Processing file: {filepath}')

    alldata_h5 = h5py.File(filepath, 'r')
    alldata_dict = {key: np.array(alldata_h5[key]) for key in alldata_h5.keys()}
    alldata_h5.close()
    alldata = alldata_dict.copy()
    print("Keys in alldata_dict:", list(alldata.keys()))

    # if time_indx==0:
    if True:
        doshow = False
    else:
        doshow = False

    time_ms = np.array(alldata['t(s)']) * 1.0e3 # ms
    print("time_ms=", time_ms)

    ###############################################
    # Neutrino off diagonal terms
    ###############################################

    rho  = np.sqrt(np.array(alldata['N01_Re(1|ccm)'])**2+np.array(alldata['N01_Im(1|ccm)'])**2) / ( np.array(alldata['N00_Re(1|ccm)']) + np.array(alldata['N11_Re(1|ccm)']) + np.array(alldata['N22_Re(1|ccm)']))

    rho_z_axis = rho[x_slice_indx, y_slice_indx, :]
    z_axis = Zc[x_slice_indx, y_slice_indx, :]
    fig, ax = plt.subplots()
    ax.plot(z_axis, rho_z_axis)
    ax.set_xlabel("z")
    ax.set_ylabel("rho_01")
    ax.grid(True)
    fig.savefig("plots/hsr_rho_z_axis.pdf", format="pdf", bbox_inches="tight")

    rho_x_axis = rho[:, y_slice_indx, z_slice_indx]
    x_axis = Xc[:, y_slice_indx, z_slice_indx]
    fig, ax = plt.subplots()
    ax.plot(x_axis, rho_x_axis)
    ax.set_xlabel("x")
    ax.set_ylabel("rho_01")
    ax.grid(True)
    fig.savefig("plots/hsr_rho_x_axis.pdf", format="pdf", bbox_inches="tight")
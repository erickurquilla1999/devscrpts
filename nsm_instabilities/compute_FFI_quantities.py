import numpy as np
import glob
import h5py
import time
import multiprocessing as mp

import FFI_functions as ffi
import os

NF = 3 # number of flavors
cellvolume = 1.0e5**3 # ccm

# Get the cell indices
cell_file_names = glob.glob('cell_*_*_*')
cell_file_names = [file_name.split('/')[-1] for file_name in cell_file_names]
x_cell_ind = np.array([int(file_name.split('_')[1]) for file_name in cell_file_names])
y_cell_ind = np.array([int(file_name.split('_')[2]) for file_name in cell_file_names])
z_cell_ind = np.array([int((file_name.split('_')[3]).split('.')[0]) for file_name in cell_file_names])
cell_indices = np.array(list(zip(x_cell_ind, y_cell_ind, z_cell_ind)))

# Get the cell indices for ix fix but iy and iz varying all available cells
x_idx_slice = 48
mask_yz_slice = cell_indices[:,0] == x_idx_slice # fixing the x index in this value
cell_indices_yz_slice = cell_indices[mask_yz_slice]

# Get the cell indices for iy fix but ix and iz varying all available cells
y_idx_slice = 48
mask_xz_slice = cell_indices[:,1] == y_idx_slice # fixing the y index in this value
cell_indices_xz_slice = cell_indices[mask_xz_slice]

# Get the cell indices for iz fix but ix and iy varying all available cells
z_idx_slice = 16
mask_xy_slice = cell_indices[:,2] == z_idx_slice # fixing the z index in this value
cell_indices_xy_slice = cell_indices[mask_xy_slice]

# Combine all cell indices from the three slices
# cell_indices_all = np.concatenate((cell_indices_yz_slice, cell_indices_xz_slice, cell_indices_xy_slice), axis=0)
cell_indices_all = np.concatenate((cell_indices_xz_slice, cell_indices_xy_slice), axis=0)
cell_indices_all = np.unique(cell_indices_all, axis=0)

cell_indices = cell_indices_all

print('Number of cells:', len(cell_indices))

def writehdf5file(indexpair):
    i, j, k = indexpair
    h5_filename = f'FFI_cell_{i}_{j}_{k}.h5'
    if os.path.exists(h5_filename):
        print(f'File {h5_filename} already exists. Skipping calculation.')
        return
    sigma, GnPos, GnNeg = ffi.compute_sigma_GnPos_GnNeg(i, j, k, '.', cellvolume, NF)
    with h5py.File(h5_filename, 'w') as h5f:
        h5f.create_dataset('sigma_inverse_s', data=sigma)
        h5f.create_dataset('GnPos_inverse_s', data=GnPos)
        h5f.create_dataset('GnNeg_inverse_s', data=GnNeg)

if __name__ == '__main__':
    start_time = time.time()
    nproc = mp.cpu_count()
    print(f'Using {nproc} processes for parallel computation.')
    with mp.Pool(nproc) as pool:
        pool.map(writehdf5file, cell_indices)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")

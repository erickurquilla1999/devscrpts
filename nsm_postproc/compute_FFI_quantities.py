import numpy as np
import glob
import h5py
import time
import multiprocessing as mp

import FFI_functions as ffi
import os


# Get the cell indices
cell_file_names = glob.glob('cell_*_*_*')
cell_file_names = [file_name.split('/')[-1] for file_name in cell_file_names]
x_cell_ind = np.array([int(file_name.split('_')[1]) for file_name in cell_file_names])
y_cell_ind = np.array([int(file_name.split('_')[2]) for file_name in cell_file_names])
z_cell_ind = np.array([int((file_name.split('_')[3]).split('.')[0]) for file_name in cell_file_names])
cell_indices = np.array(list(zip(x_cell_ind, y_cell_ind, z_cell_ind)))
print('Number of cells:', len(cell_indices))

cellvolume = 1.0  # ccm

def writehdf5file(indexpair):
    i, j, k = indexpair
    h5_filename = f'FFI_cell_{i}_{j}_{k}.h5'
    if os.path.exists(h5_filename):
        print(f'File {h5_filename} already exists. Skipping calculation.')
        return
    sigma, GnPos, GnNeg = ffi.compute_sigma_GnPos_GnNeg(i, j, k, '.', cellvolume)
    with h5py.File(h5_filename, 'w') as h5f:
        h5f.create_dataset('sigma_inverse_s', data=sigma)
        h5f.create_dataset('GnPos_inverse_s', data=GnPos)
        h5f.create_dataset('GnNeg_inverse_s', data=GnNeg)
    print(f'Cell ({i}, {j}, {k}): sigma = {sigma:.3e} 1/s, GnPos = {GnPos:.3e} 1/s, GnNeg = {GnNeg:.3e} 1/s')

if __name__ == '__main__':
    start_time = time.time()
    nproc = mp.cpu_count()
    print(f'Using {nproc} processes for parallel computation.')
    with mp.Pool(nproc) as pool:
        pool.map(writehdf5file, cell_indices)
    end_time = time.time()
    print(f"Processing time: {end_time - start_time} seconds")

import numpy as np
from scipy.interpolate import griddata
from pathlib import Path
import h5py

def get_unique_momentum(momentum, tolerance=1e-5):
    """
    Filters out unique momentum vectors within a given tolerance.

    Parameters:
    momentum (np.ndarray): Array of momentum vectors.
    tolerance (float): Absolute tolerance for considering two vectors as equal.

    Returns:
    np.ndarray: Array of unique momentum vectors.
    """
    if momentum.size == 0:
        return np.array([])  # Handle empty input

    unique_indices = []  # Store indices of unique vectors

    for i, vec in enumerate(momentum):
        if not unique_indices:  # First vector is always unique
            unique_indices.append(i)
            continue

        # Compare vec with already stored unique vectors
        unique_momentum_np = momentum[np.array(unique_indices)]  # Select only unique ones so far

        if not np.any(np.all(np.isclose(unique_momentum_np, vec, atol=tolerance), axis=1)):
            unique_indices.append(i)

    return np.real(momentum[np.array(unique_indices)])  # Return unique momenta

def compute_unique_fluxes(momentum, flux, unique_momentum, tolerance=1e-5):
    """
    Computes the total flux for each unique momentum vector using approximate equality.

    Parameters:
    momentum (np.ndarray): Array of momentum vectors (N, D).
    flux (np.ndarray): Array of flux values corresponding to momentum vectors (N, 3).
    unique_momentum (np.ndarray): Array of unique momentum vectors (M, D).
    tolerance (float): Absolute tolerance for considering two vectors as equal.

    Returns:
    tuple:
        - unique_fluxes (np.ndarray): Summed flux values for each unique momentum (M, 3).
        - unique_fluxes_mag (np.ndarray): Magnitudes of the unique flux vectors (M,).
    """
    unique_fluxes = np.zeros((len(unique_momentum), 3), dtype=complex)

    # Vectorized computation of masks for all unique momenta
    for i, um in enumerate(unique_momentum):
        mask = np.all(np.isclose(momentum, um, atol=tolerance), axis=1)  # Approximate equality check
        unique_fluxes[i] = np.sum(flux[mask], axis=0)  # Sum corresponding flux values along axis 0

    # Compute magnitude of flux vectors
    unique_fluxes_mag = np.linalg.norm(unique_fluxes, axis=1)  # Equivalent to sqrt(sum(abs(fluxes)**2))

    return unique_fluxes, unique_fluxes_mag

def do_interpolation(phi, theta, data):
    """
    Interpolates irregularly sampled data on a spherical grid defined by phi and theta.

    This function handles periodic boundary conditions in the azimuthal angle (phi) by duplicating
    data near the 0 and 2π boundaries, ensuring smooth interpolation across the seam. It then
    interpolates the input data onto a regular grid in (phi, mu), where mu = cos(theta).

    Parameters
    ----------
    phi : array_like
        Array of azimuthal angles (in radians), shape (N,).
    theta : array_like
        Array of polar angles (in radians), shape (N,).
    data : array_like
        Array of data values corresponding to each (phi, theta) pair, shape (N,).

    Returns
    -------
    phi_fi : ndarray
        Extended array of azimuthal angles after periodic boundary handling.
    mu_fi : ndarray
        Extended array of mu = cos(theta) values after periodic boundary handling.
    data_fi : ndarray
        Extended array of data values after periodic boundary handling.
    phi_centers : ndarray
        1D array of phi bin centers for the regular grid.
    mu_centers : ndarray
        1D array of mu bin centers for the regular grid.
    eln_xln_interpolated : ndarray
        2D array of interpolated data values on the regular (phi, mu) grid.
    """

    phi_fi = phi
    mu_fi = np.cos(theta)
    data_fi = data

    mask_phi_l = phi_fi <   0.0   + np.pi / 2
    mask_phi_r = phi_fi > 2*np.pi - np.pi / 2

    phi_fi_new_l     = phi_fi    [mask_phi_l] + 2 * np.pi
    mu_fi_new_l      = mu_fi     [mask_phi_l]
    data_fi_new_l = data_fi[mask_phi_l]

    phi_fi_new_r     = phi_fi    [mask_phi_r] - 2 * np.pi
    mu_fi_new_r      = mu_fi     [mask_phi_r]
    data_fi_new_r = data_fi[mask_phi_r]

    phi_fi     = np.concatenate((phi_fi,     phi_fi_new_l,     phi_fi_new_r))
    mu_fi      = np.concatenate((mu_fi,      mu_fi_new_l,      mu_fi_new_r))
    data_fi = np.concatenate((data_fi, data_fi_new_l, data_fi_new_r))

    n_phi = 80 # between 0 and 2pi
    n_mu  = 50 # between -1 and 1
    phi_faces = np.linspace(np.min(phi_fi), np.max(phi_fi), n_phi + 1)
    mu_faces = np.linspace(np.min(mu_fi), np.max(mu_fi), n_mu + 1)
    phi_centers = (phi_faces[:-1] + phi_faces[1:]) / 2
    mu_centers = (mu_faces[:-1] + mu_faces[1:]) / 2
    phi_centers_mesh, mu_centers_mesh = np.meshgrid(phi_centers, mu_centers)

    # Interpolate the value of eln_xln based on the irregular data of phi_fi, mu_fi, and data_fi
    eln_xln_interpolated = griddata(
        points=(phi_fi, mu_fi),
        values=data_fi,
        xi=(phi_centers_mesh, mu_centers_mesh),
        method='linear'
    )
    return phi_fi, mu_fi, data_fi, phi_centers, mu_centers, eln_xln_interpolated

def load_particle_data(cell_x: int, cell_y: int, cell_z: int, directory: str):
    """
    Loads particle data from HDF5 files located in a 3x3x3 grid of neighboring cells.
    
    Parameters:
        cell_x (int): X-index of the target cell.
        cell_y (int): Y-index of the target cell.
        cell_z (int): Z-index of the target cell.
        directory (str): Path to the directory containing the HDF5 files.

    Returns:
        dict: A dictionary where keys are dataset names, and values are NumPy arrays
              containing concatenated data from cell (cell_x, cell_y, cell_z) and also neighboring cells..
        dict: A dictionary where keys are dataset names, and values are NumPy arrays
              containing concatenated data only from cell (cell_x, cell_y, cell_z).
    """

    particle_data_center_cell = {} # Dictionary with data only in cell (cell_x, cell_y, cell_z).
 
    file_path_center_cell = Path(f"{directory}/cell_{cell_x}_{cell_y}_{cell_z}.h5")
    print(f"Loading data from {file_path_center_cell}")
 
    if file_path_center_cell.exists():
        with h5py.File(file_path_center_cell, 'r') as hdf_file:
            # print(hdf_file.keys())
            for dataset_name in hdf_file.keys():
                data_array = np.array(hdf_file[dataset_name])
                particle_data_center_cell[dataset_name] = data_array

    return particle_data_center_cell
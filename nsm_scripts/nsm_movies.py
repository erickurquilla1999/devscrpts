'''
Created by Erick Urquilla, Department of Physics and Astronomy, University of Tennessee, Knoxville.
This script is used to generate an mp4 movies of the NSM simulation data.
Run first the script Emu/Scripts/data_reduction/convertToHDF5.py in the data directory
Then run this script to generate the MP4 movies.
'''

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

############################################################
############################################################
# PLOT SETTINGS
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator
import os
import cv2
import glob

# Font settings
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=True)

# Tick settings
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.major.pad'] = 8
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 2

# Axis linewidth
mpl.rcParams['axes.linewidth'] = 2

# Tick direction and enabling ticks on all sides
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True

# Function to apply custom tick locators and other settings to an Axes object
def apply_custom_settings(ax, leg, log_scale_y=False):

    if log_scale_y:
        # Use LogLocator for the y-axis if it's in log scale
        ax.set_yscale('log')
        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
    else:
        # Use AutoLocator for regular scales
        ax.yaxis.set_major_locator(AutoLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    
    # Apply the AutoLocator for the x-axis
    ax.xaxis.set_major_locator(AutoLocator())
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    
    # Legend settings
    leg.get_frame().set_edgecolor('w')
    leg.get_frame().set_linewidth(0.0)
############################################################
############################################################

bh_radius = 6.0928e+05 # cm
bh_center_x = 1.5e6 # cm
bh_center_y = 1.5e6 # cm
bh_center_z = 1.5e6 # cm

file_path = './allData.h5'
with h5py.File(file_path, 'r') as file:
    datasets = list(file.keys())
    print("Datasets in the file:", datasets)
    N_00_Re = file['N00_Re(1|ccm)'][:]  # Adjust the dataset path as needed
    Fx00_Re = file['Fx00_Re(1|ccm)'][:]
    Fy00_Re = file['Fy00_Re(1|ccm)'][:]
    Fz00_Re = file['Fz00_Re(1|ccm)'][:]
    time = file['t(s)'][:]
    dx = np.array(file['dx(cm)'])
    dy = np.array(file['dy(cm)'])
    dz = np.array(file['dz(cm)'])
    Nx = np.array(file['Nx'])
    Ny = np.array(file['Ny'])
    Nz = np.array(file['Nz'])

    # cell faces
    x = np.linspace(0, dx * Nx, Nx + 1)
    y = np.linspace(0, dy * Ny, Ny + 1)
    z = np.linspace(0, dz * Nz, Nz + 1)
    # cell faces mesh
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    # cell centers
    xc = np.linspace(dx / 2, dx * (Nx - 0.5), Nx)
    yc = np.linspace(dy / 2, dy * (Ny - 0.5), Ny)
    zc = np.linspace(dz / 2, dz * (Nz - 0.5), Nz)
    # cell centers mesh
    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij')

    os.makedirs('./frames_f_00', exist_ok=True)
    os.makedirs('./frames_n_00', exist_ok=True)

    for time_index in range(len(time)):

        # Create the figure and subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

        min_nee = np.min([np.min(N_00_Re), np.min(N_00_Re)])
        max_nee = np.max([np.max(N_00_Re), np.max(N_00_Re)])

        # Plot neutrino data for the y-z plane
        c1 = ax1.pcolormesh(Xc[:, :, 10], Yc[:, :, 10], N_00_Re[time_index, :, :, 10], shading='auto', cmap='viridis', vmin=min_nee, vmax=max_nee)

        # Plot the mesh grid
        # ax1.plot(X[:, :, 10], Y[:, :, 10], color='gray', linestyle='-', linewidth=0.5)
        # ax1.plot(X[:, :, 10].T, Y[:, :, 10].T, color='gray', linestyle='-', linewidth=0.5)

        # Draw a circle representing the black hole in the y-z plane
        circle1 = plt.Circle((bh_center_x, bh_center_y), bh_radius, color='black')
        ax1.add_patch(circle1)

        # Plot settings for the y-z plane
        ax1.set_ylabel(r'$y \, (\mathrm{cm})$')

        leg1 = ax1.legend(framealpha=0.0, ncol=1, fontsize=10)
        apply_custom_settings(ax1, leg1, False)

        # Plot neutrino data for the x-z plane
        c2 = ax2.pcolormesh(Xc[:, 10, :], Zc[:, 10, :], N_00_Re[time_index, :, 10, :], shading='auto', cmap='viridis', vmin=min_nee, vmax=max_nee)

        # Plot the mesh grid for the x-z plane
        # ax2.plot(X[:, 10, :], Z[:, 10, :], color='gray', linestyle='-', linewidth=0.5)
        # ax2.plot(X[:, 10, :].T, Z[:, 10, :].T, color='gray', linestyle='-', linewidth=0.5)

        # Draw a circle representing the black hole in the x-z plane
        circle2 = plt.Circle((bh_center_x, bh_center_z), bh_radius, color='black')
        ax2.add_patch(circle2)

        # Plot settings for the x-z plane
        ax2.set_xlabel(r'$x \, (\mathrm{cm})$')
        ax2.set_ylabel(r'$z \, (\mathrm{cm})$')

        # Share a single color bar for both plots
        fig.subplots_adjust(right=0.85, hspace=0)  # Set hspace to 0 to remove the white space
        cbar_ax = fig.add_axes([0.82, 0.113, 0.04, 0.76]) # [left, bottom, width, height]
        cbar = fig.colorbar(c2, cax=cbar_ax, label=r'$n_e \, (\mathrm{cm}^{-3}$)')
        
        # Apply minor ticks to the color bar
        cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Ensure the plots use a box-like scale (equal aspect ratio)
        ax1.set_aspect('equal', 'box')
        ax2.set_aspect('equal', 'box')

        leg2 = ax2.legend(framealpha=0.0, ncol=1, fontsize=10)
        apply_custom_settings(ax2, leg2, False)

        # Save the figure as a PDF
        plt.savefig(f'./frames_n_00/n_00_{time_index}.png', bbox_inches='tight')

        plt.close(fig)

        # Create the figure and subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

        F_00_mag = np.sqrt(Fx00_Re**2 + Fy00_Re**2 + Fz00_Re**2) 

        F_00 = np.stack((Fx00_Re, Fy00_Re, Fz00_Re), axis=-1)

        Fx_00_unit = np.where(F_00_mag != 0, F_00[:,:,:,:,0] / F_00_mag, 0)
        Fy_00_unit = np.where(F_00_mag != 0, F_00[:,:,:,:,1] / F_00_mag, 0)
        Fz_00_unit = np.where(F_00_mag != 0, F_00[:,:,:,:,2] / F_00_mag, 0)

        F_00_for_plot = min(dx,dy,dz) * np.stack((Fx_00_unit, Fy_00_unit, Fz_00_unit), axis=-1)

        min_nee = np.min([np.min(F_00_mag), np.min(F_00_mag)])
        max_nee = np.max([np.max(F_00_mag), np.max(F_00_mag)])

        # Plot neutrino data for the y-z plane
        c1 = ax1.pcolormesh(Xc[:, :, 10], Yc[:, :, 10], Fy00_Re[time_index, :, :, 10], shading='auto', cmap='viridis', vmin=min_nee, vmax=max_nee)

        # Plot the arrows of the vector F_00_xy starting in each cell center
        ax1.quiver(Xc[:, :, 10], Yc[:, :, 10], F_00_for_plot[time_index,:,:,10,0], F_00_for_plot[time_index,:,:,10,1], color='white', scale=1, scale_units='xy', angles='xy')

        # Plot the mesh grid
        # ax1.plot(X[:, :, 10], Y[:, :, 10], color='gray', linestyle='-', linewidth=0.5)
        # ax1.plot(X[:, :, 10].T, Y[:, :, 10].T, color='gray', linestyle='-', linewidth=0.5)

        # Draw a circle representing the black hole in the y-z plane
        circle1 = plt.Circle((bh_center_x, bh_center_y), bh_radius, color='black')
        ax1.add_patch(circle1)

        # Plot settings for the y-z plane
        ax1.set_ylabel(r'$y \, (\mathrm{cm})$')

        leg1 = ax1.legend(framealpha=0.0, ncol=1, fontsize=10)
        apply_custom_settings(ax1, leg1, False)

        # Plot neutrino data for the x-z plane
        c2 = ax2.pcolormesh(Xc[:, 10, :], Zc[:, 10, :], Fy00_Re[time_index, :, 10, :], shading='auto', cmap='viridis', vmin=min_nee, vmax=max_nee)

        # Plot the arrows of the vector F_00_xy starting in each cell center
        ax2.quiver(Xc[:, 10, :], Zc[:, 10, :], F_00_for_plot[time_index,:,:,10,1], F_00_for_plot[time_index,:,:,10,2], color='white', scale=1, scale_units='xy', angles='xy')

        # Plot the mesh grid for the x-z plane
        # ax2.plot(X[:, 10, :], Z[:, 10, :], color='gray', linestyle='-', linewidth=0.5)
        # ax2.plot(X[:, 10, :].T, Z[:, 10, :].T, color='gray', linestyle='-', linewidth=0.5)

        # Draw a circle representing the black hole in the x-z plane
        circle2 = plt.Circle((bh_center_x, bh_center_z), bh_radius, color='black')
        ax2.add_patch(circle2)

        # Plot settings for the x-z plane
        ax2.set_xlabel(r'$x \, (\mathrm{cm})$')
        ax2.set_ylabel(r'$z \, (\mathrm{cm})$')

        # Share a single color bar for both plots
        fig.subplots_adjust(right=0.85, hspace=0)  # Set hspace to 0 to remove the white space
        cbar_ax = fig.add_axes([0.82, 0.113, 0.04, 0.76]) # [left, bottom, width, height]
        cbar = fig.colorbar(c2, cax=cbar_ax, label=r'$|\vec{f}_e| \, (\mathrm{cm}^{-3}$)')
        
        # Apply minor ticks to the color bar
        cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Ensure the plots use a box-like scale (equal aspect ratio)
        ax1.set_aspect('equal', 'box')
        ax2.set_aspect('equal', 'box')

        leg2 = ax2.legend(framealpha=0.0, ncol=1, fontsize=10)
        apply_custom_settings(ax2, leg2, False)

        # Save the figure as a PDF
        plt.savefig(f'./frames_f_00/f_00_{time_index}.png', bbox_inches='tight')

        plt.close(fig)


# Get the list of image files
image_files = sorted(glob.glob('./frames_f_00/*.png'))
# Sort the image files by the number after 'f_00_' and before '.png'
image_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Read the first image to get the dimensions
frame = cv2.imread(image_files[0])
height, width, layers = frame.shape

# Define the codec and create VideoWriter object
video = cv2.VideoWriter('./frames_f_00/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

for image_file in image_files:
    video.write(cv2.imread(image_file))

# Release the video writer object
video.release()


# Get the list of image files for n
image_files_n = sorted(glob.glob('./frames_n_00/*.png'))
# Sort the image files by the number after 'n_00_' and before '.png'
image_files_n = sorted(image_files_n, key=lambda x: int(x.split('_')[-1].split('.')[0]))

# Read the first image to get the dimensions
frame_n = cv2.imread(image_files_n[0])
height_n, width_n, layers_n = frame_n.shape

# Define the codec and create VideoWriter object for n
video_n = cv2.VideoWriter('./frames_n_00/output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (width_n, height_n))

for image_file_n in image_files_n:
    video_n.write(cv2.imread(image_file_n))

# Release the video writer object for n
video_n.release()
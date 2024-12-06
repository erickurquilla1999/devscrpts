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
    data_dict = {}
    for dataset in datasets:
        data_dict[dataset] = np.array(file[dataset])
   
    data_labels_2flavors = ['Fx00_Re(1|ccm)', 'Fx00_Rebar(1|ccm)', 'Fx01_Im(1|ccm)', 'Fx01_Imbar(1|ccm)', 'Fx01_Re(1|ccm)', 'Fx01_Rebar(1|ccm)', 'Fx11_Re(1|ccm)', 'Fx11_Rebar(1|ccm)', 'Fy00_Re(1|ccm)', 'Fy00_Rebar(1|ccm)', 'Fy01_Im(1|ccm)', 'Fy01_Imbar(1|ccm)', 'Fy01_Re(1|ccm)', 'Fy01_Rebar(1|ccm)', 'Fy11_Re(1|ccm)', 'Fy11_Rebar(1|ccm)', 'Fz00_Re(1|ccm)', 'Fz00_Rebar(1|ccm)', 'Fz01_Im(1|ccm)', 'Fz01_Imbar(1|ccm)', 'Fz01_Re(1|ccm)', 'Fz01_Rebar(1|ccm)', 'Fz11_Re(1|ccm)', 'Fz11_Rebar(1|ccm)', 'N00_Re(1|ccm)', 'N00_Rebar(1|ccm)', 'N01_Im(1|ccm)', 'N01_Imbar(1|ccm)', 'N01_Re(1|ccm)', 'N01_Rebar(1|ccm)', 'N11_Re(1|ccm)', 'N11_Rebar(1|ccm)', 'Nx', 'Ny', 'Nz', 'dx(cm)', 'dy(cm)', 'dz(cm)', 'it', 't(s)']
    data_labels_3flavors = ['Fx00_Re(1|ccm)', 'Fx00_Rebar(1|ccm)', 'Fx01_Im(1|ccm)', 'Fx01_Imbar(1|ccm)', 'Fx01_Re(1|ccm)', 'Fx01_Rebar(1|ccm)', 'Fx02_Im(1|ccm)', 'Fx02_Imbar(1|ccm)', 'Fx02_Re(1|ccm)', 'Fx02_Rebar(1|ccm)', 'Fx11_Re(1|ccm)', 'Fx11_Rebar(1|ccm)', 'Fx12_Im(1|ccm)', 'Fx12_Imbar(1|ccm)', 'Fx12_Re(1|ccm)', 'Fx12_Rebar(1|ccm)', 'Fx22_Re(1|ccm)', 'Fx22_Rebar(1|ccm)', 'Fy00_Re(1|ccm)', 'Fy00_Rebar(1|ccm)', 'Fy01_Im(1|ccm)', 'Fy01_Imbar(1|ccm)', 'Fy01_Re(1|ccm)', 'Fy01_Rebar(1|ccm)', 'Fy02_Im(1|ccm)', 'Fy02_Imbar(1|ccm)', 'Fy02_Re(1|ccm)', 'Fy02_Rebar(1|ccm)', 'Fy11_Re(1|ccm)', 'Fy11_Rebar(1|ccm)', 'Fy12_Im(1|ccm)', 'Fy12_Imbar(1|ccm)', 'Fy12_Re(1|ccm)', 'Fy12_Rebar(1|ccm)', 'Fy22_Re(1|ccm)', 'Fy22_Rebar(1|ccm)', 'Fz00_Re(1|ccm)', 'Fz00_Rebar(1|ccm)', 'Fz01_Im(1|ccm)', 'Fz01_Imbar(1|ccm)', 'Fz01_Re(1|ccm)', 'Fz01_Rebar(1|ccm)', 'Fz02_Im(1|ccm)', 'Fz02_Imbar(1|ccm)', 'Fz02_Re(1|ccm)', 'Fz02_Rebar(1|ccm)', 'Fz11_Re(1|ccm)', 'Fz11_Rebar(1|ccm)', 'Fz12_Im(1|ccm)', 'Fz12_Imbar(1|ccm)', 'Fz12_Re(1|ccm)', 'Fz12_Rebar(1|ccm)', 'Fz22_Re(1|ccm)', 'Fz22_Rebar(1|ccm)', 'N00_Re(1|ccm)', 'N00_Rebar(1|ccm)', 'N01_Im(1|ccm)', 'N01_Imbar(1|ccm)', 'N01_Re(1|ccm)', 'N01_Rebar(1|ccm)', 'N02_Im(1|ccm)', 'N02_Imbar(1|ccm)', 'N02_Re(1|ccm)', 'N02_Rebar(1|ccm)', 'N11_Re(1|ccm)', 'N11_Rebar(1|ccm)', 'N12_Im(1|ccm)', 'N12_Imbar(1|ccm)', 'N12_Re(1|ccm)', 'N12_Rebar(1|ccm)', 'N22_Re(1|ccm)', 'N22_Rebar(1|ccm)', 'Nx', 'Ny', 'Nz', 'dx(cm)', 'dy(cm)', 'dz(cm)', 'it', 't(s)']
    
    if len(datasets) == len(data_labels_2flavors):
        data_labels = data_labels_2flavors
        n_flavors = 2
    elif len(datasets) == len(data_labels_3flavors):
        data_labels = data_labels_3flavors
        n_flavors = 3
    else:
        raise ValueError('The number of datasets in the file does not match any of the expected values.')

    dx = data_dict['dx(cm)']
    dy = data_dict['dy(cm)']
    dz = data_dict['dz(cm)']
    Nx = data_dict['Nx']
    Ny = data_dict['Ny']
    Nz = data_dict['Nz']

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

    # Function to plot the neutrino densities
    def plotting_densities(x_up, y_up, z_up, x_down, y_down, z_down, min_color_bar, max_color_bar, color_bar_label, file_name, frame_index):
            
        # Create the figure and subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

        # Plot neutrino data for the y-z plane
        if max_color_bar == min_color_bar:
            max_color_bar += 1
            min_color_bar -= 1

        c1 = ax1.pcolormesh(x_up, y_up, z_up, shading='auto', cmap='viridis', vmin=min_color_bar, vmax=max_color_bar)
        c2 = ax2.pcolormesh(x_down, y_down, z_down, shading='auto', cmap='viridis', vmin=min_color_bar, vmax=max_color_bar)

        # Draw a circle representing the black hole in the y-z plane
        circle1 = plt.Circle((bh_center_x, bh_center_y), bh_radius, color='black')
        circle2 = plt.Circle((bh_center_x, bh_center_z), bh_radius, color='black')
        ax1.add_patch(circle1)
        ax2.add_patch(circle2)

        # Plot settings for the y-z plane
        ax2.set_xlabel(r'$x \, (\mathrm{cm})$')
        ax1.set_ylabel(r'$y \, (\mathrm{cm})$')
        ax2.set_ylabel(r'$z \, (\mathrm{cm})$')

        leg1 = ax1.legend(framealpha=0.0, ncol=1, fontsize=10)
        leg2 = ax2.legend(framealpha=0.0, ncol=1, fontsize=10)
        apply_custom_settings(ax1, leg1, False)
        apply_custom_settings(ax2, leg2, False)

        # Share a single color bar for both plots
        fig.subplots_adjust(right=0.85, hspace=0)  # Set hspace to 0 to remove the white space
        cbar_ax = fig.add_axes([0.82, 0.113, 0.04, 0.76]) # [left, bottom, width, height]
        cbar = fig.colorbar(c2, cax=cbar_ax, label=r'$'+color_bar_label+' \, (\mathrm{cm}^{-3}$)')
        
        # Apply minor ticks to the color bar
        cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Ensure the plots use a box-like scale (equal aspect ratio)
        ax1.set_aspect('equal', 'box')
        ax2.set_aspect('equal', 'box')

        # Save the figure as a PDF
        plt.savefig("./"+file_name+"/"+file_name+"_"+str(frame_index)+".png", bbox_inches='tight')

        # Close the figure
        plt.close(fig)

    # Function to plot the neutrino fluxes
    def plotting_fluxes(x_up, y_up, z_up, vector_field_up, x_down, y_down, z_down, vector_field_down, min_color_bar, max_color_bar, color_bar_label, file_name, frame_index):
 
        # Create the figure and subplots with shared x-axis
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1]})

        # Plot neutrino data for the y-z plane
        if max_color_bar == min_color_bar:
            max_color_bar += 1
            min_color_bar -= 1

        c1 = ax1.pcolormesh(x_up, y_up, z_up, shading='auto', cmap='viridis', vmin=min_color_bar, vmax=max_color_bar)
        c2 = ax2.pcolormesh(x_down, y_down, z_down, shading='auto', cmap='viridis', vmin=min_color_bar, vmax=max_color_bar)

        # Draw a circle representing the black hole in the y-z plane
        circle1 = plt.Circle((bh_center_x, bh_center_y), bh_radius, color='black')
        circle2 = plt.Circle((bh_center_x, bh_center_z), bh_radius, color='black')
        ax1.add_patch(circle1)
        ax2.add_patch(circle2)

        # vector field for the y-z plane
        ax1.quiver(x_up, y_up, vector_field_up[0], vector_field_up[1], color='white', scale=1, scale_units='xy', angles='xy')
        ax2.quiver(x_down, y_down, vector_field_down[0], vector_field_down[1], color='white', scale=1, scale_units='xy', angles='xy')

        # Plot settings for the y-z plane
        ax2.set_xlabel(r'$x \, (\mathrm{cm})$')
        ax1.set_ylabel(r'$y \, (\mathrm{cm})$')
        ax2.set_ylabel(r'$z \, (\mathrm{cm})$')

        leg1 = ax1.legend(framealpha=0.0, ncol=1, fontsize=10)
        leg2 = ax2.legend(framealpha=0.0, ncol=1, fontsize=10)
        apply_custom_settings(ax1, leg1, False)
        apply_custom_settings(ax2, leg2, False)

        # Share a single color bar for both plots
        fig.subplots_adjust(right=0.85, hspace=0)  # Set hspace to 0 to remove the white space
        cbar_ax = fig.add_axes([0.82, 0.113, 0.04, 0.76]) # [left, bottom, width, height]
        cbar = fig.colorbar(c2, cax=cbar_ax, label=r'$'+color_bar_label+' \, (\mathrm{cm}^{-3}$)')
        
        # Apply minor ticks to the color bar
        cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Ensure the plots use a box-like scale (equal aspect ratio)
        ax1.set_aspect('equal', 'box')
        ax2.set_aspect('equal', 'box')

        # Save the figure as a PDF
        plt.savefig("./"+file_name+"/"+file_name+"_"+str(frame_index)+".png", bbox_inches='tight')

        # Close the figure
        plt.close(fig)

    ####################################################################################
    # Plotting densities
    ####################################################################################

    folders_2flavors_densities = ['n_ee', 'n_eu', 'n_uu', 'nbar_ee', 'nbar_eu', 'nbar_uu']
    folders_3flavors_densities = ['n_et', 'n_ut', 'n_tt', 'nbar_et', 'nbar_ut', 'nbar_tt']

    for folder in folders_2flavors_densities:
        os.makedirs('./'+folder, exist_ok=True)
    if n_flavors == 3:
        for folder in folders_3flavors_densities:
            os.makedirs('./'+folder, exist_ok=True)

    def get_magnitude_densities(data_dict, keys):
        if len(keys) == 1:
            return data_dict[keys[0]]
        return np.sqrt(sum(data_dict[key]**2 for key in keys))

    def plot_all_data_densities(data_dict, labels, folder_prefix, n_flavors):
        for label in labels:
            mag = get_magnitude_densities(data_dict, label['keys'])
            mag_min = np.min(mag)
            mag_max = np.max(mag)
            for time_index in range(len(data_dict['t(s)'])):
                z_slice = 10
                y_slice = 10
                mag_z_slice = mag[time_index, :, :, z_slice]
                mag_y_slice = mag[time_index, :, y_slice, :]
                plotting_densities(Xc[:, :, z_slice], Yc[:, :, z_slice], mag_z_slice,
                              Xc[:, y_slice, :], Zc[:, y_slice, :], mag_y_slice, 
                              mag_min, mag_max, label['label'], folder_prefix + label['folder'], time_index)

    labels_2flavors = [
        {'keys': ['N00_Re(1|ccm)'], 'label': 'n_{ee}', 'folder': 'n_ee'},
        {'keys': ['N01_Re(1|ccm)', 'N01_Im(1|ccm)'], 'label': 'n_{eu}', 'folder': 'n_eu'},
        {'keys': ['N11_Re(1|ccm)'], 'label': 'n_{uu}', 'folder': 'n_uu'},
        {'keys': ['N00_Rebar(1|ccm)'], 'label': r'\bar{n}_{ee}', 'folder': 'nbar_ee'},
        {'keys': ['N01_Rebar(1|ccm)', 'N01_Imbar(1|ccm)'], 'label': r'\bar{n}_{eu}', 'folder': 'nbar_eu'},
        {'keys': ['N11_Rebar(1|ccm)'], 'label': r'\bar{n}_{uu}', 'folder': 'nbar_uu'}
    ]

    labels_3flavors = [
        {'keys': ['N02_Re(1|ccm)', 'N02_Im(1|ccm)'], 'label': 'n_{et}', 'folder': 'n_et'},
        {'keys': ['N12_Re(1|ccm)', 'N12_Im(1|ccm)'], 'label': 'n_{ut}', 'folder': 'n_ut'},
        {'keys': ['N22_Re(1|ccm)'], 'label': 'n_{tt}', 'folder': 'n_tt'},
        {'keys': ['N02_Rebar(1|ccm)', 'N02_Imbar(1|ccm)'], 'label': r'\bar{n}_{et}', 'folder': 'nbar_et'},
        {'keys': ['N12_Rebar(1|ccm)', 'N12_Imbar(1|ccm)'], 'label': r'\bar{n}_{ut}', 'folder': 'nbar_ut'},
        {'keys': ['N22_Rebar(1|ccm)'], 'label': r'\bar{n}_{tt}', 'folder': 'nbar_tt'}
    ]

    plot_all_data_densities(data_dict, labels_2flavors, '', n_flavors)
    if n_flavors == 3:
        plot_all_data_densities(data_dict, labels_3flavors, '', n_flavors)

    ####################################################################################
    # Plotting fluxes
    ####################################################################################

    folders_2flavors_fluxes = ['f_ee', 'f_uu', 'fbar_ee', 'fbar_uu']
    folders_3flavors_fluxes = ['f_tt', 'fbar_tt']
    
    for folder in folders_2flavors_fluxes:
        os.makedirs('./'+folder, exist_ok=True)
    if n_flavors == 3:
        for folder in folders_3flavors_fluxes:
            os.makedirs('./'+folder, exist_ok=True)

    def get_flux_magnitud_and_unit_vector(data_dict, keys):
        fx = data_dict[keys[0]]
        fy = data_dict[keys[1]]
        fz = data_dict[keys[2]]
        mag = np.sqrt(fx**2+fy**2+fz**2)
        fx = np.where(mag != 0, fx / mag, 0)
        fy = np.where(mag != 0, fy / mag, 0)
        fz = np.where(mag != 0, fz / mag, 0)
        return mag, np.stack((fx, fy, fz), axis=-1)

    def plot_all_fluxes(data_dict, labels, folder_prefix):
        for label in labels:
            mag, flux_unit = get_flux_magnitud_and_unit_vector(data_dict, label['keys'])
            flux = min(dx,dy,dz) * flux_unit
            mag_min = np.min(mag)
            mag_max = np.max(mag)
            for time_index in range(len(data_dict['t(s)'])):
                z_slice = 10
                y_slice = 10
                mag_z_slice = mag[time_index, :, :, z_slice]
                mag_y_slice = mag[time_index, :, y_slice, :]
                flux_z_slice = [flux[time_index, :, :, z_slice, 0], flux[time_index, :, :, z_slice,1]]
                flux_y_slice = [flux[time_index, :, y_slice, :, 0], flux[time_index, :, y_slice, :, 2]]
                plotting_fluxes(Xc[:, :, z_slice], Yc[:, :, z_slice], mag_z_slice, flux_z_slice,
                                Xc[:, y_slice, :], Zc[:, y_slice, :], mag_y_slice, flux_y_slice,
                                mag_min, mag_max, label['label'], folder_prefix + label['folder'], time_index)

    labels_2flavors_flux = [
        {'keys': ['Fx00_Re(1|ccm)','Fy00_Re(1|ccm)','Fz00_Re(1|ccm)'], 'label': '|\\vec{f}_{ee}|', 'folder': 'f_ee'},
        {'keys': ['Fx00_Rebar(1|ccm)','Fy00_Rebar(1|ccm)','Fz00_Rebar(1|ccm)'], 'label': r'|\\vec{f}_{\bar{ee}}|', 'folder': 'fbar_ee'},
        {'keys': ['Fx11_Re(1|ccm)', 'Fy11_Re(1|ccm)', 'Fz11_Re(1|ccm)'], 'label': '|\\vec{f}_{uu}|', 'folder': 'f_uu'},
        {'keys': ['Fx11_Rebar(1|ccm)','Fy11_Rebar(1|ccm)','Fz11_Rebar(1|ccm)'], 'label': r'|\\vec{f}_{\bar{uu}}|', 'folder': 'fbar_uu'}
        
    ]

    labels_3flavors_flux = [
        {'keys': ['Fx22_Re(1|ccm)', 'Fy22_Re(1|ccm)', 'Fz22_Re(1|ccm)'], 'label': '|\\vec{f}_{tt}|', 'folder': 'f_tt'},
        {'keys': ['Fx22_Rebar(1|ccm)', 'Fy22_Rebar(1|ccm)', 'Fz22_Rebar(1|ccm)'], 'label': r'|\\vec{f}_{\bar{tt}}|', 'folder': 'fbar_tt'},
    ]

    plot_all_fluxes(data_dict, labels_2flavors_flux, '')
    if n_flavors == 3:
        plot_all_fluxes(data_dict, labels_3flavors_flux, '')

####################################################################################
# Movies
####################################################################################

# Function to create a movie from a directory of images
def create_movie_from_images(directory, output_filename, frame_rate=5):

    # Get the list of image files
    image_files = sorted(glob.glob(os.path.join(directory, '*.png')))
    # Sort the image files by the number after the last underscore and before '.png'
    image_files = sorted(image_files, key=lambda x: int(x.split('_')[-1].split('.')[0]))

    if not image_files:
        raise ValueError(f"No images found in directory: {directory}")

    # Read the first image to get the dimensions
    frame = cv2.imread(image_files[0])
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    video = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for image_file in image_files:
        video.write(cv2.imread(image_file))

    # Release the video writer object
    video.release()

# Create movies from the images in the directories
for folder in folders_2flavors_densities:
    create_movie_from_images('./'+folder, './'+folder+'/output.mp4')
if n_flavors == 3:
    for folder in folders_3flavors_densities:
        create_movie_from_images('./'+folder, './'+folder+'/output.mp4')

for folder in folders_2flavors_fluxes:
    create_movie_from_images('./'+folder, './'+folder+'/output.mp4')
if n_flavors == 3:
    for folder in folders_3flavors_fluxes:
        create_movie_from_images('./'+folder, './'+folder+'/output.mp4')
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator
import os
import glob
from IPython.display import clear_output, display
import time
















# Font settings
mpl.rcParams['font.size'] = 22
mpl.rcParams['font.family'] = 'serif'
mpl.rc('text', usetex=False)

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























bh_radius =    5.43e+05 # cm
bh_center_x = 48.0e+5 # cm
bh_center_y = 48.0e+5 # cm
bh_center_z = 16.0e+5 # cm

pscratch_dir = '/pscratch/sd/u/uo1999'

# List directories in pscratch_dir that contain 'att_convergence_' and get their absolute paths
att_convergence_dirs = [os.path.join(pscratch_dir, d) for d in os.listdir(pscratch_dir) if 'att_convergence_' in d and os.path.isdir(os.path.join(pscratch_dir, d))]

att_data_dirs =[]
for att_dir in att_convergence_dirs:
    att_dat_dir = [os.path.join(att_dir, d) for d in os.listdir(att_dir) if 'att_' in d and os.path.isdir(os.path.join(att_dir, d))]
    att_data_dirs.append(att_dat_dir)
















def create_plot(x, y, z, min_color_bar, max_color_bar, output_file, xlabel, ylabel, title, cbarlabel, cmap='viridis', apply_settings=None):

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the data
    c = ax.pcolormesh(x, y, z, shading='auto', cmap=cmap, vmin=min_color_bar, vmax=max_color_bar)
    contour = ax.contour(x, y, z, colors='black', linewidths=0.5)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%1.1e')
    
    # Set plot labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    # Add color bar
    cbar = fig.colorbar(c, ax=ax, label=cbarlabel)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

    # Apply custom settings if provided
    if apply_settings:
        leg = ax.legend(framealpha=0.0, ncol=1, fontsize=10)
        apply_settings(ax,leg)
    
    # Ensure equal aspect ratio
    ax.set_aspect('equal', 'box')
    
    # Save the plot
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # Close the figure
    plt.close(fig)









for i in range(len(att_data_dirs)):
    for j in range(len(att_data_dirs[i])):
        
        file_path = att_data_dirs[i][j] + '/allData.h5'

        if os.path.isfile(file_path):

            print(file_path)



            



            with h5py.File(file_path, 'r') as file:
                datasets = list(file.keys())
                data_dict = {}
                for dataset in datasets:
                    data_dict[dataset] = np.array(file[dataset])

            data_labels_2flavors = ['Fx00_Re(1|ccm)', 'Fx00_Rebar(1|ccm)', 'Fx01_Im(1|ccm)', 'Fx01_Imbar(1|ccm)', 'Fx01_Re(1|ccm)', 'Fx01_Rebar(1|ccm)', 'Fx11_Re(1|ccm)', 'Fx11_Rebar(1|ccm)', 'Fy00_Re(1|ccm)', 'Fy00_Rebar(1|ccm)', 'Fy01_Im(1|ccm)', 'Fy01_Imbar(1|ccm)', 'Fy01_Re(1|ccm)', 'Fy01_Rebar(1|ccm)', 'Fy11_Re(1|ccm)', 'Fy11_Rebar(1|ccm)', 'Fz00_Re(1|ccm)', 'Fz00_Rebar(1|ccm)', 'Fz01_Im(1|ccm)', 'Fz01_Imbar(1|ccm)', 'Fz01_Re(1|ccm)', 'Fz01_Rebar(1|ccm)', 'Fz11_Re(1|ccm)', 'Fz11_Rebar(1|ccm)', 'N00_Re(1|ccm)', 'N00_Rebar(1|ccm)', 'N01_Im(1|ccm)', 'N01_Imbar(1|ccm)', 'N01_Re(1|ccm)', 'N01_Rebar(1|ccm)', 'N11_Re(1|ccm)', 'N11_Rebar(1|ccm)', 'Nx', 'Ny', 'Nz', 'dx(cm)', 'dy(cm)', 'dz(cm)', 'it', 't(s)']
            data_labels_3flavors = ['Fx00_Re(1|ccm)', 'Fx00_Rebar(1|ccm)', 'Fx01_Im(1|ccm)', 'Fx01_Imbar(1|ccm)', 'Fx01_Re(1|ccm)', 'Fx01_Rebar(1|ccm)', 'Fx02_Im(1|ccm)', 'Fx02_Imbar(1|ccm)', 'Fx02_Re(1|ccm)', 'Fx02_Rebar(1|ccm)', 'Fx11_Re(1|ccm)', 'Fx11_Rebar(1|ccm)', 'Fx12_Im(1|ccm)', 'Fx12_Imbar(1|ccm)', 'Fx12_Re(1|ccm)', 'Fx12_Rebar(1|ccm)', 'Fx22_Re(1|ccm)', 'Fx22_Rebar(1|ccm)', 'Fy00_Re(1|ccm)', 'Fy00_Rebar(1|ccm)', 'Fy01_Im(1|ccm)', 'Fy01_Imbar(1|ccm)', 'Fy01_Re(1|ccm)', 'Fy01_Rebar(1|ccm)', 'Fy02_Im(1|ccm)', 'Fy02_Imbar(1|ccm)', 'Fy02_Re(1|ccm)', 'Fy02_Rebar(1|ccm)', 'Fy11_Re(1|ccm)', 'Fy11_Rebar(1|ccm)', 'Fy12_Im(1|ccm)', 'Fy12_Imbar(1|ccm)', 'Fy12_Re(1|ccm)', 'Fy12_Rebar(1|ccm)', 'Fy22_Re(1|ccm)', 'Fy22_Rebar(1|ccm)', 'Fz00_Re(1|ccm)', 'Fz00_Rebar(1|ccm)', 'Fz01_Im(1|ccm)', 'Fz01_Imbar(1|ccm)', 'Fz01_Re(1|ccm)', 'Fz01_Rebar(1|ccm)', 'Fz02_Im(1|ccm)', 'Fz02_Imbar(1|ccm)', 'Fz02_Re(1|ccm)', 'Fz02_Rebar(1|ccm)', 'Fz11_Re(1|ccm)', 'Fz11_Rebar(1|ccm)', 'Fz12_Im(1|ccm)', 'Fz12_Imbar(1|ccm)', 'Fz12_Re(1|ccm)', 'Fz12_Rebar(1|ccm)', 'Fz22_Re(1|ccm)', 'Fz22_Rebar(1|ccm)', 'N00_Re(1|ccm)', 'N00_Rebar(1|ccm)', 'N01_Im(1|ccm)', 'N01_Imbar(1|ccm)', 'N01_Re(1|ccm)', 'N01_Rebar(1|ccm)', 'N02_Im(1|ccm)', 'N02_Imbar(1|ccm)', 'N02_Re(1|ccm)', 'N02_Rebar(1|ccm)', 'N11_Re(1|ccm)', 'N11_Rebar(1|ccm)', 'N12_Im(1|ccm)', 'N12_Imbar(1|ccm)', 'N12_Re(1|ccm)', 'N12_Rebar(1|ccm)', 'N22_Re(1|ccm)', 'N22_Rebar(1|ccm)', 'Nx', 'Ny', 'Nz', 'dx(cm)', 'dy(cm)', 'dz(cm)', 'it', 't(s)']

            n_flavors = 3
            n_plt_files = len(data_dict['Fx00_Re(1|ccm)'])
            print('Number of files:', n_plt_files)









            # cell size
            dx = data_dict['dx(cm)'] # cm
            dy = data_dict['dy(cm)'] # cm
            dz = data_dict['dz(cm)'] # cm
            print(f'dx = {dx} cm, dy = {dy} cm, dz = {dz} cm')

            # number of cells
            Nx = data_dict['Nx'] 
            Ny = data_dict['Ny']
            Nz = data_dict['Nz']
            print(f'Nx = {Nx}, Ny = {Ny}, Nz = {Nz}')

            # cell faces
            x = np.linspace(0, dx * Nx, Nx + 1) # cm
            y = np.linspace(0, dy * Ny, Ny + 1) # cm
            z = np.linspace(0, dz * Nz, Nz + 1) # cm

            # cell faces mesh
            X, Y, Z = np.meshgrid(x, y, z, indexing='ij') # cm

            # cell centers
            xc = np.linspace(dx / 2, dx * (Nx - 0.5), Nx) # cm
            yc = np.linspace(dy / 2, dy * (Ny - 0.5), Ny) # cm
            zc = np.linspace(dz / 2, dz * (Nz - 0.5), Nz) # cm

            # cell centers mesh
            Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing='ij') # cm









            distance_from_bh = np.sqrt( (Xc - bh_center_x)**2 + (Yc - bh_center_y)**2 + (Zc - bh_center_z)**2 )
            mask_bh = (distance_from_bh < bh_radius)
            mask_x = (Xc < dx) | (Xc > (Nx - 1) * dx)
            mask_y = (Yc < dy) | (Yc > (Ny - 1) * dy)
            mask_z = (Zc < dz) | (Zc > (Nz - 1) * dz)
            mask = mask_bh | mask_x | mask_y | mask_z







            # 1 index run between 0 and 1. 0 is for neutrinos and 1 is for antineutrinos
            # 2 index runs over the plt files
            # 3 index runs over the cells in the x direction
            # 4 index runs over the cells in the y direction
            # 5 index runs over the cells in the z direction
            # 6 and 7 indices runs over the neutrino number density matrix components
            n_all = np.zeros((2, n_plt_files, Nx, Ny, Nz, n_flavors, n_flavors), dtype=complex) # 1/cm^3

            components = ['00', '01', '02', '11', '12', '22']
            neutrino_types = ['', 'bar']

            for k, nt in enumerate(neutrino_types):
                for comp in components:
                    re_key = f'N{comp}_Re{nt}(1|ccm)'
                    im_key = f'N{comp}_Im{nt}(1|ccm)'
                    if im_key in data_dict:
                        n_all[k, :, :, :, :, int(comp[0]), int(comp[1])] = data_dict[re_key] + 1j * data_dict[im_key]
                    else:
                        n_all[k, :, :, :, :, int(comp[0]), int(comp[1])] = data_dict[re_key]

            # absolute value of the number density matrix
            n_all_abs = np.abs(n_all)

            # real part of the number density matrix
            n_all_re = np.real(n_all)













            # Time
            time_ = data_dict['t(s)'] # s















            n_ee_xy_slices_dir = att_data_dirs[i][j] + '/n_ee_xy_slices'
            os.makedirs(n_ee_xy_slices_dir, exist_ok=True)

            zup = n_all_abs[0,:,:,:,:,0,0]
            zup[:,mask] = -1.0
            max_color_bar = np.max(zup)
            zup[:,mask] = max_color_bar + 1.0
            min_color_bar = np.min(zup)
            
            for f in range(n_all_abs.shape[1]):

                # for z_slice_indx in range(Nz): # cell number
                for z_slice_indx in [16]: # cell number

                    z_slice_value = Zc[0,0,z_slice_indx] # cm

                    # Data x and y for upper panel
                    xup = Xc[:, :, z_slice_indx] / 1e5 # cm
                    yup = Yc[:, :, z_slice_indx] / 1e5 # cm
                    maskup = mask[:,:,z_slice_indx]

                    # Data for the color map
                    zup = n_all_abs[0,f,:,:,z_slice_indx,0,0] # 1/cm^3
                    zup[maskup] = -1.0
                    this_max_color_bar = np.max(zup)
                    zup[maskup] = this_max_color_bar + 1.0
                    this_min_color_bar = np.min(zup)

                    # zup = np.where(zup<1e-5, 1e-5, zup)
                    # zup = np.log10(zup)
                    zup[maskup] = np.nan

                    output_file = os.path.join(n_ee_xy_slices_dir, f'n_ee_{z_slice_indx}_f_{f}_t_{time_[f]:.2e}.png')                   
                    plot_title = f'$z={z_slice_value/1e5:.1f}\,\mathrm{{km}},\,t={time_[f]/1e-3:.2f}\,\mathrm{{ms}}$\nmax = {this_max_color_bar:.2e}\nmin = {this_min_color_bar:.2e}'
                    create_plot(
                        x=xup, 
                        y=yup, 
                        z=zup, 
                        min_color_bar=min_color_bar, 
                        max_color_bar=max_color_bar, 
                        output_file=output_file, 
                        xlabel=r'$x \, (\mathrm{km})$', 
                        ylabel=r'$y \, (\mathrm{km})$',
                        title=plot_title,
                        cbarlabel=f'$n_{{ee}} \, (\mathrm{{cm}}^{{-3}})$',
                        cmap='viridis', 
                        apply_settings=apply_custom_settings,                         
                    )

            n_ee_xz_slices_dir = att_data_dirs[i][j] + '/n_ee_xz_slices'
            os.makedirs(n_ee_xz_slices_dir, exist_ok=True)

            for f in range(n_all_abs.shape[1]):

                # for y_slice_indx in range(Ny): # cell number
                for y_slice_indx in [48]: # cell number

                    y_slice_value = Yc[0,y_slice_indx,0] # cm

                    # Data x and y for lower panel
                    xlow = Xc[:, y_slice_indx, :] / 1e5 # cm
                    ylow = Zc[:, y_slice_indx, :] / 1e5 # cm
                    masklow = mask[:, y_slice_indx, :]

                    # Data for the color map
                    zlow = n_all_abs[0,f,:,y_slice_indx,:,0,0] # 1/cm^3

                    zlow[masklow] = -1.0
                    this_max_color_bar = np.max(zlow)
                    zlow[masklow] = this_max_color_bar + 1.0
                    this_min_color_bar = np.min(zlow)

                    # zlow = np.where(zlow<1e-5, 1e-5, zlow)
                    # zlow = np.log10(zlow)
                    zlow[masklow] = np.nan

                    ####################################################################################
                    output_file = os.path.join(n_ee_xz_slices_dir, f'n_ee_{y_slice_indx}_f_{f}_t_{time_[f]:.2e}.png')
                    plot_title = f'$y={y_slice_value/1e5:.1f}\,\mathrm{{km}},\,t={time_[f]/1e-3:.2f}\,\mathrm{{ms}}$\nmax = {this_max_color_bar:.2e}\nmin = {this_min_color_bar:.2e}'
                    create_plot(
                        x=xlow, 
                        y=ylow, 
                        z=zlow, 
                        min_color_bar=min_color_bar, 
                        max_color_bar=max_color_bar, 
                        output_file=output_file, 
                        xlabel=r'$x \, (\mathrm{km})$', 
                        ylabel=r'$z \, (\mathrm{km})$',
                        title=plot_title,
                        cbarlabel=f'$n_{{ee}} \, (\mathrm{{cm}}^{{-3}})$',
                        cmap='viridis', 
                        apply_settings=apply_custom_settings,                         
                    )




            z_data = n_all_abs[0,:,:,:,:,0,1] # 0,1 is e,mu
            z_data[:,mask] = -1.0
            max_color_bar = np.max(z_data)
            z_data[:,mask] = max_color_bar + 1.0
            min_color_bar = np.min(z_data)
            z_data[:,mask] = np.nan

            n_eu_xy_slices_dir = att_data_dirs[i][j] + '/n_eu_xy_slices'
            os.makedirs(n_eu_xy_slices_dir, exist_ok=True)
            
            for f in range(n_all_abs.shape[1]):

                # for z_slice_indx in range(Nz): # cell number
                for z_slice_indx in [16]: # cell number

                    z_slice_value = Zc[0,0,z_slice_indx] # cm

                    # Data x and y for upper panel
                    xup = Xc[:, :, z_slice_indx] / 1e5 # cm
                    yup = Yc[:, :, z_slice_indx] / 1e5 # cm
                    maskup = mask[:,:,z_slice_indx]

                    # Data for the color map
                    zup = z_data[f,:,:,z_slice_indx] # 1/cm^3
                    zup[maskup] = -1.0
                    this_max_color_bar = np.max(zup)
                    zup[maskup] = this_max_color_bar + 1.0
                    this_min_color_bar = np.min(zup)

                    # zup = np.where(zup<1e-5, 1e-5, zup)
                    # zup = np.log10(zup)
                    zup[maskup] = np.nan
                    
                    ####################################################################################
                    output_file = os.path.join(n_eu_xy_slices_dir, f'n_eu_{z_slice_indx}_f_{f}_t_{time_[f]:.2e}.png')                   
                    plot_title = f'$z={z_slice_value/1e5:.1f}\,\mathrm{{km}},\,t={time_[f]/1e-3:.2f}\,\mathrm{{ms}}$\nmax = {this_max_color_bar:.2e}\nmin = {this_min_color_bar:.2e}'
                    create_plot(
                        x=xup, 
                        y=yup, 
                        z=zup, 
                        min_color_bar=min_color_bar, 
                        max_color_bar=max_color_bar, 
                        output_file=output_file, 
                        xlabel=r'$x \, (\mathrm{km})$', 
                        ylabel=r'$y \, (\mathrm{km})$',
                        title=plot_title,
                        cbarlabel=f'$n_{{e\mu}} \, (\mathrm{{cm}}^{{-3}})$',
                        cmap='viridis', 
                        apply_settings=apply_custom_settings,                         
                    )

            n_eu_xz_slices_dir = att_data_dirs[i][j] + '/n_eu_xz_slices'
            os.makedirs(n_eu_xz_slices_dir, exist_ok=True)

            for f in range(n_all_abs.shape[1]):

                # for y_slice_indx in range(Ny): # cell number
                for y_slice_indx in [48]: # cell number

                    y_slice_value = Yc[0,y_slice_indx,0] # cm

                    # Data x and y for lower panel
                    xlow = Xc[:, y_slice_indx, :] / 1e5 # cm
                    ylow = Zc[:, y_slice_indx, :] / 1e5 # cm
                    masklow = mask[:, y_slice_indx, :]

                    # Data for the color map
                    zlow = z_data[f,:,y_slice_indx,:] # 1/cm^3

                    zlow[masklow] = -1.0
                    this_max_color_bar = np.max(zlow)
                    zlow[masklow] = this_max_color_bar + 1.0
                    this_min_color_bar = np.min(zlow)

                    # zlow = np.where(zlow<1e-5, 1e-5, zlow)
                    # zlow = np.log10(zlow)
                    zlow[masklow] = np.nan

                    # max_color_bar = +4e32
                    # min_color_bar = -4e32

                    ####################################################################################
                    output_file = os.path.join(n_eu_xz_slices_dir, f'n_ee_{y_slice_indx}_f_{f}_t_{time_[f]:.2e}.png')
                    plot_title = f'$y={y_slice_value/1e5:.1f}\,\mathrm{{km}},\,t={time_[f]/1e-3:.2f}\,\mathrm{{ms}}$\nmax = {this_max_color_bar:.2e}\nmin = {this_min_color_bar:.2e}'
                    create_plot(
                        x=xlow, 
                        y=ylow, 
                        z=zlow, 
                        min_color_bar=min_color_bar, 
                        max_color_bar=max_color_bar, 
                        output_file=output_file, 
                        xlabel=r'$x \, (\mathrm{km})$', 
                        ylabel=r'$z \, (\mathrm{km})$',
                        title=plot_title,
                        cbarlabel=f'$n_{{eu}} \, (\mathrm{{cm}}^{{-3}})$',
                        cmap='viridis', 
                        apply_settings=apply_custom_settings,                         
                    )


















            z_data = n_all_abs[0,:,:,:,:,1,1] # 1,1 is mu,mu
            z_data[:,mask] = -1.0
            max_color_bar = np.max(z_data)
            z_data[:,mask] = max_color_bar + 1.0
            min_color_bar = np.min(z_data)
            z_data[:,mask] = np.nan

            slices_dir = att_data_dirs[i][j] + '/n_uu_xy_slices'
            os.makedirs(slices_dir, exist_ok=True)
            
            for f in range(n_all_abs.shape[1]):

                # for z_slice_indx in range(Nz): # cell number
                for z_slice_indx in [16]: # cell number

                    z_slice_value = Zc[0,0,z_slice_indx] # cm

                    # Data x and y for upper panel
                    xup = Xc[:, :, z_slice_indx] / 1e5 # cm
                    yup = Yc[:, :, z_slice_indx] / 1e5 # cm
                    maskup = mask[:,:,z_slice_indx]

                    # Data for the color map
                    zup = z_data[f,:,:,z_slice_indx] # 1/cm^3
                    zup[maskup] = -1.0
                    this_max_color_bar = np.max(zup)
                    zup[maskup] = this_max_color_bar + 1.0
                    this_min_color_bar = np.min(zup)

                    # zup = np.where(zup<1e-5, 1e-5, zup)
                    # zup = np.log10(zup)
                    zup[maskup] = np.nan
                    
                    ####################################################################################
                    output_file = os.path.join(slices_dir, f's_{z_slice_indx}_f_{f}_t_{time_[f]:.2e}.png')                   
                    plot_title = f'$z={z_slice_value/1e5:.1f}\,\mathrm{{km}},\,t={time_[f]/1e-3:.2f}\,\mathrm{{ms}}$\nmax = {this_max_color_bar:.2e}\nmin = {this_min_color_bar:.2e}'
                    create_plot(
                        x=xup, 
                        y=yup, 
                        z=zup, 
                        min_color_bar=min_color_bar, 
                        max_color_bar=max_color_bar, 
                        output_file=output_file, 
                        xlabel=r'$x \, (\mathrm{km})$', 
                        ylabel=r'$y \, (\mathrm{km})$',
                        title=plot_title,
                        cbarlabel=f'$n_{{\mu\mu}} \, (\mathrm{{cm}}^{{-3}})$',
                        cmap='viridis', 
                        apply_settings=apply_custom_settings,                         
                    )

            slices_dir = att_data_dirs[i][j] + '/n_uu_xz_slices'
            os.makedirs(slices_dir, exist_ok=True)

            for f in range(n_all_abs.shape[1]):

                # for y_slice_indx in range(Ny): # cell number
                for y_slice_indx in [48]: # cell number

                    y_slice_value = Yc[0,y_slice_indx,0] # cm

                    # Data x and y for lower panel
                    xlow = Xc[:, y_slice_indx, :] / 1e5 # cm
                    ylow = Zc[:, y_slice_indx, :] / 1e5 # cm
                    masklow = mask[:, y_slice_indx, :]

                    # Data for the color map
                    zlow = z_data[f,:,y_slice_indx,:] # 1/cm^3

                    zlow[masklow] = -1.0
                    this_max_color_bar = np.max(zlow)
                    zlow[masklow] = this_max_color_bar + 1.0
                    this_min_color_bar = np.min(zlow)

                    # zlow = np.where(zlow<1e-5, 1e-5, zlow)
                    # zlow = np.log10(zlow)
                    zlow[masklow] = np.nan

                    # max_color_bar = +4e32
                    # min_color_bar = -4e32

                    ####################################################################################
                    output_file = os.path.join(slices_dir, f's_{y_slice_indx}_f_{f}_t_{time_[f]:.2e}.png')
                    plot_title = f'$y={y_slice_value/1e5:.1f}\,\mathrm{{km}},\,t={time_[f]/1e-3:.2f}\,\mathrm{{ms}}$\nmax = {this_max_color_bar:.2e}\nmin = {this_min_color_bar:.2e}'
                    create_plot(
                        x=xlow, 
                        y=ylow, 
                        z=zlow, 
                        min_color_bar=min_color_bar, 
                        max_color_bar=max_color_bar, 
                        output_file=output_file, 
                        xlabel=r'$x \, (\mathrm{km})$', 
                        ylabel=r'$z \, (\mathrm{km})$',
                        title=plot_title,
                        cbarlabel=f'$n_{{\mu\mu}} \, (\mathrm{{cm}}^{{-3}})$',
                        cmap='viridis', 
                        apply_settings=apply_custom_settings,                         
                    )
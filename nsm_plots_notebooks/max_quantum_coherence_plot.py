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





final_time = []
final_max_rhoeu = []
final_max_rhoeu_coord = []
attenuation = []
time_step = []






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
            # mask_bh = (distance_from_bh < bh_radius)
            # mask_x = (Xc < dx) | (Xc > (Nx - 1) * dx)
            # mask_y = (Yc < dy) | (Yc > (Ny - 1) * dy)
            # mask_z = (Zc < dz) | (Zc > (Nz - 1) * dz)


            ncellout = 3
            mask_bh = (distance_from_bh < bh_radius + ncellout * np.max([dx,dy,dz]))
            mask_x = (Xc < ncellout * dx) | (Xc > (Nx - ncellout) * dx)
            mask_y = (Yc < ncellout * dy) | (Yc > (Ny - ncellout) * dy)
            mask_z = (Zc < ncellout * dz) | (Zc > (Nz - ncellout) * dz)

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
            time_ = data_dict['t(s)'] / 1e-3 # ms















            maxrhoeu_dir = pscratch_dir + '/plots_max_rho_eu_vs_time'
            os.makedirs(maxrhoeu_dir, exist_ok=True)

            max_rho_eu = np.zeros(n_all_abs.shape[1])
            max_val_coord = np.zeros((n_all_abs.shape[1],3)) # km

            for f in range(n_all_abs.shape[1]):
                
                zup = n_all_abs[0,f,:,:,:,0,1] # 0,1 at the end mean eu
                zup[mask] = -1 * np.inf
                max_rho_eu[f] = np.max(zup)
                max_indices = np.unravel_index(np.argmax(zup, axis=None), zup.shape)
                max_val_coord[f,0] = Xc[max_indices] / 1e5 # km
                max_val_coord[f,1] = Yc[max_indices] / 1e5 # km
                max_val_coord[f,2] = Zc[max_indices] / 1e5 # km


            # Plot max_rho_eu vs time_ using ax and fig
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.plot(time_, max_rho_eu, marker='o', linestyle='-', color='b')
            ax.set_xlabel('$t$ ($ms$)')
            ax.set_ylabel('Max($|n_{e\mu}|$)')
            ax.set_yscale('log')

            # Add point labels for max_val_coord
            for f in range(len(time_)):
                ax.annotate(f'x = {max_val_coord[f,0]:.1f} km\ny = {max_val_coord[f,1]:.1f} km\nz = {max_val_coord[f,2]:.1f} km', 
                            (time_[f], max_rho_eu[f]), 
                            textcoords="offset points", 
                            xytext=(0,10), 
                            ha='center', fontsize=5)

            leg = ax.legend(framealpha=0.0, ncol=1, fontsize=10)
            apply_custom_settings(ax,leg)
            
            name_file_1 = att_data_dirs[i][j].split('/')[-1]
            attenuation_string = name_file_1.split('_')[-1] 
            name_file_2 = att_data_dirs[i][j].split('/')[-2].split('_')[-1]               
            timestep_string = name_file_2.split('m')[0] 

            fig.savefig(maxrhoeu_dir + f'/{name_file_2}_{name_file_1}_max_rho_eu_vs_time.png', dpi=300, bbox_inches='tight')
            plt.close(fig)

            final_time.append(time_[-1]) # ms
            final_max_rhoeu.append(max_rho_eu[-1])
            final_max_rhoeu_coord.append(max_val_coord[-1]) # km
            attenuation.append(float(attenuation_string))
            time_step.append(float(timestep_string))



print(f'final_time = {final_time}')
print(f'final_max_rhoeu = {final_max_rhoeu}')
print(f'final_max_rhoeu_coord = {final_max_rhoeu_coord}') # km
print(f'attenuation = {attenuation}')
print(f'time_step = {time_step}') #



# Plot max_rho_eu vs time_ using ax and fig
fig, ax = plt.subplots(figsize=(12, 8))
ax.scatter(attenuation, final_max_rhoeu, marker='o', color='b')
ax.set_xlabel('Attenuation')
ax.set_ylabel('Max($|n_{e\mu}|$)')
ax.set_yscale('log')
ax.set_xscale('log')

# Add point labels for max_val_coord
for f in range(len(final_time)):
    ax.annotate(f'$t_f$ = {final_time[f]:.2f} ms\n$c\Delta t$ = {time_step[f]:.1f} m\nx = {final_max_rhoeu_coord[f][0]} km\ny = {final_max_rhoeu_coord[f][1]} km\nz = {final_max_rhoeu_coord[f][2]} km',
                (attenuation[f], final_max_rhoeu[f]), 
                textcoords="offset points", 
                xytext=(0,10), 
                ha='center', fontsize=10)

# leg = ax.legend(framealpha=0.0, ncol=1, fontsize=10)
# apply_custom_settings(ax,leg)

fig.savefig(maxrhoeu_dir + f'/max_rho_eu_vs_attenuation.png', dpi=300, bbox_inches='tight')
plt.close(fig)
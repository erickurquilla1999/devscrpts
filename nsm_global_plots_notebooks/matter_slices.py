import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator
import os
import glob

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

file_path = '/pscratch/sd/u/uo1999/matter_plots/rho_Ye_T.hdf5'

with h5py.File(file_path, 'r') as file:
    datasets = list(file.keys())
    print(f'datasets = {datasets}')
    data_dict = {}
    for dataset in datasets:
        data_dict[dataset] = np.array(file[dataset])

# number of cells
Nx = data_dict['ncellsx'] 
Ny = data_dict['ncellsy']
Nz = data_dict['ncellsz']
print(f'Nx = {Nx}, Ny = {Ny}, Nz = {Nz}')

# cell size
dx = ( data_dict['xmax_cm'] - data_dict['xmin_cm'] ) / data_dict['ncellsx'] # cm
dy = ( data_dict['ymax_cm'] - data_dict['ymin_cm'] ) / data_dict['ncellsy'] # cm
dz = ( data_dict['zmax_cm'] - data_dict['zmin_cm'] ) / data_dict['ncellsz'] # cm
print(f'dx = {dx} cm, dy = {dy} cm, dz = {dz} cm')

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

bh_radius =    5.43e+05 # cm
bh_center_x = 48.0e+5 # cm
bh_center_y = 48.0e+5 # cm
bh_center_z = 16.0e+5 # cm

distance_from_bh = np.sqrt( (Xc - bh_center_x)**2 + (Yc - bh_center_y)**2 + (Zc - bh_center_z)**2 )
mask_bh = (distance_from_bh < bh_radius)
# mask_x = (Xc < dx) | (Xc > (Nx - 1) * dx)
# mask_y = (Yc < dy) | (Yc > (Ny - 1) * dy)
# mask_z = (Zc < dz) | (Zc > (Nz - 1) * dz)
# mask = mask_bh | mask_x | mask_y | mask_z
mask = mask_bh

data_dict['T_Mev'][mask] = np.nan
data_dict['Ye'][mask] = np.nan
data_dict['rho_g|ccm'][mask] = np.nan

def plot_slice(x, y, z, xlabel, ylabel, title, cbar_label, output_file):

    fig, ax = plt.subplots(figsize=(12, 8))
    c = ax.pcolormesh(x, y, z, shading='auto', cmap='viridis', vmin=np.nanmin(z), vmax=np.nanmax(z))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = fig.colorbar(c, ax=ax, label=cbar_label)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())
    leg = ax.legend(framealpha=0.0, ncol=1, fontsize=10)
    apply_custom_settings(ax, leg, False)
    ax.set_aspect('equal', 'box')
    fig.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close(fig)

os.makedirs("/pscratch/sd/u/uo1999/matter_plots/temperature_slices_xy", exist_ok=True)
os.makedirs("/pscratch/sd/u/uo1999/matter_plots/rho_slices_xy", exist_ok=True)
os.makedirs("/pscratch/sd/u/uo1999/matter_plots/ye_slices_xy", exist_ok=True)

for z_slice_indx in range(Nz):
    z_slice_value = Zc[0,0,z_slice_indx] # cm
    x = Xc[:, :, z_slice_indx] / 1e5 # cm
    y = Yc[:, :, z_slice_indx] / 1e5 # cm
    temp_slice_xy = data_dict['T_Mev']    [:,:,z_slice_indx]
    ye_slice_xy   = data_dict['Ye']       [:,:,z_slice_indx]
    rho_slice_xy  = data_dict['rho_g|ccm'][:,:,z_slice_indx]
    plot_slice(
        x=x,
        y=y,
        z=temp_slice_xy,
        xlabel=r'$x \, (\mathrm{km})$',
        ylabel=r'$y \, (\mathrm{km})$',
        title=f'$z={z_slice_value/1e5:.1f}\,\mathrm{{km}}$\nmax = {np.max(temp_slice_xy):.2e}\nmin = {np.min(temp_slice_xy):.2e}',
        cbar_label=fr'$ T\,(\mathrm{{MeV}})$',
        output_file='/pscratch/sd/u/uo1999/matter_plots/temperature_slices_xy/z_'+str(z_slice_indx)+'.png'
    )
    plot_slice(
        x=x,
        y=y,
        z=rho_slice_xy,
        xlabel=r'$x \, (\mathrm{km})$',
        ylabel=r'$y \, (\mathrm{km})$',
        title=f'$z={z_slice_value/1e5:.1f}\,\mathrm{{km}}$\nmax = {np.max(rho_slice_xy):.2e}\nmin = {np.min(rho_slice_xy):.2e}',
        cbar_label=fr'$\rho\,(\mathrm{{g/ccm}})$',
        output_file='/pscratch/sd/u/uo1999/matter_plots/rho_slices_xy/z_'+str(z_slice_indx)+'.png'
    )
    plot_slice(
        x=x,
        y=y,
        z=ye_slice_xy,
        xlabel=r'$x \, (\mathrm{km})$',
        ylabel=r'$y \, (\mathrm{km})$',
        title=f'$z={z_slice_value/1e5:.1f}\,\mathrm{{km}}$\nmax = {np.max(ye_slice_xy):.2e}\nmin = {np.min(ye_slice_xy):.2e}',
        cbar_label=fr'$Ye$',
        output_file='/pscratch/sd/u/uo1999/matter_plots/ye_slices_xy/z_'+str(z_slice_indx)+'.png'
    )

os.makedirs("/pscratch/sd/u/uo1999/matter_plots/temperature_slices_xz", exist_ok=True)
os.makedirs("/pscratch/sd/u/uo1999/matter_plots/rho_slices_xz", exist_ok=True)
os.makedirs("/pscratch/sd/u/uo1999/matter_plots/ye_slices_xz", exist_ok=True)

for y_slice_indx in range(Ny):
    y_slice_value = Yc[0,y_slice_indx,0] # cm
    x = Xc[:, y_slice_indx, :] / 1e5 # cm
    y = Zc[:, y_slice_indx, :] / 1e5 # cm
    temp_slice_xz   = data_dict['T_Mev']    [:,y_slice_indx,:]
    ye_slice_up_xz  = data_dict['Ye']       [:,y_slice_indx,:]
    rho_slice_up_xz = data_dict['rho_g|ccm'][:,y_slice_indx,:]
    plot_slice(
        x=x,
        y=y,
        z=temp_slice_xz,
        xlabel=r'$x \, (\mathrm{km})$',
        ylabel=r'$z \, (\mathrm{km})$',
        title=f'$y={y_slice_value/1e5:.1f}\,\mathrm{{km}}$\nmax = {np.max(temp_slice_xz):.2e}\nmin = {np.min(temp_slice_xz):.2e}',
        cbar_label=fr'$ T\,(\mathrm{{MeV}})$',
        output_file='/pscratch/sd/u/uo1999/matter_plots/temperature_slices_xz/y_'+str(y_slice_indx)+'.png'
    )
    plot_slice(
        x=x,
        y=y,
        z=rho_slice_up_xz,
        xlabel=r'$x \, (\mathrm{km})$',
        ylabel=r'$z \, (\mathrm{km})$',
        title=f'$y={y_slice_value/1e5:.1f}\,\mathrm{{km}}$\nmax = {np.max(rho_slice_up_xz):.2e}\nmin = {np.min(rho_slice_up_xz):.2e}',
        cbar_label=fr'$\rho\,(\mathrm{{g/ccm}})$',
        output_file='/pscratch/sd/u/uo1999/matter_plots/rho_slices_xz/y_'+str(y_slice_indx)+'.png'
    )
    plot_slice(
        x=x,
        y=y,
        z=ye_slice_up_xz,
        xlabel=r'$x \, (\mathrm{km})$',
        ylabel=r'$z \, (\mathrm{km})$',
        title=f'$y={y_slice_value/1e5:.1f}\,\mathrm{{km}}$\nmax = {np.max(ye_slice_up_xz):.2e}\nmin = {np.min(ye_slice_up_xz):.2e}',
        cbar_label=fr'$Ye$',
        output_file='/pscratch/sd/u/uo1999/matter_plots/ye_slices_xz/y_'+str(y_slice_indx)+'.png'
    )
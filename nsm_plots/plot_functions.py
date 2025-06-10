import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.ticker import AutoLocator, AutoMinorLocator, LogLocator

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
def apply_custom_settings(ax, log_scale_y=False):

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

def plot_color_map(x, y, z, min_cb, max_cb, x_label, y_label, title, cbar_label, colormap, filename, doshow=True, dosave=True):

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot pcolormesh
    c = ax.pcolormesh(x, y, z, shading='auto', cmap=colormap, vmin=min_cb, vmax=max_cb)

    # Add contour lines
    contour = ax.contour(x, y, z, colors='black', linewidths=1.5, levels=3)
    # contour = ax.contour(x, y, z, colors='black', linewidths=1.5, levels=[0.5,2.0,3.5])
    ax.clabel(contour, inline=True, fontsize=15, fmt='%1.1f')
    # ax.clabel(contour, inline=True, fontsize=15, fmt='%1.1e')

    # Plot settings
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    # ax.set_title(title+'\nmin: {:.2e}\nmax: {:.2e}'.format(np.nanmin(z), np.nanmax(z)))
    ax.set_title(title)

    # Add color bar
    cbar = fig.colorbar(c, ax=ax, label=cbar_label)
    cbar.ax.yaxis.set_minor_locator(AutoMinorLocator())

    apply_custom_settings(ax, False)

    # Ensure equal aspect ratio
    ax.set_aspect('equal', 'box')

    # Save figure
    if dosave:
        fig.savefig(filename, format='png', bbox_inches='tight')

    # Display figure
    if doshow:
        plt.show()
    # display(fig)
    
    # Close figure
    plt.close(fig)
import sys
import os
import numpy as np
import functions_angular_crossings as fac

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../phys_const')))
import phys_const as pc

def compute_sigma_GnPos_GnNeg(i, j, k, directory, cellvolume):

        # Dictionary keys
        # <KeysViewHDF5 ['N00_Re', 'N00_Rebar', 'N01_Im', 'N01_Imbar', 'N01_Re', 'N01_Rebar', 'N02_Im', 'N02_Imbar', 'N02_Re', 'N02_Rebar', 'N11_Re', 'N11_Rebar', 'N12_Im', 'N12_Imbar', 'N12_Re', 'N12_Rebar', 'N22_Re', 'N22_Rebar', 'TrHN', 'Vphase', 'pos_x', 'pos_y', 'pos_z', 'pupt', 'pupx', 'pupy', 'pupz', 'time', 'x', 'y', 'z']>
        particles_dict_this_cell = fac.load_particle_data(i, j, k, directory)

        px = particles_dict_this_cell['pupx']/particles_dict_this_cell['pupt']
        py = particles_dict_this_cell['pupy']/particles_dict_this_cell['pupt']
        pz = particles_dict_this_cell['pupz']/particles_dict_this_cell['pupt']
        # print(f'pz.shape = {pz.shape}')

        momentum = np.stack((px, py, pz), axis=-1)
        tolerance = 1.0e-4
        unique_momentum = fac.get_unique_momentum(momentum, tolerance)

        # Define solid angle differential
        n_directions = unique_momentum.shape[0]
        domega = 4*np.pi / n_directions # solid angle per particle

        # theta = np.arccos(unique_momentum[:,2])
        # phi = np.arctan2(unique_momentum[:,1], unique_momentum[:,0])
        # phi = np.where(phi < 0, phi + 2 * np.pi, phi) # Adjusting the angle to be between 0 and 2pi.

        nee_all     = particles_dict_this_cell['N00_Re']    / cellvolume
        nuu_all     = particles_dict_this_cell['N11_Re']    / cellvolume
        ntt_all     = particles_dict_this_cell['N22_Re']    / cellvolume
        neebar_all  = particles_dict_this_cell['N00_Rebar'] / cellvolume
        nuubar_all  = particles_dict_this_cell['N11_Rebar'] / cellvolume
        nttbar_all  = particles_dict_this_cell['N22_Rebar']  / cellvolume

        fluxee_all    = nee_all   [:, np.newaxis] * momentum
        fluxuu_all    = nuu_all   [:, np.newaxis] * momentum
        fluxtt_all    = ntt_all   [:, np.newaxis] * momentum
        fluxeebar_all = neebar_all[:, np.newaxis] * momentum
        fluxuubar_all = nuubar_all[:, np.newaxis] * momentum
        fluxnttbar_all = nttbar_all[:, np.newaxis] * momentum

        ee_unique_fluxes, ee_unique_fluxes_mag       = fac.compute_unique_fluxes(momentum, fluxee_all, unique_momentum)
        uu_unique_fluxes, uu_unique_fluxes_mag       = fac.compute_unique_fluxes(momentum, fluxuu_all, unique_momentum)
        tt_unique_fluxes, tt_unique_fluxes_mag       = fac.compute_unique_fluxes(momentum, fluxtt_all, unique_momentum)
        eebar_unique_fluxes, eebar_unique_fluxes_mag = fac.compute_unique_fluxes(momentum, fluxeebar_all, unique_momentum)
        uubar_unique_fluxes, uubar_unique_fluxes_mag = fac.compute_unique_fluxes(momentum, fluxuubar_all, unique_momentum)
        ttbar_unique_fluxes, ttbar_unique_fluxes_mag = fac.compute_unique_fluxes(momentum, fluxnttbar_all, unique_momentum)

        eln_xln = (
        (ee_unique_fluxes_mag - eebar_unique_fluxes_mag ) -
        (uu_unique_fluxes_mag - uubar_unique_fluxes_mag ) -  
        (tt_unique_fluxes_mag - ttbar_unique_fluxes_mag ) 
        )

        G_ELN = np.sqrt(2) * pc.PhysConst.GF * ( pc.PhysConst.hbarc**3 / pc.PhysConst.hbar ) * eln_xln

        mask_pos_eln = G_ELN > 0
        mask_neg_eln = G_ELN < 0

        GnPos = +1.0 * ( domega / (4 * np.pi) ) * np.sum(G_ELN[mask_pos_eln])
        GnNeg = -1.0 * ( domega / (4 * np.pi) ) * np.sum(G_ELN[mask_neg_eln])
        sigma = np.sqrt(GnPos*GnNeg) / (2 * np.pi)

        print(f'sigma={sigma:.3e} 1/s, GnPos={GnPos:.3e} 1/s, GnNeg={GnNeg:.3e} 1/s, i={i}, j={j}, k={k}')
        return sigma, GnPos, GnNeg
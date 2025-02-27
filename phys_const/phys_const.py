class CGSUnitsConst:
    eV = 1.60218e-12  # erg

class PhysConst:
    c = 2.99792458e10  # cm/s
    c2 = c * c
    c4 = c2 * c2
    hbar = 1.05457266e-27  # erg s
    hbarc = hbar * c  # erg cm
    hc = hbarc*2*np.pi
    GF = (1.1663787e-5 / (1e9 * 1e9 * CGSUnitsConst.eV * CGSUnitsConst.eV))
    Mp = 1.6726219e-24  # g
    sin2thetaW = 0.23122
    kB = 1.380658e-16  # erg/K
    G = 6.67430e-8 # cm3 g−1 s-2
    Msun = 1.9891e33 # g

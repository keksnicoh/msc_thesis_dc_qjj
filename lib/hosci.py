# -*- coding: utf-8 -*-
""" JJ operators in hosci approximation """

import qoptical as qo
import numpy as np

def Ovoid(n):
    """ zero operator """
    return np.zeros((n, n), dtype=qo.QO.T_COMPLEX)


def On(n):
    """ dn """
    o = Ovoid(n)
    o += np.diag(np.arange(n))
    return o


def Oa(n):
    """ a """
    o, r = Ovoid(n), np.arange(n - 1, dtype=np.int32)
    o[r, r + 1] = np.sqrt(r + 1)
    return o


def Oad(n):
    """ a^\dagger """
    o, r = Ovoid(n), np.arange(n - 1, dtype=np.int32)
    o[r + 1, r] = np.sqrt(r + 1)
    return o


def Ophi(n):
    """ phi = a + a^\dagger """
    return Oa(n) + Oad(n)


def Odn(n):
    """ phi = a + a^\dagger """
    return -1.0j * (Oa(n) - Oad(n))


def Ophi2(n):
    """ phi^2 = (a + a^\dagger)^2 """
    o, r = Ovoid(n), np.arange(n - 2, dtype=np.int32)

    o[r + 2, r] = \
    o[r, r + 2] = np.sqrt(r + 1) * np.sqrt(r + 2)
    o += np.diag(np.arange(1, 2 * n, 2))

    return o


def Ov(dimH, EC, J0):
    """ Voltage operator """
    Omega = np.sqrt(J0 * EC)
    V0    = -EC * 1.0j * np.sqrt(Omega / (2 * EC));
    return V0 * (Oa(dimH) - Oad(dimH))


def Oh0(dimH, EC, J0):
    """ harmonic oscillator H0 """
    return np.sqrt(J0 * EC) * On(dimH)


def hosci_rs(dimH, EC, EJ, dn_coupling=False):
    """ reduced system H0 + coupling """
    coupling = 1.0j*(Oa(dimH) - Oad(dimH)) if dn_coupling else Ophi(dimH)
    return qo.ReducedSystem(Oh0(dimH, EC, EJ), dipole=coupling)

def sys_parametric_driving(dimH, EC, J0, drvL, I0, wpL):

    Omega = np.sqrt(J0 * EC)
    P0 = 1.0 / EC * np.sqrt(EC / (2 * Omega))

    shape   = len(drvL), len(wpL)
    parr    = [(I0, wp, *drv) for drv in drvL for wp in wpL]
    systems = np.array(parr, dtype=np.dtype([
        ('I0', qo.QO.T_FLOAT),
        ('wp', qo.QO.T_FLOAT),
        ('Ad', qo.QO.T_FLOAT),
        ('wd', qo.QO.T_FLOAT),
    ])).reshape(shape)

    probe_coeff = lambda t, p: p['I0'] * np.sin(p['wp'] * t) # probe modulation
    J_coeff     = lambda t, p: p['Ad'] * np.sin(p['wd'] * t) # J modulation

    OHul = [
        [Omega * Ophi2(dimH), J_coeff], # driving
        [P0 * Ophi(dimH),  probe_coeff] # probing
    ]

    return systems, OHul

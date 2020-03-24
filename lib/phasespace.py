# -*- coding: utf-8 -*-
"""
    JJ operators in phasedens representation
"""

import numpy as np
import qoptical as qo


def Ovoid(n):
    """
    zero operator

    Arguments:
    ----------

    :n: number of states

    Returns:
    --------

    ndarray of shape (n, n)

    """
    return np.zeros((n, n), dtype=qo.QO.T_COMPLEX)


def Odn2(n):
    """
    squared imbalance operator

    Arguments:
    ----------

    :n: number of states

    Returns:
    --------

    ndarray of shape (n, n)

    """
    o, odd, m2 = Ovoid(n), n % 2 != 0, int(n / 2)
    kr = np.array([i for i in range(-m2, m2 + 1) if odd or i != 0])
    o[np.arange(n), np.arange(n)] = kr**2
    return o


def Odn(n):
    """
    imbalance operator

    Arguments:
    ----------

    :n: number of states

    Returns:
    --------

    ndarray of shape (n, n)

    """
    o, odd, m2 = Ovoid(n), n % 2 != 0, int(n / 2)
    kr = np.array([i for i in range(-m2, m2 + 1) if odd or i != 0])
    o[np.arange(n), np.arange(n)] = kr
    return o


def Ophi(n):
    """
    phase operator

    Arguments:
    ----------

    :n: number of states

    Returns:
    --------

    ndarray of shape (n, n)

    """
    o, r = Ovoid(n), np.arange(n)
    for i in r:
        for j in r:
            if j - i == 0:
                o[i, i] = np.pi
            else:
                even = (j - i + 1) % 2 == 0
                o[i, j] = 1.0j * (-1 + 2 * even) / (j - i)
    return o


def Ocos(n):
    """
    cosine

    Arguments:
    ----------

    :n: number of states

    Returns:
    --------

    ndarray of shape (n, n)

    """
    o, r = Ovoid(n), np.arange(n-1)
    o[r, r + 1] = o[r + 1, r] = 0.5
    return o


def Osin(n):
    """
    sine

    Arguments:
    ----------

    :n: number of states

    Returns:
    --------

    ndarray of shape (n, n)

    """
    o, r = Ovoid(n), np.arange(n-1)
    o[r, r + 1] = -0.5j
    o[r + 1, r] = 0.5j
    return o


def Osin2(n):
    o, rd, rd2 = Ovoid(n), np.arange(n), np.arange(n - 2)
    o[rd, rd]       = 2
    o[rd2, rd2 + 2] = -1
    o[rd2 + 2, rd2] = -1
    return 0.25 * o


def sjj_rs(dimH, EC, J0, dn_coupling=False):
    Oh0 = 0.5 * EC * Odn2(dimH) - J0 * Ocos(dimH)
    coupling = Odn(dimH) if dn_coupling else Osin(dimH)
    return qo.ReducedSystem(Oh0, dipole=coupling)


def Osjj_V(dimH, EC):
    # operators
    return EC * Odn(dimH)
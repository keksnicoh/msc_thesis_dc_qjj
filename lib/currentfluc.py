# -*- coding: utf-8 -*-
""" :author: Nicolas 'keksnicoh' Heimann
"""
import numpy as np
import qoptical as qo
from .phasespace import *
from .util import concat

class Sin2Config():

    def __init__(
        self,
        dimH,
        EC,
        J0,
        T,
        y0,
        tg,
        avg_len,
        rec_skip=1,
        yt_coeff=False,
        cmd=None,
        drv_arr=None,
        **kwargs
    ):
        self.dimH       = dimH
        self.EC         = EC
        self.J0         = J0
        self.T          = float(T) if not isinstance(T, list) else [float(t) for t in T]
        self.y0         = y0
        self.tg         = (float(tg[0]), float(tg[1]), float(tg[2]))
        self.avg_len    = avg_len
        self.rec_skip   = int(rec_skip)
        self.cmd        = cmd
        self.yt_coeff   = yt_coeff
        self.drv_arr    = drv_arr

    def dump(self):
        unnumpy = lambda v: np.asscalar(v) if hasattr(v, 'dtype') else v
        return { k: unnumpy(v) for k, v in self.__dict__.items() }


def load_fs_avg(fnames, config_dict=False, axis=0, shape=None):
    """ load a dataset from a list of filenames.

        Arguments:
        ----------

        :fnames: list of filenames

        :config_dict: whether the config should be represented as a dict or
                      (if False) as SigmaConfig instance.

        :axis: axis to merge files

        Returns:
        --------

        (dict config, ndarray systems, ndarray avg)

        """
    config, systems, avg = None, None, None
    for fname in fnames:
        if config is None:
            cfg, systems, avg = qo.load_fs(fname, ['config', 'systems', 'avg'])
            config = [cfg]
            continue

        cfg, sys, sig = qo.load_fs(fname, ['config', 'systems', 'avg'])
        print(fname, sys.shape, sig.shape)
        systems = concat(systems, sys, fname, axis)
        avg = concat(avg, sig, fname, axis)
        config.append(cfg)

    if not config_dict:
        config = [SigmaConfig(**c) for c in config]

    if shape is None:
        return config, systems, avg

    return config,\
           systems.reshape(shape),\
           avg.reshape(shape)

def persist_fs_avg(fname, cfg, systems, avg):
    if isinstance(cfg, Sin2Config):
        cfg = cfg.dump()

    return qo.persist_fs(fname, config=cfg, systems=systems, avg=avg)


def sys_parametric_driving(dimH, J0, drvL):
    """ returns paratric driving parameters and list
        of driving Hamilton which evolve in the unitary
        part of the lindblad equation.

        Arguments:
        ----------

        :dimH: Hilberts Dimension

        :drvL: list of driving parameters, [(amplitude, freq), ...]

        :I0: probe current

        :wpL: list of probing frequencies

        Returns:
        --------

        (np.ndarray, list)

        """

    shape = (len(drvL), )
    dtype_parametric = np.dtype([('Ad', qo.QO.T_FLOAT), ('wd', qo.QO.T_FLOAT)])
    systems = np.array(drvL, dtype=dtype_parametric).reshape(shape)
    OHul = [[J0 * Ocos(dimH), lambda t, p: p['Ad'] * np.sin(p['wd'] * t)],]
    return systems, OHul

def opmesolve_avg_eq(rs, T, y0, Oexpect, tg, avg_len, OHul=None, params=None, **qoptical_args):
    """
    evolves an operator `Oexpect` in time and returns the simple
    arithmetic average of the last `avg_len` samples.

    Arguments:
    ----------

    :rs: Reduced Systems

    :T: temperatures

    :y0: dampign

    :Oexpect: observable

    :tg: time gatter

    :avg_len: the length of sample over which the avg is taken

    :OHul: unitary evolution

    :params: parameters for unitary evolution

    :qoptical_args: directly passed kwargs to `qo.opmesolve_cl_expect`

    Returns:
    --------

    average of expectation value

    """
    texpect = qo.opmesolve_cl_expect(
        tg=tg,
        reduced_system=rs,
        t_bath=T,
        y_0=y0,
        rho0=rs.thermal_state(T=T),
        Oexpect=Oexpect,
        OHul=OHul,
        params=params,
        **qoptical_args
    )

    avg_list = True
    if not isinstance(avg_len, list):
        avg_len = [avg_len]
        avg_list = False

    res = np.zeros((texpect.shape[1], len(avg_len)), dtype=qo.QO.T_COMPLEX)
    for i, alen in enumerate(avg_len):
        res[:, i] = np.sum(texpect[-alen:], axis=0) / alen

    if not avg_list:
        return res[:, 0]

    return res


# -*- coding: utf-8 -*-
""" probing josephson junction tools

    relevant source
    ---------------
    -

    -

    -

    :author: Nicolas 'keksnicoh' Heimann
"""

from functools import reduce
import numpy as np
import qoptical as qo
import os
from .phasespace import *
from .util import concat
from scipy.optimize import curve_fit

# --------- Drude ---------------------------------

two_fluid_drude = lambda w, J, y: J * (y + 1.0j * w * y) / (w**2 + y**2)
drude = lambda w, p: p[0] * (p[1]**2 + 1.0j * p[1] * w) / (w**2 + p[1]**2)

# --------- plot helpers -----------------

def plot_sigma2(plt, wpL, sigma_imag, fit, linecolor="royalblue", markercolor="black"):
    wpX = np.arange(0, wpL[-1] + (wpL[-1] - wpL[-2]), 0.0001)
    plt.plot(wpX, drude(wpX, fit).imag, label="fit drude $\\sigma_2$", color=linecolor)
    plt.plot(wpL, sigma_imag, '.', label="numerics", color=markercolor)
    plt.legend()

# --------- classical model (m=1) -----------------

def jeff_m1(wd, w0, A, y0):
    """
    1606.09276 Eq. 5
    """
    zf = w0**2 - wd**2
    nf = 2 * (w0**2 - wd**2)**2 + 2 * y0**2 * wd**2
    pf = A**2 * w0**2
    return 1 - pf * zf / nf


def jeff_m1_hh1(wd, w0, A, A1, phi, y0):
    """
    1706.04554 Eq. 15
    """
    zf = 2 * wd**2 + (np.abs(A1) * np.cos(phi) - 2) * w0**2
    nf = 4 * (wd**2 - w0**2)**2 - A1**2 * w0**4 + 4 * y0**2*wd**2
    pf = A**2 * w0**2
    return 1 + pf * zf / nf

# --------- misc ----------------------------------

class SigmaConfig():

    def __init__(
        self,
        dimH,
        EC,
        J0,
        T,
        y0,
        fft_t0,
        fft_sample,
        fft_ntp,
        dt,
        yt_coeff=False,
        cmd=None,
        drv_arr=None,
        **kwargs
    ):
        self.dimH       = dimH
        self.EC         = EC
        self.J0         = J0
        self.T          = T                    # XXX vectorlike
        self.y0         = y0                  # XXX vectorlike
        self.fft_t0     = fft_t0
        self.fft_sample = fft_sample
        self.fft_ntp    = fft_ntp
        self.dt         = dt
        self.cmd        = cmd
        self.yt_coeff   = yt_coeff
        self.drv_arr    = drv_arr

    def dump(self):
        unnumpy = lambda v: np.asscalar(v) if hasattr(v, 'dtype') else v
        return { k: unnumpy(v) for k, v in self.__dict__.items() }


def load_fs(fnames, config_dict=False, axis=0, shape=None):
    """ load a dataset from a list of filenames.

        Arguments:
        ----------

        :fnames: list of filenames

        :config_dict: whether the config should be represented as a dict or
                      (if False) as SigmaConfig instance.

        :axis: axis to merge files

        Returns:
        --------

        (dict config, ndarray systems, ndarray sigma)

        """
    config, systems, sigma = None, None, None
    for fname in fnames:
        if config is None:
            cfg, systems, sigma = qo.load_fs(fname, ['config', 'systems', 'sigma'])
            config = [cfg]
            continue

        cfg, sys, sig = qo.load_fs(fname, ['config', 'systems', 'sigma'])
        systems = concat(systems, sys, fname, axis)
        sigma   = concat(sigma, sig, fname, axis)
        config.append(cfg)

    if not config_dict:
        config = [SigmaConfig(**c) for c in config]

    if shape is None:
        return config, systems, sigma

    return config,\
           systems.reshape(shape),\
           sigma.reshape(shape)

def persist_fs(fname, cfg, systems, sigma):
    """ persist config, systems and sigma data

        Arguments:
        ----------

        :fname: fs filename

        :cfg: arr of configs

        :systems: ndarray

        :sigma: ndarray
    """
    if isinstance(cfg, SigmaConfig):
        cfg = cfg.dump()

    return qo.persist_fs(fname, config=cfg, systems=systems, sigma=sigma)

# --------- code ----------------------------------

def jeff_err_drude(wp, sig, fit):
    """
    computes the relative error of the fit against
    expected functional drude form

    Arguments:
    ----------

    :wp: probing frequencies of shape (*s, Nwp)

    :sig: sigma values of shape (*s, Nwp)

    :fit: fit parameter of shape (*s, 2)

    Returns:
    --------

    relative error arr of shape (*s)

    """

    # add component to line it up with wp and sig
    f = fit.reshape((*fit.shape, 1))

    # analytical result
    fsig_im = two_fluid_drude(wp, f[..., 0, :], f[..., 1, :]).imag

    # compute error
    dx2 = ((sig.imag - fsig_im) / fsig_im) ** 2
    err_rel2 = np.sum(dx2, axis=len(fsig_im.shape) - 1)
    return np.sqrt(err_rel2) / sig.shape[-1]


def jeff_mean_rel(sigma, sigma0):
    """ effective josephson coupling as the relative change
        of sigma compared to some gauge sigma0.

        Arguments:
        ----------

        :sigma:  (N_sys, N_wp) shaped sigma

        :sigma0:     (1, N_wp) shaped gauge sigma

        Reurns:
        -------

        (N_sys, 1) shaped float ndarray

        """
    ls = len(sigma.shape)
    ns = sigma.imag.shape[ls - 1]
    return np.sum(sigma.imag / sigma0.imag, axis=ls - 1) / ns


def jeff_fit_drude(wp, sigma, fit_real=False, fit_kwargs={}):
    """ effective josephson coupling by fitting the
        drude model

             o_2(wp) = J_eff * wp / (wp**2 + yeff**2)

        Arguments:
        ----------

        :wp: (N_sys, N_wp), (N_wp) shaped probing frequencies

        :sigma0: (N_sys, N_wp), (N_wp) shaped conductivity (sigma)

        :fit_real:

        :fit_kwargs: passed to `curve_fit(..., **fit_kwargs)`

        Returns:
        --------

        - (N_sys, 2), (N_Sys, 2, 2)) shaped fit parameters
          and errors.

        or

        - (2, ), (2, 2) shaped fit parameters
          and errors.

        """

    if not np.all(wp.shape == sigma.shape):
        err = 'shape missmatch wp.shape {} != sigma.shape {}'
        raise ValueError(err.format(wp.shape, sigma.shape))

    # normalize shapes
    original_shape = sigma.shape
    if len(original_shape) == 1:
        sigma = sigma.reshape((1, original_shape[0]))
        wp = wp.reshape((1, original_shape[0]))
    elif len(original_shape) > 2:
        acc = reduce(lambda x, y: x * y, sigma.shape[:-1], 1)
        reduced_shape = acc, sigma.shape[-1]
        sigma = sigma.reshape(reduced_shape)
        wp = wp.reshape(reduced_shape)

    if fit_real:
        f = lambda w, J, y: two_fluid_drude(w, J, y).real
        sigfx = sigma.real
    else:
        f = lambda w, J, y: two_fluid_drude(w, J, y).imag
        sigfx = sigma.imag

    n = sigma.shape[0]
    fitJy = np.zeros((n, 2), dtype=np.float32)
    fitJy_err = np.zeros((n, 2, 2), dtype=np.float32)

    def fit(fx, fy):
        try:
            return curve_fit(f, fx, fy, **fit_kwargs)
        except RuntimeError as e:
            return (0, 0), ((0, 0), (0, 0))

    for i in range(n):
        fx, fy = wp[i], sigfx[i]
        fitJy[i], fitJy_err[i] = fit(fx, fy)

    fitJy = fitJy.reshape((*original_shape[:-1], 2))
    fitJy_err = fitJy_err.reshape((*original_shape[:-1], 2, 2))
    return np.abs(fitJy), fitJy_err


def sys_probe(dimH, I0, wpL):
    """ returns probing parameters and probing Hamilton

        Arguments:
        ----------

        :dimH: Hilberts Dimension

        Returns:
        --------

        (np.ndarray, list)

        """

    shape = 1, len(wpL)
    parr = [(I0, wp) for wp in wpL]
    dtype_parametric = np.dtype([
        ('I0', qo.QO.T_FLOAT),
        ('wp', qo.QO.T_FLOAT),
    ])
    systems = np.array(parr, dtype=dtype_parametric).reshape(shape)

    OHul = [
        [Ophi(dimH), lambda t, p: p['I0'] * np.sin(p['wp'] * t)],
    ]

    return systems, OHul


DTYPE_PARAM_HO = np.dtype([
    ('A', np.float32),
    ('w', np.float32),
    ('phi', np.float32),
])

def sys_parametric_hh_driving(dimH, J0, drv_arr, I0, wpL):
    """

        Arguments:
        ----------

        :dimH: Hilberts Dimension

        :J0: bare Josephson Coupling

        :drv_arr: driving configuration, (M drivings x N harmonics)
            [
                [(a11, wp11, phi11), ..., (a1N, wp1N, phi1N)],
                ...,
                [(aM1, wpM1, phiM1), ..., (aMN, wpMN, phiMN)],
            ]

            example:
            --------

            [
                [(0.1, 2.4, 0), (0.04, 4.8, 0)],        # A_0=0.1, A_1=0.04, wp0=2.4, wp1=4.8, phi0=0, phi1=0
                [(0.1, 2.4, 0), (0.04, 4.8, 0.1)],      # ...
                [(0.1, 2.4, 0), (0.04, 4.8, 0.2)],
                [(0.1, 2.4, 0), (0.04, 4.8, 0.3)],
            ]

        :I0: probe current

        :wpL: list of probing frequencies

        Returns:
        --------

        (np.ndarray, list)

        """
    try:
        hh_p = np.array([[tuple(dd) for dd in d] for d in drv_arr], dtype=DTYPE_PARAM_HO)
    except ValueError:
        raise ValueError('drv_arr is not a valid array: {}'.format(drv_arr))

    # extend driving with probing
    p_arr = [(I0, wp, *d['A'], *d['w'], *d['phi']) for d in hh_p for wp in wpL]
    drype_parametric_hh = np.dtype([
        ('I0',                  qo.QO.T_FLOAT),
        ('wp',                  qo.QO.T_FLOAT),
        *(('A_{}'.format(i),    qo.QO.T_FLOAT) for i in range(hh_p.shape[1])),
        *(('wd_{}'.format(i),   qo.QO.T_FLOAT) for i in range(hh_p.shape[1])),
        *(('phi_{}'.format(i),  qo.QO.T_FLOAT) for i in range(hh_p.shape[1])),
    ])
    systems = np.array(p_arr, dtype=drype_parametric_hh)

    # parametric driving needs to be defined this way to keep the OpenCL
    # functions simple (also, f2cl is not powerfull enough otherwise).
    if hh_p.shape[1] == 1:
        df = lambda t, p: p['A_0'] * np.sin(p['wd_0'] * t + p['phi_0'])
    elif hh_p.shape[1] == 2:
        df = lambda t, p: p['A_0'] * np.sin(p['wd_0'] * t + p['phi_0']) + \
                          p['A_1'] * np.sin(p['wd_1'] * t + p['phi_1'])
    elif hh_p.shape[1] == 3:
        df = lambda t, p: p['A_0'] * np.sin(p['wd_0'] * t + p['phi_0']) + \
                          p['A_1'] * np.sin(p['wd_1'] * t + p['phi_1']) + \
                          p['A_2'] * np.sin(p['wd_2'] * t + p['phi_2'])
    elif hh_p.shape[1] == 4:
        df = lambda t, p: p['A_0'] * np.sin(p['wd_0'] * t + p['phi_0']) + \
                          p['A_1'] * np.sin(p['wd_1'] * t + p['phi_1']) + \
                          p['A_2'] * np.sin(p['wd_2'] * t + p['phi_2']) + \
                          p['A_3'] * np.sin(p['wd_3'] * t + p['phi_3'])
    elif hh_p.shape[1] == 5:
        df = lambda t, p: p['A_0'] * np.sin(p['wd_0'] * t + p['phi_0']) + \
                          p['A_1'] * np.sin(p['wd_1'] * t + p['phi_1']) + \
                          p['A_2'] * np.sin(p['wd_2'] * t + p['phi_2']) + \
                          p['A_3'] * np.sin(p['wd_3'] * t + p['phi_3']) + \
                          p['A_4'] * np.sin(p['wd_4'] * t + p['phi_4'])
    elif hh_p.shape[1] == 6:
        df = lambda t, p: p['A_0'] * np.sin(p['wd_0'] * t + p['phi_0']) + \
                          p['A_1'] * np.sin(p['wd_1'] * t + p['phi_1']) + \
                          p['A_2'] * np.sin(p['wd_2'] * t + p['phi_2']) + \
                          p['A_3'] * np.sin(p['wd_3'] * t + p['phi_3']) + \
                          p['A_4'] * np.sin(p['wd_4'] * t + p['phi_4']) + \
                          p['A_5'] * np.sin(p['wd_5'] * t + p['phi_5'])
    elif hh_p.shape[1] == 7:
        df = lambda t, p: p['A_0'] * np.sin(p['wd_0'] * t + p['phi_0']) + \
                          p['A_1'] * np.sin(p['wd_1'] * t + p['phi_1']) + \
                          p['A_2'] * np.sin(p['wd_2'] * t + p['phi_2']) + \
                          p['A_3'] * np.sin(p['wd_3'] * t + p['phi_3']) + \
                          p['A_4'] * np.sin(p['wd_4'] * t + p['phi_4']) + \
                          p['A_5'] * np.sin(p['wd_5'] * t + p['phi_5']) + \
                          p['A_6'] * np.sin(p['wd_6'] * t + p['phi_6'])
    else:
        raise ValueError('not supported number of harmonics: {}'.format(hh_p.shape[1]))

    # probe func
    pf = lambda t, p: p['I0'] * np.sin(p['wp'] * t)

    # H(t)
    OHul = [
        [J0 * Ocos(dimH), df],
        [Ophi(dimH), pf],
    ]

    return systems, OHul


def sys_parametric_driving(dimH, J0, drvL, I0, wpL):
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

    shape = len(drvL), len(wpL)
    parr = [(I0, wp, *drv) for drv in drvL for wp in wpL]
    dtype_parametric = np.dtype([
        ('I0', qo.QO.T_FLOAT),
        ('wp', qo.QO.T_FLOAT),
        ('Ad', qo.QO.T_FLOAT),
        ('wd', qo.QO.T_FLOAT),
    ])
    systems = np.array(parr, dtype=dtype_parametric).reshape(shape)

    OHul = [
        [J0 * Ocos(dimH), lambda t, p: p['Ad'] * np.sin(p['wd'] * t)],
        [Ophi(dimH), lambda t, p: p['I0'] * np.sin(p['wp'] * t)],
    ]

    return systems, OHul

def opmesolve_probe(
    rs,
    Ov,
    T,
    fft_t0,
    fft_ntp,
    fft_sample,
    dt,
    params,
    OHul,
    y0=1,
    kappa=None,
    yt_coeff=None,
    steps_chunk_size=None,
    ctx=None,
    queue=None,
    flags=None,
):
    """

        Arguments:
        ----------

        :tg: time gatter (`qo.util.time_gatter(*tg)`)

        :reduced_system: `qo.hamilton.ReducedSystem`

        :t_bath: bath temperature

        :y_0: global damping coeff

        :rho0: initial state

        :Oexpect: expectation value operator

        :OHul: time dependent unitary operators

        :params: numpy dtyped arr of parameters

        :yt_coeff: optional y(t) coefficient functions

        :kappa: custom kappa function

        :rec_skip: skip-length for fft data

        :steps_chunk_size: OpenCL integrator steps per kernel invocation

        :ctx: OpenCL context

        :queue: OpenCL queue

    """
    # time gatter
    t_wp = 2.0 * np.pi / np.min(params['wp'][0])
    tg = (0, fft_ntp * t_wp + fft_t0, dt)

    # V(t)
    texpect_v = qo.opmesolve_cl_expect(
        tg=tg,
        reduced_system=rs,
        t_bath=T,
        y_0=y0,
        rho0=rs.thermal_state(T=T),
        Oexpect=Ov,
        OHul=OHul,
        params=params.flatten(),
        yt_coeff=yt_coeff,
        rec_skip=fft_sample,
        kappa=kappa,
        steps_chunk_size=steps_chunk_size,
        ctx=ctx,
        queue=queue,
        flags=flags,
    )

    # V(w)
    dftvwp = qo.math.dft_single_freq_window(
        texpect_v,
        params['wp'].flatten(),
        dt=fft_sample * tg[2],
        tperiod=(fft_t0, tg[1]),
    )

    # sigma
    vwp = 1.0j * dftvwp.reshape(params.shape)
    vshape = (texpect_v.shape[0], *params.shape, )

    return (params['I0'] / (2 * vwp), (tg, texpect_v.reshape(vshape)))



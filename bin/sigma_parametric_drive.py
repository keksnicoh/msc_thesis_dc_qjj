#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""" Complex Conductivity calculator for single phase space
    Josephson Junction in frequency space.

    Examples:
    ---------

    - multiple driving / probing frequencies

        ```bash
        python bin/sigma_parametric_drive.py --dimH=11 \
                                             --EC=1 \
                                             --J0=9 \
                                             --y0=0.02 \
                                             --I0=0.01 \
                                             --wp 0.01 0.03 0.05 0.07 0.09 0.11 0.13 0.15 0.2 0.3 \
                                             --Ad=0.25 \
                                             --wd 2.0 3.5 \
                                             --fft_ntp 5 \
                                             --fft_sample=16 \
                                             --dt=0.0005 \
                                             --fname=my-test-data
        ```

    Notes:
    ------

    - use `export QOP_DEBUG=1` to get more detailed progress.

    - use `export QOP_ECHO_COMPILED_KERNEL=1` to get more detailed insights
      into the OpenCL kernel.

    :author: keksnicoh
    """

import argparse
import sys
import numpy as np
from pyma.phasespace import sjj_rs, Osjj_V
from pyma.probe import SigmaConfig, sys_parametric_driving, opmesolve_probe, persist_fs
from pyma.cliutil import file_exists_check, cl_ctx_queue

def stationary(T, p, w):
    if T == 0:
        return w ** 3 * np.heaviside(w, 0)
    return w**3 * (1.0 + 1.0 / (np.exp(1.0 / T * w) - 1 + 0.0000000001))

def main(argv):
    args = read_argv(argv[1:])
    assert_args(args)

    # exit / prompt user if fname does allready exist
    file_exists_check(args.fname, args.silent)

    # get OpenCL context+queue
    ctx, queue = cl_ctx_queue(args.cl_platform, args.cl_gpu_device)

    # create config
    cfg = SigmaConfig(
        dimH=args.dimH,
        EC=args.EC,
        J0=args.J0,
        y0=args.y0,
        T=args.T,
        fft_t0=args.fft_t0,
        fft_ntp=args.fft_ntp,
        fft_sample=args.fft_sample,
        dt=args.dt,
        yt_coeff=args.yt_coeff,
        cmd=' '.join(argv)
    )

    # create list of drivings [(Ad, wd), ]
    if len(args.wd) > 1 and len(args.Ad) == 1:
        drv = list(zip([args.Ad[0]] * len(args.wd), args.wd))
    elif len(args.Ad) > 1 and len(args.wd) == 1:
        drv = list(zip(args.Ad, [args.wd[0]] * len(args.Ad)))
    else:
        drv = list(zip(args.Ad, args.wd))

    # create reduced systems
    rs = sjj_rs(cfg.dimH, cfg.EC, cfg.J0, dn_coupling=args.dn_coupling)

    # voltage operator
    Ov = Osjj_V(cfg.dimH, cfg.EC)

    # create (driving, H(t))
    params, OHul = sys_parametric_driving(args.dimH, args.J0, drv, args.I0, args.wp)

    # create yt_coeff
    yt_coeff = None
    if args.yt_coeff:
        yt_coeff = lambda t, p: 1 + p['Ad'] * np.sin(p['wd'] * t)

    # Experimental ------------------------------------------------------------------------------------------
    #
    # we try some possibilities to work with dynamical coupling. This section might be removed
    # if this idea turns out to be a one-way.
    #
    kappa = None
    if args.kappa is not None:
        if args.kappa == 'kappa1':
            def kappa(T, p, w):
                return stationary(T, p, w) + p['Ad']**2 / 4 * (
                    stationary(T, p, w + p['wd']) + stationary(T, p, w - p['wd'])
                )
        elif args.kappa == 'kappa2':
            def kappa(T, p, w):
                if T == 0:
                    return w * np.heaviside(w, 0)
                return w * (1.0 + 1.0 / (np.exp(1.0 / T * w) - 1 + 0.0000000001))
        else:
            err = 'invalid kappa "{}"'
            raise ValueError(err.format(args.kappa))
    # Experimental ------------------------------------------------------------------------------------------

    # calculate sigma
    sigma, _ = opmesolve_probe(
        Ov=Ov,
        rs=rs,
        T=cfg.T,
        y0=cfg.y0,
        fft_t0=cfg.fft_t0,
        fft_ntp=cfg.fft_ntp,
        fft_sample=cfg.fft_sample,
        dt=cfg.dt,
        params=params,
        OHul=OHul,
        yt_coeff=yt_coeff,
        kappa=kappa,
        steps_chunk_size=args.chunk,
        ctx=ctx,
        queue=queue)

    # persist
    persist_fs(args.fname, cfg=cfg, systems=params, sigma=sigma)


def assert_args(args):

    if len(args.Ad) > 1 and len(args.wd) > 1 and len(args.Ad) != len(args.wd):
        err = 'cannot zip {} amplitudes Ad with {} frequencies wd.'
        raise ValueError(err.format(len(args.Ad), len(args.wd)))


def read_argv(argv):
    parser = argparse.ArgumentParser(description=(
        'thermal Josephson-Junction sigma calculator'
      + ' in phase space representation.'
    ))
    parser.add_argument('--fname',          type=str,           help="where the result is persisted.", required=True)
    parser.add_argument('--dimH',           type=int,           help="Hilbert-space dimension", required=True)
    parser.add_argument('--EC',             type=np.float32,    help="Charge Energy", default=np.float32(1))
    parser.add_argument('--J0',             type=np.float32,    help="Bare Josephson Coupling", default=np.float32(1))
    parser.add_argument('--y0',             type=np.float32,    help="Emission-rate", required=True)
    parser.add_argument('--T',              type=np.float32,    help="Bath-temperature", default=np.float32(0))
    parser.add_argument('--I0',             type=np.float32,    help="Probing-current", default=np.float32(0.01))
    parser.add_argument('--wp',             type=np.float32,    help="Probing-frequencies", nargs='+', required=True)
    parser.add_argument('--Ad',             type=np.float32,    help="driving amplitudes", nargs='+', required=True)
    parser.add_argument('--wd',             type=np.float32,    help="driving frequencies", nargs='+', required=True)
    parser.add_argument('--fft_t0',         type=np.float32,    help="time at which fft starts", default=np.float32(0))
    parser.add_argument('--fft_ntp',        type=np.int32,      help="the number of probing cycles the system must obey.", default=np.int32(2))
    parser.add_argument('--fft_sample',     type=np.int32,      help="Probing-frequencies", default=np.int32(1))
    parser.add_argument('--dt',             type=np.float32,    help="rk4 step", required=True)
    parser.add_argument('--silent',         type=bool,          help="if true the program won\'t prompt anything to the user.", default=False)
    parser.add_argument('--yt_coeff',       type=bool,          help="whether y is driven or not", default=False)
    parser.add_argument('--dn_coupling',    type=bool,          help="whether dn or sin(phi) couples", default=False)
    parser.add_argument('--kappa',          type=str,           help="custom kappa function", default=None)
    parser.add_argument('--chunk',          type=np.int32,      help="number of iterations in gpu-kernel", default=None)
    parser.add_argument('--cl_platform',    type=int,           help="OpenCL platform id", default=0)
    parser.add_argument('--cl_gpu_device',  type=int,           help="index of GPU device", default=0)

    return parser.parse_args()

if __name__ == "__main__":
    main(sys.argv)
    sys.exit(0)

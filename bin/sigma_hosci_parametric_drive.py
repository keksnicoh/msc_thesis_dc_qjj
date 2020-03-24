#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""" Linear AC Response calculator for hosci Josephson Junction.

    Examples:
    ---------

    - multiple driving / probing frequencies

        ```bash
        python ../bin/sigma_hosci_parametric_drive.py --dimH 9 \
                                                      --T 0 \
                                                      --EC 2 \
                                                      --J0 4.5 \
                                                      --y0 0.01 \
                                                      --I0 0.01 \
                                                      --wp 0.09 0.1 0.11 0.13 0.15 \
                                                      --Ad 0.05 \
                                                      --wd 2.8 3.2 \
                                                      --fft_ntp 3 \
                                                      --fft_sample 16 \
                                                      --dt 0.001 \
                                                      --fname my-test-data
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
from pyma.hosci import hosci_rs, Ov, sys_parametric_driving
from pyma.probe import SigmaConfig, opmesolve_probe, persist_fs
from pyma.cliutil import file_exists_check, cl_ctx_queue


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
    rs = hosci_rs(cfg.dimH, cfg.EC, cfg.J0, dn_coupling=args.dn_coupling)

    # create (driving, H(t))
    params, OHul = sys_parametric_driving(args.dimH, args.EC, args.J0, drv, args.I0, args.wp)

    # calculate sigma
    sigma, _ = opmesolve_probe(
        Ov=Ov(cfg.dimH, cfg.EC, cfg.J0),
        rs=rs,
        T=cfg.T,
        y0=cfg.y0,
        fft_t0=cfg.fft_t0,
        fft_ntp=cfg.fft_ntp,
        fft_sample=cfg.fft_sample,
        dt=cfg.dt,
        params=params,
        OHul=OHul,
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
      + ' in hosci approximation.'
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
    parser.add_argument('--chunk',          type=np.int32,      help="number of iterations in gpu-kernel", default=None)
    parser.add_argument('--cl_platform',    type=int,           help="OpenCL platform id", default=0)
    parser.add_argument('--cl_gpu_device',  type=int,           help="index of GPU device", default=0)
    parser.add_argument('--dn_coupling',    type=bool,          help="whether dn or sin(phi) couples", default=False)

    return parser.parse_args()

if __name__ == "__main__":
    main(sys.argv)
    sys.exit(0)

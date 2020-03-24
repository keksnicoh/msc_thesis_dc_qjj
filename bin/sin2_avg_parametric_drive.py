#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
""" current fluctuations average over avg_len samples in the evolution

    Examples:
    ---------

    ```bash
        python bin/sin2_avg_parametric_drive.py --dimH=11 \
                                             --EC=1 \
                                             --J0=9 \
                                             --y0=0.01 \
                                             --Ad=0.25 \
                                             --wd 1 \
                                             --T 0 0.71675 1.4335 2.1502 2.867 \
                                             --tf 150 \
                                             --avg_len 20000 10000 5000 1000 \
                                             --dt=0.001 \
                                             --fname=sin2-test
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
from pyma.phasespace import sjj_rs, Osin2, Ocos
from pyma.currentfluc import Sin2Config, sys_parametric_driving, opmesolve_avg_eq, persist_fs_avg
from pyma.cliutil import file_exists_check, cl_ctx_queue

def main(argv):
    args = read_argv(argv[1:])
    assert_args(args)

    # exit / prompt user if fname does allready exist
    file_exists_check(args.fname, args.silent)

    # get OpenCL context+queue
    ctx, queue = cl_ctx_queue(args.cl_platform, args.cl_gpu_device)

    # create config
    cfg = Sin2Config(
        dimH=args.dimH,
        EC=args.EC,
        J0=args.J0,
        y0=args.y0,
        T=args.T,
        avg_len=args.avg_len,
        rec_skip=args.rec_skip,
        tg=(args.t0, args.tf, args.dt),
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

    # create (driving, H(t))
    params, OHul = sys_parametric_driving(args.dimH, args.J0, drv)

    # create yt_coeff
    yt_coeff = None
    if args.yt_coeff:
        yt_coeff = lambda t, p: 1 + p['Ad'] * np.sin(p['wd'] * t)

    # calculate avg
    avg = opmesolve_avg_eq(
        rs=rs,
        T=cfg.T,
        y0=cfg.y0,
        tg=cfg.tg,
        params=params,
        Oexpect=Osin2(cfg.dimH),
        avg_len=cfg.avg_len,
        OHul=OHul,
        yt_coeff=yt_coeff,
        steps_chunk_size=args.chunk,
        rec_skip=cfg.rec_skip,
        ctx=ctx,
        queue=queue
    )

    # persist
    persist_fs_avg(args.fname, cfg=cfg, systems=params, avg=avg)


def assert_args(args):

    if args.t0 < 0 or args.t0 >= args.tf:
        err = 'invalid time gatter ({}, {}, {}'
        raise ValueError(err.format(args.t0, args.tf, args.dt))

    if len(args.Ad) > 1 and len(args.wd) > 1 and len(args.Ad) != len(args.wd):
        err = 'cannot zip {} amplitudes Ad with {} frequencies wd.'
        raise ValueError(err.format(len(args.Ad), len(args.wd)))


def read_argv(argv):
    parser = argparse.ArgumentParser(description=(
        'thermal Josephson-Junction current fluctuations calculator'
      + ' in phase space representation.'
    ))
    parser.add_argument('--fname',          type=str,           help="filepath to persist the result.", required=True)
    parser.add_argument('--dimH',           type=int,           help="Hilbert-space dimension", required=True)
    parser.add_argument('--EC',             type=np.float32,    help="Charge Energy", default=np.float32(1))
    parser.add_argument('--J0',             type=np.float32,    help="Bare Josephson Coupling", default=np.float32(1))
    parser.add_argument('--y0',             type=np.float32,    help="Damping", required=True)
    parser.add_argument('--T',              type=np.float32,    help="Bath-temperature", nargs='+', default=np.float32(0))
    parser.add_argument('--Ad',             type=np.float32,    help="driving amplitudes", nargs='+', required=True)
    parser.add_argument('--wd',             type=np.float32,    help="driving frequencies", nargs='+', required=True)
    parser.add_argument('--avg_len',        type=int,           help="list of sample lengths to create average", nargs='+', required=True)
    parser.add_argument('--t0',             type=np.float32,    help="initial time", default=0)
    parser.add_argument('--tf',             type=np.float32,    help="final time", required=True)
    parser.add_argument('--dt',             type=np.float32,    help="timestep", required=True)
    parser.add_argument('--silent',         type=bool,          help="if true the program won\'t prompt anything to the user.", default=False)
    parser.add_argument('--yt_coeff',       type=bool,          help="whether y is driven or not", default=False)
    parser.add_argument('--kappa',          type=str,           help="custom kappa function", default=None)
    parser.add_argument('--chunk',          type=np.int32,      help="number of iterations in gpu-kernel", default=None)
    parser.add_argument('--cl_platform',    type=int,           help="OpenCL platform id", default=0)
    parser.add_argument('--cl_gpu_device',  type=int,           help="index of GPU device", default=0)
    parser.add_argument('--rec_skip',       type=np.int32,      help="to define how many points of the signal are skipped", default=np.int32(1))
    parser.add_argument('--dn_coupling',    type=bool,          help="whether dn or sin(phi) couples", default=False)

    return parser.parse_args()

if __name__ == "__main__":
    main(sys.argv)
    sys.exit(0)

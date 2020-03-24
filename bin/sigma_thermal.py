#!/usr/local/bin/python3
## -*- coding: utf-8 -*-
""" to calculate sigma for a josephson junction
    :author: keksnicoh
    """
import argparse
import sys
import os
import numpy as np
import qoptical as qo
import shutil
from pyma.probe import SigmaConfig, sys_probe, opmesolve_ps_sjj, persist_fs
from pyma.cliutil import file_exists_check


def main():
    # read cli args
    args = read_argv(sys.argv[1:])

    # check whether the file to persist the result
    # does allready exist.
    file_exists_check(args.fname, args.silent)

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
    )

    # create (driving, H(t))
    driving = sys_probe(args.dimH, args.I0, args.wp)

    # calculate sigma
    sigma = opmesolve_ps_sjj(cfg, *driving, steps_chunk_size=args.chunk)

    # persist
    persist_fs(args.fname, cfg=cfg, systems=driving[0], sigma=sigma)


def read_argv(argv):
    parser = argparse.ArgumentParser(description=(
        'thermal Josephson-Junction sigma calculator'
      + ' in phase space representation.'
    ))
    parser.add_argument('--fname', type=str, help="where the result is persisted.", required=True)
    parser.add_argument('--dimH', type=int, help="Hilbert-space dimension", required=True)
    parser.add_argument('--EC', type=np.float32, help="Charge Energy", default=np.float32(1))
    parser.add_argument('--J0', type=np.float32, help="Bare Josephson Coupling", default=np.float32(1))
    parser.add_argument('--y0', type=np.float32, help="Emission-rate", required=True)
    parser.add_argument('--T', type=np.float32, help="Bath-temperature", default=np.float32(0))
    parser.add_argument('--I0', type=np.float32, help="Probing-current", default=np.float32(0.01))
    parser.add_argument('--wp', type=np.float32, help="Probing-frequencies", nargs='+', required=True)
    parser.add_argument('--fft_t0', type=np.float32, help="time at which fft starts", default=np.float32(0))
    parser.add_argument('--fft_ntp', type=np.int32, help="the number of probing cycles the system must obey.", default=np.int32(2))
    parser.add_argument('--fft_sample', type=np.int32, help="Probing-frequencies", default=np.int32(1))
    parser.add_argument('--dt', type=np.float32, help="Probing-frequencies", required=True)
    parser.add_argument('--silent', type=bool, help="Probing-frequencies", default=False)
    parser.add_argument('--chunk', type=np.int32, help="number of iterations in gpu-kernel", default=None)

    return parser.parse_args()


if __name__ == "__main__":
    main()
    sys.exit(0)

# -*- coding: utf-8 -*-
""" utils for stuff in the bin/ folder

    :author: keksnicoh
    """
import os
import shutil
import pyopencl as cl

def file_exists_check(fname, silent):
    """ checks whether a file exists and raises an FileExsitsError
        if `silent` is True or the user can decide if the file
        should be deleted or not via stdin/stdout.

        Arguments:
        ----------

        :fname: string filename

        :silent: bool silent

        """
    if os.path.exists(fname):
        if silent:
            raise FileExistsError(fname)
        else:
            print('[\033[33m???\033[0m] file "\033[36m{}\033[0m" exists, delete and continue? [N/y]: '.format(fname), end='')
            answer = input().lower().strip()
            if answer == 'y':
                shutil.rmtree(fname)
            else:
                raise FileExistsError(fname)


def cl_ctx_queue(cl_platform=0, cl_gpu_device=0):
    """ returns ctx+queue for given cl_platform
        and gpu device index

        Arguments:
        ----------

        :cl_platform: platform index

        :cl_gpu_device: indev of gpu device in the list of devides
                        having type == cl.device_type.GPU

        Returns:
        --------

        (ctx, queue) tuple

        """
    assert isinstance(cl_platform, int)
    assert isinstance(cl_gpu_device, int)

    if cl_platform < 0:
        raise ValueError('cl_platform must be positive, {} given.'.format(cl_platform))

    if cl_gpu_device < 0:
        raise ValueError('cl_gpu_device must be positive, {} given.'.format(cl_gpu_device))

    platforms = cl.get_platforms()
    if cl_platform >= len(platforms):
        err = '[\033[31mERR\033[0m] cl_platform \033[36m{}\033[0m '\
            + 'out of range, \033[36m0..{}\033[0m are available: '
        print(err.format(cl_platform, len(platforms) -1))

        for i, d in enumerate(platforms):
            hint = '      [\033[36m{}\033[0m] {}'
            print(hint.format(i, d))

        exit(1)

    platform = platforms[cl_platform]
    print('[\033[33m...\033[0m] OpenCL platform {}'.format(platform))

    gpu_devices = list(d for d in platform.get_devices() if d.type == cl.device_type.GPU)
    if cl_gpu_device >= len(gpu_devices):
        err = '[\033[31mERR\033[0m] cl_gpu_device \033[36m{}\033[0m '\
            + 'out of range, \033[36m0..{}\033[0m are available: '
        print(err.format(cl_gpu_device, len(gpu_devices) -1))

        for i, d in enumerate(gpu_devices):
            hint = '      [\033[36m{}\033[0m] {}'
            print(hint.format(i, d))

        exit(1)

    device = gpu_devices[cl_gpu_device]
    print('[\033[33m...\033[0m] OpenCL device   {}'.format(device))

    ctx = cl.Context(devices=[device])
    print('[\033[33m...\033[0m] OpenCL context  {}'.format(ctx))

    return ctx, cl.CommandQueue(ctx)


# -*- coding: utf-8 -*-
import numpy as np

def concat(a, b, fname, axis):
    try:
        return np.concatenate((a, b), axis=axis)
    except ValueError:
        raise ValueError('shape {} of {} differ from previous shapes {}'.format(b.shape, fname, a.shape))

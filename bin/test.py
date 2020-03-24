#!/usr/local/bin/python3
import numpy as np
import qoptical as qo
from qoptical.kernel_opencl import OpenCLKernel

if __name__ == "__main__":
    qo.QO.DEBUG = True

    # two state system
    rs = qo.ReducedSystem([0, 0, 0, 1], dipole=[0, 1, 1, 0])

    # three states at different temperatures
    rho0 = rs.thermal_state(T=[0, 0.1, 0.2, 0.3])

    kernel = OpenCLKernel(system=rs)
    kernel.compile()
    kernel.sync(state=rho0, t_bath=[0, 0, 0, 15], y_0=0.5)

    runner = kernel.run((0, 200, 0.0005))
    tf, rhof = kernel.reader_tfinal_rho(runner)

    print(tf, np.round(rhof, 3))
    rhoth = rs.thermal_state(T=[0, 0, 0, 15])
    print(tf, np.round(rhoth, 3))
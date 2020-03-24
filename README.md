# Dynamical Control of Quantum Josephson Junction

This repository contains the PDF of my master thesis as well as some library which was used to perform numerical calculations.

The optical lindblad master equation solver is implemented seperately in the [qoptical][1] project. This library contains specialized hamiltonians.

Please note that this repository does not contains the fully configured numerical experiments and visualization render in the thesis but the tools which where used. The numerics where perfomed partly on Amazon AWS clusters (Tesla K80) and PHYSNET (University of Hamburg).

## Setup

```bash
# Install qoptical (master equation integrator)
./setup_quoptical.sh

# install local requirements to execute bin/ files
python3 -m pip install -r requirements.txt

# install jupyter notebook + requirements
python3 -m pip install juypter
python3 -m pip install -r notebooks/requirements.txt
```

[1]: https://github.com/keksnicoh/qoptical
# Dynamical Control of Quantum Josephson Junction

This repository contains the [PDF of my master thesis][2] as well as a library which was used to perform numerical experiments. Additionally, the slides of the [pre-thesis talk][3] and the [colloqium][4] are provided.

The optical lindblad master equation solver is implemented seperately in the [qoptical][1] project. This library contains specialized hamiltonians.

Please note that this repository does not contains the fully configured numerical experiments or the plots which are rendered in the thesis. All experiments used this library. The numerics where perfomed partly on Amazon AWS clusters (Tesla K80) and PHYSNET (University of Hamburg).

Checkout the notebooks for some small configured examples.

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
[2]: https://github.com/keksnicoh/msc_thesis_dc_qjj/blob/master/dynamical-control-of-quantum-josephson-junction.pdf
[3]: https://github.com/keksnicoh/msc_thesis_dc_qjj/blob/master/pre-thesis-talk.pdf
[4]: https://github.com/keksnicoh/msc_thesis_dc_qjj/blob/master/colloquium.pdf
#!/bin/bash
set -e

UNHIGHLIGHT='\033[37m'
GREEN='\033[32m'
NC='\033[0m'

# the following packages must be installed before we
# install the qoptical lib since I'm too stupid to
# define a setup.py which can deal with installing
# all the dependencies. The problem seems that if the
# following packages are installed in the "same-run"
# like for example "qutip", the qutip setup does not recognize
# the packages. It looks like qutip is loading some packages
# to prepare install, but I dunno... just ensure that the
# following packages are installed...
echo -e "[ ${UNHIGHLIGHT}...${NC} ] install pip packages"
python3 -m pip install numpy scipy cython pybind11

# install qoptical lib
echo -e "[ ${UNHIGHLIGHT}...${NC} ] install qoptical"
python3 -m pip install -e git+https://github.com/keksnicoh/qoptical#egg=qoptical

echo -e "[ ${GREEN}OK${NC}  ] "
exit 0
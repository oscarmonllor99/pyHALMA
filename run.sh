#!/bin/bash
export OMP_STACKSIZE=4000m
export OMP_WAIT_POLICY=active
export KMP_BLOCKTIME=1000000
export memoryuse=32000m

vmemoryuse=32000m
stacksize=32000m
coredumpsize=1m

ulimit -s unlimited

python3 pyHALMA.py



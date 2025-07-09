#!/bin/bash
PBS -l walltime=00:04:00,nodes=4:ppn=2,mem=1GB
export PATH=/opt/torque-6.1.3/bin:$PATH
echo 'export PATH=/opt/torque-6.1.3/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
module load torque
module load maui/maui-3.3.1
module load mpi/openmpi-x86_64.local
pwd
# write your own command here

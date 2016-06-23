#!/bin/bash
#PBS -l select=1:ncpus=16:mem=252GB
#PBS -l walltime=48:00:00
#PBS -q ldan
#PBS -m abe
#PBS -N output_netcdf

# force reading of bash profile
# why is this necessary?!?
. $HOME/.bash_profile 

module load python/2.7.9
which python

cd /pleiades/home3/rpaberna/Private/MITgcm_parallel_analysis/scripts
echo "LLC: $LLC"
echo "running output_tile_netcdf.py"

# 701 - Gulf Stream
# 263 - ACC
# 415 - Subtropical Pac.

tid=725
niter=106128
python output_grid_netcdf.py $tid $niter
python output_tile_netcdf.py $tid $niter



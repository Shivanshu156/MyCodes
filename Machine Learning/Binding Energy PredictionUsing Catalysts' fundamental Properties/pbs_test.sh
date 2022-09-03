#!/usr/bin/env bash
#PBS -N PythonJob
#PBS -P chemical 
#PBS -m bea
#PBS -M $USER@iitd.ac.in
#PBS -l select=1:ncpus=24:mpiprocs=24
#PBS -l walltime=02:00:00
#PBS -q standard

# Environment
echo "==============================="
echo $PBS_JOBID
cat $PBS_NODEFILE
echo "==============================="
cd $PBS_O_WORKDIR
module purge
module load apps/anaconda3/4.6.9 

#job 
python catalyst_data_extract.py

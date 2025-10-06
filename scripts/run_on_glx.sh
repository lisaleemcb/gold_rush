#!/bin/bash

#SBATCH --job-name=gold_rush
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1                     # One task per job
#SBATCH --cpus-per-task=1              # One CPU per task
#SBATCH --output=gold_output.out
#SBATCH --error=gold_error.err
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=lisaleemcb@gmail.com

# use the bash shell
set -x
# echo each command to standard out before running it
date

# source bash profile
source /home/emc-brid/.bashrc
source ~/venvs/gold_rush/bin/activate


# run the Unix 'date' command
echo "Hello world, from the Cluster!"
# run the Unix 'echo' command
# which mamba
# mamba activate kSZ
which python

python -u /home/emc-brid/gold_rush/scripts/run_mcmc.py

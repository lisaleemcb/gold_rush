#!/bin/bash

#SBATCH --job-name=gold_rush
#SBATCH --time=00-12:00:00
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=lisaleemcb@gmail.com
#SBATCH --ntasks-per-node=12

# use the bash shell
set -x
# echo each command to standard out before running it
date
# run the Unix 'date' command
echo "Hello world, from Bridges-2!"
# run the Unix 'echo' command
python /jet/home/emcbride/packages/gold_rush/src/gold_rush/run_mcmc.py

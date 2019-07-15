#!/bin/bash
#SBATCH --no-requeue
#SBATCH --qos=normal
#SBATCH --partition=evanreed
#SBATCH --output=job_out_rsf.log
#SBATCH --error=job_err_rsf.log
#SBATCH --mem=64G
#SBATCH --job-name=rsf_Ih
#SBATCH --ntasks-per-node=20
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1

# Modules for Ovito.
ml load system
ml load qt

compute_rsf="$HOME/simulations/compute_rsf.py"
cart_coords="$HOME/rsf_code_and_data/cartesian/dump_250K_1960000.dat"
out_rsf_file="$HOME/rsf_code_and_data/rsfs/rsfs_mixed_bigsigma_250k_1960000.dat"

srun ovitos ${compute_rsf} --cart_coord_fname ${cart_coords} \
                           --out_rsf_fname ${out_rsf_file}

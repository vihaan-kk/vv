#!/bin/bash
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH -t 0:30:00
#SBATCH -p l40-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --mail-type=begin,end,fail
#SBATCH --mail-user=vkereka@email.unc.edu

module purge
module load anaconda

cd ~/vv_bench
python run.py

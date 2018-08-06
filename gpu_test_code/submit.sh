#!/bin/bash
#
#SBATCH --job-name=RedTossOK
#SBATCH --output=redtossok.txt
#SBATCH --error=redtossok.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:p100:4

python gpu_list.py
python gpu_test.py

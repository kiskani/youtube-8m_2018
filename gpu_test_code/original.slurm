#!/bin/bash
#note - there can be no empty lines between #SBATCH directives.
#SBATCH --job-name=gpuTest
#SBATCH --output=gpuTest_%j.out
#SBATCH --error=gpuTest_%j.err
#SBATCH --ntasks=1
# allocate 5GB or RAM on node. You must declare --mem in all scripts
#SBATCH --mem=5000
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=96x24gpu4
# The following designates you are using GPU 1 - Telse P100 model
#SBATCH --gres=gpu:p100:1

module load cuda/cuda-9.1
module load python-3.6.2

#python3 gpu_list.py


#!/bin/bash
#
#SBATCH --job-name=Test_Down
#SBATCH --error=/hb/scratch/mkarimz1/error3_test
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=Instruction

curl data.yt8m.org/download.py | partition=2/frame/test mirror=us python 

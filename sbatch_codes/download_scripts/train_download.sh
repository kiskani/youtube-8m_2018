#!/bin/bash
#
#SBATCH --job-name=Download
#SBATCH --error=/hb/scratch/mkarimz1/error3
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=128x24

curl data.yt8m.org/download.py | partition=2/frame/train mirror=us python 

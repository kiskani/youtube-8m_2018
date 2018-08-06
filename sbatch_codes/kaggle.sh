#!/bin/bash
#
#SBATCH --job-name=Submission
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/error/kaggle-submission-LstmModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=Instruction

kaggle competitions submit -c youtube8m-2018 -f l.csv -m "New model from rank 5 last year"

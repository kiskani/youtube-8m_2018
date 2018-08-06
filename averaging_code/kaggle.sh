#!/bin/bash
#
#SBATCH --job-name=Submission
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/error/frame/kaggle-submission-LstmModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=128x24

kaggle competitions submit -c youtube8m-2018 -f /hb/scratch/mkarimz1/yt8m/rank5-2017/averaging_code/combined_submission_2.csv -m "Combined average of the past 5 csv submissions"

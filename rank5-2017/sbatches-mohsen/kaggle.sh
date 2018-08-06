#!/bin/bash
#
#SBATCH --job-name=Submission
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/error/kaggle-submission-LstmModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=128x24

kaggle competitions submit -c youtube8m-2018 -f /hb/scratch/mkarimz1/yt8m/rank5-2017/model/frame/Lstmbidirect//ema_cp114403/predictions_lstm_ema_ckpt-125416.csv -m "Rank 5 last year with 125416 training checkpoint" 

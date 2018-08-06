#!/bin/bash
#
#SBATCH --job-name=ValidateFrame
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/log_outputs/frame/LstmModel_validate_full_log_cpu
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#SBATCH --time=480:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=128x24

python3 /hb/home/mkarimz1/yt8m/code/eval.py --batch_size=512 --eval_data_pattern=/hb/scratch/mkarimz1/yt8m/frame/validate*.tfrecord --train_dir=/hb/scratch/mkarimz1/yt8m/model/frame/LstmModel_full

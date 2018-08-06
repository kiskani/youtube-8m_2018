#!/bin/bash
#
#SBATCH --job-name=Inference
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/log_outputs/frame/LstmModel_inference_full_log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=128x24

python3 /hb/home/mkarimz1/yt8m/code/inference.py --batch_size=4096 --input_data_pattern=/hb/scratch/mkarimz1/yt8m/frame/test*.tfrecord --train_dir=/hb/scratch/mkarimz1/yt8m/model/frame/LstmModel_full --output_file=/hb/scratch/mkarimz1/yt8m/output/frame/kaggle_solution_LstmModel_full.csv



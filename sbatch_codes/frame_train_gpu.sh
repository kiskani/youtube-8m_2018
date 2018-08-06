#!/bin/bash
#
#SBATCH --job-name=FrameTrain
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/log_outputs/frame/NetVLAD_train_full_log
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#SBATCH --time=480:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:p100:1

python /hb/home/mkarimz1/yt8m/willow-code/train.py --frame_features --feature_names='rgb,audio' --feature_sizes='1024,128' --batch_size=512 --train_data_pattern=/hb/scratch/mkarimz1/yt8m/frame/train*.tfrecord --train_dir=/hb/scratch/mkarimz1/yt8m/model/frame/NetVLAD_full --model=NetVLAD(1024,20,20,20,True) --start_new_model

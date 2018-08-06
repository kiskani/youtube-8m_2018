#!/bin/bash
#
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/log_outputs/frame/DBoF_train_full_log
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#SBATCH --time=480:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:p100:1

model="DBoF"

path_to_code="/hb/home/mkarimz1/yt8m/willow-code"
path_to_features="/hb/scratch/mkarimz1/yt8m/frame_short"
path_to_model="/hb/scratch/mkarimz1/yt8m/model/frame/$model"

python "$path_to_code/train.py" --frame_features --feature_names='rgb,audio' --feature_sizes='1024,128' --batch_size=512 --train_data_pattern="$path_to_features/train*.tfrecord" --train_dir=$path_to_model --model=$model --start_new_model

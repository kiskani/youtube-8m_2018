#!/bin/bash
#
#SBATCH --job-name=RedTossOK
#SBATCH --output=redtossok.txt
#SBATCH --error=redtossok.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=256x44

module load cuda/cuda-9.1
module load python-3.6.2 

python3 /hb/home/mkarimz1/yt8m/code/train.py --feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128' --train_data_pattern=/hb/home/mkarimz1/yt8m/data/train*.tfrecord --train_dir=/hb/home/mkarimz1/yt8m/model/deep_combine_chain_model --model=DeepCombineChainModel --start_new_model

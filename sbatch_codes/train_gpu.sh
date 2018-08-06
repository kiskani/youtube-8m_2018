#!/bin/bash
#
#SBATCH --job-name=GreenToss
#SBATCH --error=/hb/home/mkarimz1/yt8m/log_outputs/DeepCombineChainModel_numsupp100_layer7_relucells500_train_log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:p100:1

python /hb/home/mkarimz1/yt8m/code/train.py --feature_names='mean_rgb,mean_audio' --feature_sizes='1024,128' --train_data_pattern=/hb/home/mkarimz1/yt8m/data/train*.tfrecord --train_dir=/hb/home/mkarimz1/yt8m/model/DeepCombineChainModel_numsupp100_layer7_relucells500 --model=DeepCombineChainModel --start_new_model --num_supports=100 --deep_chain_layers=7 --deep_chain_relu_cells=500

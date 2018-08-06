#!/bin/bash
#
#SBATCH --job-name=Train
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/rank5-2017/log_outputs/frame/Lstmbidirect_train_log
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#SBATCH --time=480:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:p100:1

model="Lstmbidirect"

CODEPATH="/hb/home/mkarimz1/yt8m/rank5-2017/marcin-pekalski-code/frame_level_code/train.py"
SAVEPATH="/hb/scratch/mkarimz1/yt8m/rank5-2017/model/frame/$model"
DATAPATH="/hb/scratch/mkarimz1/yt8m/frame/train*.tfrecord"

python $CODEPATH \
  --train_data_pattern=$DATAPATH \
  --model=$model \
  --frame_features \
  --feature_names='rgb,audio' --feature_sizes='1024,128' \
  --batch_size=256  \
  --train_dir=$SAVEPATH \
  --base_learning_rate=0.00025 \
  --lstm_cells=1200 \
  --num_epochs=6

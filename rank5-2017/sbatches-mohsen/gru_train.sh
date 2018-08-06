#!/bin/bash
#
#SBATCH --job-name=GRUTrain
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/rank5-2017/log_outputs/frame/GRUbidirect_train_log
#SBATCH --ntasks=10
#SBATCH --mem=50000
#SBATCH --cpus-per-task=4
#SBATCH --time=480:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=256x44
#SBATCH --nodes=1

model="GRUbidirect"

CODEPATH="/hb/home/mkarimz1/yt8m/rank5-2017/marcin-pekalski-code/frame_level_code/train.py"
GRUSAVEPATH="/hb/scratch/mkarimz1/yt8m/rank5-2017/model/frame/$model"
DATAPATH="/hb/scratch/mkarimz1/yt8m/frame/train*.tfrecord"

module load python-3.6.2

for FOLD in {0..4}; do 
  SAVEPATH="${GRUSAVEPATH}_${FOLD}"

  python3 $CODEPATH \
     --train_data_pattern=$DATAPATH \
     --model=$model \
     --video_level_classifier_model="MoeModel" \
     --frame_features \
     --feature_names='rgb,audio' --feature_sizes='1024,128' \
     --batch_size=256  \
     --use_cv=True \
     --fold=${FOLD} \
     --split_seed=11 \
     --train_dir=$SAVEPATH \
     --base_learning_rate=0.00025 \
     --lstm_cells=1250 \
     --num_epochs=6

done

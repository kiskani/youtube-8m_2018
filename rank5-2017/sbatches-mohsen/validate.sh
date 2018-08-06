#!/bin/bash
#
#SBATCH --job-name=Validate
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/rank5-2017/log_outputs/frame/Lstmbidirect_validate_log
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#SBATCH --time=480:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=256x44

module load python-3.6.2

model="Lstmbidirect"

CODEPATH="/hb/home/mkarimz1/yt8m/rank5-2017/marcin-pekalski-code/frame_level_code/eval.py"
SAVEPATH="/hb/scratch/mkarimz1/yt8m/rank5-2017/model/frame/$model"
DATAPATH="/hb/scratch/mkarimz1/yt8m/frame/validate*.tfrecord"

python3 $CODEPATH \
  --eval_data_pattern=$DATAPATH \
  --batch_size=256  \
  --train_dir=$SAVEPATH \

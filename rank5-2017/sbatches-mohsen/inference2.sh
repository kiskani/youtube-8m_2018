#!/bin/bash
#
#SBATCH --job-name=Inf2
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/rank5-2017/log_outputs/frame/Lstmbidirect_inference_2_log
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#SBATCH --time=480:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=Instruction

module load python-3.6.2

model="Lstmbidirect"
CHECKPOINT=108940
TESTDATAPATH="/hb/scratch/mkarimz1/yt8m/frame/test*.tfrecord"
SAVEPATH="/hb/scratch/mkarimz1/yt8m/rank5-2017/model/frame/$model"
CODEPATH="/hb/home/mkarimz1/yt8m/rank5-2017/marcin-pekalski-code/frame_level_code/inference.py"
SAVEPATH2=${SAVEPATH}/ema_cp104617

python3 $CODEPATH --feature_names='rgb,audio' --feature_sizes='1024,128' --top_k=30 --use_ema_var=True --batch_size=300 --num_readers=10 --frame_features --checkpoint_file="${SAVEPATH2}/model.ckpt-${CHECKPOINT}" --train_dir="${SAVEPATH2}" --input_data_pattern="${TESTDATAPATH}" --output_file="${SAVEPATH2}/predictions_lstm_ema_ckpt-${CHECKPOINT}.csv"

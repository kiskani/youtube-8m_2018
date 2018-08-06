#!/bin/bash
#
#SBATCH --job-name=EMA
#SBATCH --error=/hb/scratch/mkarimz1/yt8m/rank5-2017/log_outputs/frame/Lstmbidirect_EMA_log
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --cpus-per-task=1
#SBATCH --time=480:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=96x24gpu4
#SBATCH --gres=gpu:p100:1

model="Lstmbidirect"
SAVEPATH="/hb/scratch/mkarimz1/yt8m/rank5-2017/model/frame/$model"
CODEPATH="/hb/home/mkarimz1/yt8m/rank5-2017/marcin-pekalski-code/frame_level_code/generate_EMAmodel.py"
WEIGHTSSOURCE=${SAVEPATH}/model.ckpt-104617
MODELSOURCE=${SAVEPATH}/model.ckpt-104617
SAVEPATH2=${SAVEPATH}/ema_cp104617

python "$CODEPATH" "$WEIGHTSSOURCE" "$MODELSOURCE" "$SAVEPATH2/model.ckpt-104617" 

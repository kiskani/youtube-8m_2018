#!/bin/bash
#
#SBATCH --job-name=ValidateFrame
#SBATCH --error=/hb/home/mkarimz1/yt8m/log_outputs/FrameLevelLogisticModel
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=480:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkarimz1@ucsc.edu
#SBATCH --partition=96x24gpu4

python /hb/home/mkarimz1/yt8m/code/eval.py --eval_data_pattern=/hb/scratch/mkarimz1/frame/validate*.tfrecord --train_dir=/hb/scratch/mkarimz1/model/FrameLevelLogisticModel --run_once 

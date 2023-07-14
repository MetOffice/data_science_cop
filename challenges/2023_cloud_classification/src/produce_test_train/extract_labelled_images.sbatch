#!/bin/bash
#SBATCH --array=0-220
#SBATCH --mem=8000
#SBATCH --output=/home/h06/meastman/projects/understanding-clouds-kaggle/src/sbatch/o/extract_lab_img_%a.out
#SBATCH --error=/home/h06/meastman/projects/understanding-clouds-kaggle/src/sbatch/e/extract_lab_img_%a.err
#SBATCH --time=240
#SBATCH --mail-user=michael.eastman@metoffice.gov.uk
#SBATCH --mail-type=ALL

module load scitools

cd /home/h06/meastman/projects/understanding-clouds-kaggle

# 5546 unique images. Iterate through and process 25 at a time
python src/functions/extract_labelled_images.py --step $SLURM_ARRAY_TASK_ID --step_len 25

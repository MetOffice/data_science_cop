#!/bin/bash -l
#SBATCH --mem=32G
#SBATCH --ntasks=4
#SBATCH --time=60
#SBATCH --job-name=conda_build_spice

export REQUIREMENTS_FILE=$1
export ENV_NAME=$2
export TARGET=$3

if [$TARGET == "gpu"]; then

    echo targeting GPU, setting cuda override
    export CONDA_CUDA_OVERRIDE=13.2
else
    echo targeting platform $TARGET
fi

conda env create --file $REQUIREMENTS_FILE --name ${ENV_NAME}


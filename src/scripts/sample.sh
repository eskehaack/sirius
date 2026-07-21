#!/bin/bash
#BSUB -q gpuv100
#BSUB -J sample_icyalert
#BSUB -n 1
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:10
#BSUB -o hpc_outputs/cfg_sample_%J.out
#BSUB -e hpc_outputs/cfg_sample_%J.err
#BSUB -B
#BSUB -N

# module load python3/3.12.4
# module load cuda/12.6.0

cd /work3/s214643/sirius
source /work3/s214643/sirius/.venv/bin/activate

python -m src.sample \
    --timestamp "20260721144907" \
    --checkpoint "last" \
    --sample_index 0 \
    --sample_dim 0
#!/bin/bash
#BSUB -q gpuv100
#BSUB -J sample_icyalert
#BSUB -n 4
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:05
#BSUB -o hpc_outputs/cfg_sample_%J.out
#BSUB -e hpc_outputs/cfg_sample_%J.err
#BSUB -B
#BSUB -N

# module load python3/3.12.4
# module load cuda/12.8.0

cd /work3/s214643/sandbox
source /work3/s214643/sandbox/.venv/bin/activate

python sample.py \
    --timestamp "20260719120354" \
    --checkpoint "last" \
    --sample_index 0 \
    --sample_dim 1
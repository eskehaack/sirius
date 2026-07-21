#!/bin/bash
#BSUB -q gpuv100
#BSUB -J pilot_icyalert_4
#BSUB -n 4
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 3:00
#BSUB -o hpc_outputs/cfg_%J.out
#BSUB -e hpc_outputs/cfg_%J.err
#BSUB -B
#BSUB -N

# module load python3/3.12.4
# module load cuda/12.6.0

cd /work3/s214643/sirius
source /work3/s214643/sirius/.venv/bin/activate
ts="$(date +%Y%m%d%H%M%S)"

python -m src.train \
    --batch_size 4 \
    --max_epochs 2 \
    --lr 1e-4 \
    --base_channels 64 \
    --num_workers 4 \
    --target_channels 1 \
    --condition_channels 23 \
    --checkpoint_dir "./checkpoints/$ts" \
    --timesteps 100

python -m src.sample \
    --timestamp "$ts" \
    --sample_index 0

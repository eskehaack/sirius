#!/bin/bash
#BSUB -q gpuv100
#BSUB -J pilot_icyalert
#BSUB -n 16
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 0:30
#BSUB -o hpc_outputs/cfg_%J.out
#BSUB -e hpc_outputs/cfg_%J.err
#BSUB -B
#BSUB -N

# module load python3/3.12.4
# module load cuda/12.8.0

cd /work3/s214643/sandbox
source /work3/s214643/sandbox/.venv/bin/activate
ts="$(date +%Y%m%d%H%M%S)"

python train.py \
    --data_root ./.data \
    --batch_size 16 \
    --max_epochs 30 \
    --lr 1e-4 \
    --base_channels 64 \
    --num_workers 16 \
    --target_channels 3 \
    --condition_channels 3 \
    --checkpoint_dir "./checkpoints/$ts" \
    --timesteps 50

python sample.py \
    --timestamp "$ts" \
    --sample_index 0
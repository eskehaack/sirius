#!/bin/bash
#BSUB -q hpc
#BSUB -J download_icyalert
#BSUB -n 1
#BSUB -R "rusage[mem=5GB]"
#BSUB -R "span[hosts=1]"
#BSUB -W 10:00
#BSUB -o hpc_outputs/cfg_download_%J.out
#BSUB -e hpc_outputs/cfg_download_%J.err
#BSUB -B
#BSUB -N

# module load python3/3.12.4
# module load cuda/12.8.0

cd /work3/s214643/sandbox
source /work3/s214643/sandbox/.venv/bin/activate

python cdsClass.py
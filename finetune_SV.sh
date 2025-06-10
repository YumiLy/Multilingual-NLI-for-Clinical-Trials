#!/bin/bash
#SBATCH --partition=gpu-1semaine
#SBATCH --job-name=finetune_SV
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6-8:00:00
#SBATCH --output=results/logs_finetune_SV.out
#SBATCH --error=results/logs_finetune_SV.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=lai@isir.upmc.fr


# ====== Load your environment ======
source ~/anaconda3/etc/profile.d/conda.sh
conda activate multimodal

# ====== Optional: sanity check ======
echo "Running on host: $(hostname)"
nvidia-smi
python -c "import torch; print(torch.version.cuda)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.9,max_split_size_mb:512"
export CUDA_LAUNCH_BLOCKING=0  # Disable for better throughput


for idx in 4 6 8 10; do
    echo "=============================="
    echo "Running with freeze_layers: $idx"
    echo "=============================="
    python -u /home/lai/models/Wav2Vec2/finetune_SV.py --freeze_layers=$idx \
        > results/log_finetune_SV_layer${idx}.out 2> results/log_finetune_SV_layer${idx}.err
done
#!/bin/bash

#SBATCH --qos=default
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=MT_NLLB_600M_statement.stdout
#SBATCH --job-name=MT_NLLB_600M_statement
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n55

source /mnt/beegfs/home/lai/miniconda3/etc/profile.d/conda.sh
conda activate mt

python nllb_statement.py \
    --data ../nli4ct/ \
    --output ../nli4ct_nllb600m/ \
    --ct_model_path mt_model/ct2/nllb-200-distilled-600M-int8 \
    --sp_model_path flores200_sacrebleu_tokenizer_spm.model
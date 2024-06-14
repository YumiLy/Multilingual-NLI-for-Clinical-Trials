#!/bin/bash

#SBATCH --qos=default
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=MT_NLLB_3.3B.stdout
#SBATCH --job-name=MT_NLLB_3.3B
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n53

source /mnt/beegfs/home/lai/miniconda3/etc/profile.d/conda.sh
conda activate mt

python nllb_CT_json.py \
    --data ../nli4ct/CT_json \
    --output ../nli4ct_nllb3.3b/CT_json \
    --ct_model_path mt_model/ct2/nllb-200-3.3B-int8 \
    --sp_model_path flores200_sacrebleu_tokenizer_spm.model
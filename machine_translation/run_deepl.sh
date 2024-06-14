#!/bin/bash

#SBATCH --qos=default
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=MT_DeepL.stdout
#SBATCH --job-name=MT_DeepL
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n101

source /mnt/beegfs/home/lai/miniconda3/etc/profile.d/conda.sh
conda activate mt


# 检查并删除已存在的结果目录，仅为了测试
if [ -d "../nli4ct_deepl" ]; then
    echo "Removing existing results directory."
    rm -rf ../nli4ct_deepl
fi

# 创建新的结果目录
mkdir ../nli4ct_deepl

python deepl_json.py \
    --data ../nli4ct/ \
    --output ../nli4ct_deepl/
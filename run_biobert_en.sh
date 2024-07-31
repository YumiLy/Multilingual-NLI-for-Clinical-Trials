#!/bin/bash

#SBATCH --qos=default
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=test_biobert_en.stdout
#SBATCH --job-name=biobert_en
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n101


source /mnt/beegfs/home/lai/miniconda3/etc/profile.d/conda.sh
conda activate ocnli  # 激活您的 Conda 环境


# 检查并删除已存在的结果目录，仅为了测试
if [ -d "results_biobert" ]; then
    echo "Removing existing results directory."
    rm -rf ../results_biobert
fi

# 创建新的结果目录
mkdir results_biobert

# 运行 Python 脚本
python -m run_classifier \
       --task_name=nli4ct \
       --do_train=true \
       --do_eval=true \
       --do_predict=true \
       --data_dir=../data/nli4ct_en/ \
       --vocab_file=./weights_biobert/vocab.txt \
       --bert_config_file=./weights_biobert/bert_config.json \
       --init_checkpoint=./weights_biobert/model.ckpt-150000 \
       --max_seq_length=256 \
       --train_batch_size=64 \
       --learning_rate=2e-5 \
       --num_train_epochs=20.0 \
       --output_dir=results_biobert \
       --keep_checkpoint_max=1 \
       --save_checkpoints_steps=2500 \
       --max_input=inf



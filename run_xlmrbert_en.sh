#!/bin/bash

#SBATCH --qos=default
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=2
#SBATCH --nodes=1
#SBATCH --output=test_xlmrbert_en.stdout
#SBATCH --job-name=xlmrbert_en
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --nodelist=n102


source /mnt/beegfs/home/lai/miniconda3/etc/profile.d/conda.sh
conda activate ocnli  # 激活您的 Conda 环境

if [ -d "results_xlmrbert_en" ]; then
    echo "Removing existing results directory."
    rm -rf ../results_xlmrbert_en
fi

mkdir results_xlmrbert_en

# 运行 Python 脚本
python -m run_classifier \
       --task_name=nli4ct \
       --do_train=true \
       --do_eval=true \
       --do_predict=true \
       --data_dir=../data/nli4ct_en/ \
       --vocab_file=./weights_xlmr/vocab.txt \
       --bert_config_file=./weights_xlmr/bert_config.json \
       --init_checkpoint=./weights_xlmr/bert_model.ckpt \
       --max_seq_length=128 \
       --train_batch_size=32 \
       --learning_rate=2e-5 \
       --num_train_epochs=20.0 \
       --output_dir=results_xlmrbert_en \
       --keep_checkpoint_max=1 \
       --save_checkpoints_steps=2500 \
       --max_input=inf



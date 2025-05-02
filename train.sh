#!/bin/bash
#SBATCH --job-name=etcaps_train        
#SBATCH --output=logs/etcaps_%j.out   
#SBATCH --error=logs/etcaps_%j.err     
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=4              
#SBATCH --gres=gpu:7 
#SBATCH --mem=32G                      
#SBATCH --time=24:00:00                
#SBATCH --partition=gpu                

source ./venv/bin/activate

python main.py \
    --data_dir ./data \
    --dataset svhn \
    --batch_size 64 \
--model_name etcaps \
--epochs 50 \
--num_caps 32 \
--depth 1 \


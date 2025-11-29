#!/bin/bash

dataset_path="C:/your_dataset_path"
model_path="C:/your_model_path"
chkpoint_save_path="C:/your_checkpoint_path"

epoch=100
dataset_num=5000

python train.py --epoch "$epoch" --dataset_num "$dataset_num" --batch_size 20 --dataset_path "$dataset_path" --chkpoint_save_path "$chkpoint_save_path"


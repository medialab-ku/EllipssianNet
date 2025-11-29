#!/bin/bash

save_path="C:/your_dataset_path"

for (( x=0; x<=9; x++ ))
do
   iter=$((x+1))
   echo "Running iteration $iter/3"
   python create_dataset.py --iteration $x --batch 200 --render True --save_path "$save_path"
done
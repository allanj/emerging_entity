#!/bin/bash


## To run this script in background: nohup bash pipeline.bash > YOUR_LOG_FILE 2>&1 &

dataset=conll2003

extraction_training_log=logs/extraction.log
typing_training_log=logs/typing.log
device=cpu



## train the extraction model and obtain results for baseline
python3 trainer.py --dataset ${dataset} --extraction_model 1 --device ${device} --model_folder extraction > ${extraction_training_log} 2>&1 &

## Train the typing model using the original training data
python3 trainer.py --dataset ${dataset} --typing_model 1 --model_folder typing --device ${device} > ${typing_training_log} 2>&1


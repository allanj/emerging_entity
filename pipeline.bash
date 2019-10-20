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


## We actually run the above two program at the same time, so make sure their model folder name do not clash with each other

## use the trained typing model to decode on the predicition in the first step
python3 typing_extracted_results.py --typing_model typing --extraction_result extraction.results


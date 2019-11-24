#!/bin/bash


## To run this script in background: nohup bash pipeline.bash > YOUR_LOG_FILE 2>&1 &

datasets=(conll2003)

device=cuda:1
negs=(1)
boundarys=(1)
starts=(0 1 2 3 4)

run_latent_model=1
context_emb=elmo

if [ $run_latent_model = 0 ]
then
    ## run baseline model
    python3 trainer.py --dataset ${dataset} --model_folder benchmark --use_fined_labels 0 > logs/ontonotes.log --device cuda:0 2>&1

else
    ## Below for latent model
    for (( d=0; d<${#datasets[@]}; d++ )) do
        dataset=${datasets[$d]}
        for (( s=0; s<${#starts[@]}; s++ )) do
            start=${starts[$s]}
            for (( n=0; n<${#negs[@]}; n++ )) do
                neg=${negs[$n]}
                for (( b=0; b<${#boundarys[@]}; b++ )) do
                    boundary=${boundarys[$b]}
                    if [ $neg = 0 ] && [ $boundary = 0 ] ## no operation done for both are 0
                    then
                        continue
                    fi
                    model_folder=${dataset}_${start}_neg_${neg}_boundary_${boundary}_${context_emb}
                    log_file=logs/${dataset}_${start}_neg_${neg}_boundary_${boundary}_${context_emb}.log
                    python3 trainer.py --dataset ${dataset}  --device ${device} --model_folder ${model_folder} \
                     --add_label_constraint 1 --new_type MISC --use_neg_labels ${neg} --use_boundary ${boundary} --use_fined_labels 1 \
                     --inference_method softmax --use_hypergraph 1 --start_num ${start} --context_emb ${context_emb} > ${log_file} 2>&1
                done
            done
        done
    done
fi





#!/bin/bash


## To run this script in background: nohup bash pipeline.bash > YOUR_LOG_FILE 2>&1 &

datasets=(conll2003)
new_type=MISC
train_num=100
choose_new_type=1


device=cuda:1
negs=(1)
boundarys=(1)
starts=(0)

run_latent_model=0
context_emb=none

num_epoch=150
emb=data/glove.6B.100d.txt


if [ $run_latent_model = 0 ]
then
    ## run baseline model
    for (( d=0; d<${#datasets[@]}; d++ )) do
 	       dataset=${datasets[$d]}
           model_folder=${dataset}_${choose_new_type}
           log_file=logs/${model_folder}.log
           python3 trainer.py --dataset ${dataset} --model_folder ${model_folder} --use_fined_labels 0 --device ${device} \
                    --choose_by_new_type ${choose_new_type} --train_num ${train_num} --embedding_file ${emb}  > ${log_file} 2>&1 &

    done

else
    ## Below for latent model
    for (( d=0; d<${#datasets[@]}; d++ )) do
        dataset=${datasets[$d]}
        for (( b=0; b<${#boundarys[@]}; b++ )) do
            boundary=${boundarys[$b]}
            for (( n=0; n<${#negs[@]}; n++ )) do
                neg=${negs[$n]}
                if [ $neg = 0 ] && [ $boundary = 0 ] ## no operation done for both are 0
                then
                    continue
                fi
                if [ $neg = 1 ] && [ $boundary = 0 ] #
                then
                    starts=(0 1 2 3)
                fi
                if [ $neg = 0 ] && [ $boundary = 1 ] #
                then
                    starts=(0 1)
                fi

                for (( s=0; s<${#starts[@]}; s++ )) do
                    start=${starts[$s]}
                    model_folder=${dataset}_${start}_neg_${neg}_boundary_${boundary}_${context_emb}
                    log_file=logs/${dataset}_${start}_neg_${neg}_boundary_${boundary}_${context_emb}.log
                    python3 trainer.py --dataset ${dataset}  --device ${device} --model_folder ${model_folder} --embedding_file ${emb} \
                     --add_label_constraint 1 --new_type ${new_type} --use_neg_labels ${neg} --use_boundary ${boundary} --use_fined_labels 1 \
                     --inference_method softmax --use_hypergraph 1 --start_num ${start} --num_epocs ${num_epoch} \
                     --train_num ${train_num}  --context_emb ${context_emb} > ${log_file} 2>&1
                done
            done
        done
    done
fi





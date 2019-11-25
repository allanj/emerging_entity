#!/bin/bash


## To run this script in background: nohup bash pipeline.bash > YOUR_LOG_FILE 2>&1 &

datasets=(conll2003)
new_types=(ORG)
train_nums=(10 20 50 100)
choose_new_type=1


device=cuda:1
negs=(0 1)
boundarys=(0 1)
starts=(0 1 2 3 4)

run_latent_model=0
context_emb=none
heuristic=0

num_epoch=150

if [ $run_latent_model = 0 ]
then
    ## run baseline model
    for (( d=0; d<${#datasets[@]}; d++ )) do
        for (( n=0; n<${#new_types[@]}; n++ )) do
             for (( t=0; t<${#train_nums[@]}; t++ )) do
 	           dataset=${datasets[$d]}
               new_type=${new_types[$n]}
               number=${train_nums[$t]}
               model_folder=${dataset}_${new_type}_random_${number}_${choose_new_type}
               data_file=${dataset}/${new_type}/few_random_${number}
               log_file=logs/${model_folder}_choose_new_type_${choose_new_type}.log
               python3 trainer.py --dataset ${data_file} --model_folder ${model_folder} --use_fined_labels 0 --device ${device} \
                    --choose_by_new_type ${choose_new_type} --new_type ${new_type} --num_epochs ${num_epoch} > ${log_file} 2>&1 &
             done
        done
    done

else
    ## Below for latent model
    for (( d=0; d<${#datasets[@]}; d++ )) do
        dataset=${datasets[$d]}
        for (( n=0; n<${#new_types[@]}; n++ )) do
            for (( t=0; t<${#train_nums[@]}; t++ )) do
                new_type=${new_types[$n]}
                number=${train_nums[$t]}
                data_file=${dataset}/${new_type}/few_random_${number}
                for (( b=0; b<${#boundarys[@]}; b++ )) do
                    boundary=${boundarys[$b]}
                    for (( n1=0; n1<${#negs[@]}; n1++ )) do
                        neg=${negs[$n1]}
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
                            model_folder=${dataset}_${new_type}_random_${number}_${choose_new_type}_start_${start}_neg_${neg}_boundary_${boundary}_${context_emb}_ht_${heuristic}
                            log_file=logs/${model_folder}.log
                            python3 trainer.py --dataset ${data_file}  --device ${device} --model_folder ${model_folder} --heuristic ${heuristic} \
                             --add_label_constraint 1 --new_type ${new_type} --use_neg_labels ${neg} --use_boundary ${boundary} --use_fined_labels 1 \
                             --inference_method softmax --use_hypergraph 1 --start_num ${start} --context_emb ${context_emb} \
                              --choose_by_new_type ${choose_new_type} --num_epochs ${num_epoch} > ${log_file} 2>&1
                        done
                    done
                done
            done
        done
    done
fi





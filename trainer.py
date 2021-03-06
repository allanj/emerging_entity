import argparse
import random
import numpy as np
from config import Reader, Config, ContextEmb, lr_decay, simple_batching, evaluate_batch_insts, get_optimizer, write_results, batching_list_instances, build_type_id_mapping
from config import get_metric
import time
from model.neuralcrf import NNCRF
import torch
from typing import List
from common import Instance
from termcolor import colored
import os
from config.utils import load_elmo_vec
import pickle
import tarfile
import shutil
from collections import Counter

def set_seed(opt, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if opt.device.startswith("cuda"):
        print("using GPU...", torch.cuda.current_device())
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--device', type=str, default="cpu", choices=['cpu', 'cuda:0', 'cuda:1', 'cuda:2'],
                        help="GPU/CPU devices")
    parser.add_argument('--seed', type=int, default=42, help="random seed")
    parser.add_argument('--digit2zero', action="store_true", default=True,
                        help="convert the number to 0, make it true is better")
    parser.add_argument('--dataset', type=str, default="conll2003", help="whether use new dataset")
    parser.add_argument('--embedding_file', type=str, default="data/glove.6B.100d.txt",
                        help="we will be using random embeddings if file do not exist")
    parser.add_argument('--embedding_dim', type=int, default=100)
    parser.add_argument('--optimizer', type=str, default="sgd")
    parser.add_argument('--learning_rate', type=float, default=0.01)  ##only for sgd now
    parser.add_argument('--momentum', type=float, default=0.0)
    parser.add_argument('--l2', type=float, default=1e-8)
    parser.add_argument('--lr_decay', type=float, default=0)
    parser.add_argument('--batch_size', type=int, default=10, help="default batch size is 10 (works well)")
    parser.add_argument('--num_epochs', type=int, default=100, help="Usually we set to 10.")
    parser.add_argument('--train_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--dev_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--test_num', type=int, default=-1, help="-1 means all the data")
    parser.add_argument('--start_num', type=int, default=0, help="the size of combinations")

    ##model hyperparameter
    parser.add_argument('--model_folder', type=str, default="english", help="The name to save the model files")
    parser.add_argument('--hidden_dim', type=int, default=200, help="hidden size of the LSTM")
    parser.add_argument('--use_crf_layer', type=int, default=1, help="1 is for using crf layer, 0 for not using CRF layer", choices=[0,1])
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout for embedding")
    parser.add_argument('--use_char_rnn', type=int, default=1, choices=[0, 1], help="use character-level lstm, 0 or 1")
    parser.add_argument('--context_emb', type=str, default="none", choices=["none", "elmo"],
                        help="contextual word embedding")

    parser.add_argument('--add_label_constraint', type=int, default=0, choices=[0, 1], help="Add BIES constraints")
    parser.add_argument('--new_type', type=str, default="MISC", help="The new entity type for zero-shot entity recognition.")
    
    parser.add_argument('--use_neg_labels', type=int, default=0, choices=[0, 1], help="Use finer labels before going to the CRF layer")
    parser.add_argument('--use_boundary', type=int, default=0, choices=[0, 1], help="Use boundary which contains prefix label")
    parser.add_argument('--choose_by_new_type', type=int, default=0, choices=[0, 1], help="Choose best model by the performance on new type entities!")
    parser.add_argument('--inference_method', type=str, default="softmax", choices=["sum", "max", "softmax"], help="Inference method for the latent-variable model!")
    parser.add_argument('--use_hypergraph', type=int, default=0, choices=[0, 1], help="Whether use hypergraph model or not")
    parser.add_argument('--use_fined_labels', type=int, default=1, choices=[0, 1], help="use fined labels or not, this argument determine the latent model")
    parser.add_argument('--heuristic', type=int, default=0, choices=[0, 1],
                        help="use heuristic combinations, using this will disable to ability of the start")
    parser.add_argument('--latent_base', type=int, default=0, choices=[0, 1],
                        help="a baseline with random latent variables.")
    """
    NOTE: if you use end2end, `extraction_model` and `typing_model` should be 0 both.
    """

    """
    Extraction model whether we ignore the type information to train the model.
    """
    parser.add_argument('--extraction_model', type=int, default=0, choices=[0, 1], help="entity extraction model")
    """
    Typing model is given the BIOES segmented sequence, predict the labels direclty.
    NOTE: `extraction_model` and `typing_model` should not be both `1` at the same time.
    """
    parser.add_argument('--typing_model', type=int, default=0, choices=[0,1], help="If use typing model or not, in this case, the input should be regarded as already segmented")
    parser.add_argument('--model_strict', type=str, default="hard", choices=["soft", "hard"], help="If this model is hard, it will follow the extraction result strictly")

    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args


def train_model(config: Config, epoch: int, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance]):
    model = NNCRF(config)
    optimizer = get_optimizer(config, model)
    train_num = len(train_insts)
    print("number of instances: %d" % (train_num))
    print(colored("[Shuffled] Shuffle the training instance ids", "red"))
    random.shuffle(train_insts)

    batched_data = batching_list_instances(config, train_insts)
    dev_batches = batching_list_instances(config, dev_insts)
    test_batches = batching_list_instances(config, test_insts)

    best_dev = [-1, 0]
    best_test = [-1, 0]

    model_folder = config.model_folder
    res_folder = "results"
    if os.path.exists("model_files/" + model_folder):
        raise FileExistsError(f"The folder model_files/{model_folder} exists. Please either delete it or create a new one "
                              f"to avoid override.")
    model_name = "model_files/" + model_folder + "/lstm_crf.m".format()
    config_name = "model_files/" + model_folder + "/config.conf"
    res_name = f"{res_folder}/{model_folder}_test.results"
    dev_name = f"{res_folder}/{model_folder}_dev.results"
    print("[Info] The model will be saved to: %s.tar.gz" % (model_folder))
    if not os.path.exists("model_files"):
        os.makedirs("model_files")
    if not os.path.exists("model_files/" + model_folder):
        os.makedirs("model_files/" + model_folder)
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    for i in range(1, epoch + 1):
        epoch_loss = 0
        start_time = time.time()
        model.zero_grad()
        if config.optimizer.lower() == "sgd":
            optimizer = lr_decay(config, optimizer, i)
        for index in np.random.permutation(len(batched_data)):
            model.train()
            loss = model(*batched_data[index])
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            model.zero_grad()

        end_time = time.time()
        print(colored("[Training] Epoch %d: %.5f, Time is %.2fs" % (i, epoch_loss, end_time - start_time), 'red'), flush=True)

        model.eval()
        dev_metrics = evaluate_model(config, model, dev_batches, "dev", dev_insts)
        print()
        test_metrics = evaluate_model(config, model, test_batches, "test", test_insts)
        if dev_metrics[2] > best_dev[0]:
            print("[Training] saving the best model...")
            best_dev[0] = dev_metrics[2]
            best_dev[1] = i
            best_test[0] = test_metrics[2]
            best_test[1] = i
            torch.save(model.state_dict(), model_name)
            # Save the corresponding config as well.
            f = open(config_name, 'wb')
            pickle.dump(config, f)
            f.close()
            write_results(res_name, test_insts)
            write_results(dev_name, dev_insts)
        model.zero_grad()

    print("[Training] Archiving the best Model...")
    with tarfile.open("model_files/" + model_folder + "/" + model_folder + ".tar.gz", "w:gz") as tar:
        tar.add("model_files/" + model_folder, arcname=os.path.basename(model_folder))

    print("[Training] Finished archiving the models")

    print("[Training] The best dev: %.2f" % (best_dev[0]))
    print("[Training] The corresponding test: %.2f" % (best_test[0]))
    print("[Training] Final testing.")
    model.load_state_dict(torch.load(model_name))
    model.eval()
    evaluate_model(config, model, test_batches, "test", test_insts)
    write_results(res_name, test_insts)
    write_results(dev_name, dev_insts)


def evaluate_model(config: Config, model: NNCRF, batch_insts_ids, name: str, insts: List[Instance]):
    ## evaluation
    p_dict, total_predict_dict, total_entity_dict = Counter(), Counter(), Counter()
    batch_id = 0
    batch_size = config.batch_size
    for batch in batch_insts_ids:
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batch_max_scores, batch_max_ids = model.decode(batch)
        batch_p , batch_predict, batch_total = evaluate_batch_insts(one_batch_insts, batch_max_ids, batch[-1], batch[1], config.idx2labels, config.use_crf_layer)
        p_dict += batch_p
        total_predict_dict += batch_predict
        total_entity_dict += batch_total
        batch_id += 1


    for key in total_entity_dict:
        precision_key, recall_key, fscore_key = get_metric(p_dict[key], total_entity_dict[key], total_predict_dict[key])
        print("[%s] Prec.: %.2f, Rec.: %.2f, F1: %.2f" % (key, precision_key, recall_key, fscore_key))
        if key == config.new_type:
            precision_new_type, recall_new_type, fscore_new_type = get_metric(p_dict[key], total_entity_dict[key], total_predict_dict[key])

    total_p = sum(list(p_dict.values()))
    total_predict = sum(list(total_predict_dict.values()))
    total_entity = sum(list(total_entity_dict.values()))
    precision, recall, fscore = get_metric(total_p, total_entity, total_predict)
    print(colored("[%s set Total] Prec.: %.2f, Rec.: %.2f, F1: %.2f" % (name, precision, recall, fscore), 'blue'), flush=True)
    if config.choose_by_new_type:
        return [precision_new_type, recall_new_type, fscore_new_type]
    else:
        return [precision, recall, fscore]

def main():
    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)
    conf = Config(opt)

    reader = Reader(conf.digit2zero)
    set_seed(opt, conf.seed)

    if "ontonotes" in conf.train_file:
        trains = reader.read_conll(conf.train_file, conf.train_num)
        devs = reader.read_conll(conf.dev_file, conf.dev_num)
        tests = reader.read_conll(conf.test_file, conf.test_num)
    else:
        trains = reader.read_txt(conf.train_file, conf.train_num)
        if conf.typing_model:
            devs = reader.read_txt_with_extraction(conf.dev_file, conf.dev_extraction, conf.dev_num)
            tests = reader.read_txt_with_extraction(conf.test_file, conf.test_extraction, conf.test_num)
        else:
            devs = reader.read_txt(conf.dev_file, conf.dev_num)
            tests = reader.read_txt(conf.test_file, conf.test_num)

    if conf.context_emb != ContextEmb.none:
        print('Loading the ELMo vectors for all datasets.')
        conf.context_emb_size = load_elmo_vec(conf.train_file + "." + conf.context_emb.name + ".vec", trains)
        load_elmo_vec(conf.dev_file + "." + conf.context_emb.name + ".vec", devs)
        load_elmo_vec(conf.test_file + "." + conf.context_emb.name + ".vec", tests)

    conf.use_iobes(trains + devs + tests)
    conf.build_label_idx(trains + devs + tests)

    conf.build_word_idx(trains, devs, tests)
    conf.build_emb_table()

    conf.map_insts_ids(trains + devs + tests)

    if conf.typing_model:
        """
        Building mapping, for example: {B-per: [B-per, B-org, B-misc], O: O, I-org: [I-per, I-org]}
        Will be used when creating the mask
        """
        conf.typing_map = build_type_id_mapping(conf)

    print("num chars: " + str(conf.num_char))
    # print(str(config.char2idx))

    print("num words: " + str(len(conf.word2idx)))
    # print(config.word2idx)
    train_model(conf, conf.num_epochs, trains, devs, tests)


if __name__ == "__main__":
    main()

import numpy as np
import torch
from typing import List, Tuple, Dict, Set
from common import Instance, Span
import pickle
import torch.optim as optim

import torch.nn as nn
import random


from config import PAD, ContextEmb, Config
from termcolor import colored
from collections import defaultdict

def log_sum_exp_pytorch(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize * from_label * to_label].
    :return: [batchSize * to_label]
    """
    maxScores, idx = torch.max(vec, 1)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] ,1 , vec.shape[2]).expand(vec.shape[0], vec.shape[1], vec.shape[2])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 1))

def batching_list_instances(config: Config, insts: List[Instance], is_train:bool = True):
    train_num = len(insts)
    batch_size = config.batch_size
    total_batch = train_num // batch_size + 1 if train_num % batch_size != 0 else train_num // batch_size
    batched_data = []
    for batch_id in range(total_batch):
        one_batch_insts = insts[batch_id * batch_size:(batch_id + 1) * batch_size]
        batched_data.append(simple_batching(config, one_batch_insts, is_train=is_train))

    return batched_data

def build_type_id_mapping(config: Config) -> Dict[int, List[int]]:
    """
    Build the mapping for the typing model. For example:
        ## B- -> B-PER, B-ORG..
        ## I- -> I-PER
    :param config:
    :return: label id to a set of feasible label id.
    """
    # type_id_mapping = defaultdict(list)
    # for label in config.label2idx:
    #     type_id_mapping[config.label2idx[label]] = [config.label2idx[sub_label] for sub_label in config.label2idx if label[:2] == sub_label[:2]]
    ## The above three lines work, but we just want to make it simple as one line.
    if config.hard_model == "hard":
        type_id_mapping = {config.label2idx[label]: [config.label2idx[sub_label] for sub_label in config.label2idx if
                                                 label[:2] == sub_label[:2]] for label in config.label2idx}
    elif config.hard_model == "soft":
        type_id_mapping = {config.label2idx[label]: [config.label2idx[sub_label] for sub_label in config.label2idx if
                                                 ('-' in label and '-' in sub_label) or (label[:2] == sub_label[:2])] for label in config.label2idx}
    return type_id_mapping



def simple_batching(config, insts: List[Instance], is_train: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor]:

    """
    batching these instances together and return tensors. The seq_tensors for word and char contain their word id and char id.
    :return 
        word_seq_tensor: Shape: (batch_size, max_seq_length)
        word_seq_len: Shape: (batch_size), the length of each sentence in a batch.
        context_emb_tensor: Shape: (batch_size, max_seq_length, context_emb_size)
        char_seq_tensor: Shape: (batch_size, max_seq_len, max_char_seq_len)
        char_seq_len: Shape: (batch_size, max_seq_len), 
        label_seq_tensor: Shape: (batch_size, max_seq_length)
    """
    batch_size = len(insts)
    batch_data = insts
    label_size = config.label_size
    # probably no need to sort because we will sort them in the model instead.
    # batch_data = sorted(insts, key=lambda inst: len(inst.input.words), reverse=True) ##object-based not direct copy
    word_seq_len = torch.LongTensor(list(map(lambda inst: len(inst.input.words), batch_data)))
    max_seq_len = word_seq_len.max()

    # NOTE: Use 1 here because the CharBiLSTM accepts
    char_seq_len = torch.LongTensor([list(map(len, inst.input.words)) + [1] * (int(max_seq_len) - len(inst.input.words)) for inst in batch_data])
    max_char_seq_len = char_seq_len.max()

    context_emb_tensor = None
    if config.context_emb != ContextEmb.none:
        emb_size = insts[0].elmo_vec.shape[1]
        context_emb_tensor = torch.zeros((batch_size, max_seq_len, emb_size))

    word_seq_tensor = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    # label_seq_tensor =  torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    label_mask_tensor = torch.zeros((batch_size, max_seq_len, label_size), dtype=torch.long) if is_train else torch.zeros((batch_size, max_seq_len), dtype=torch.long)
    char_seq_tensor = torch.zeros((batch_size, max_seq_len, max_char_seq_len), dtype=torch.long)

    for idx in range(batch_size):
        word_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].word_ids)
        if batch_data[idx].output_ids is not None:

            # label_seq_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)
            if is_train:
                for pos in range(len(batch_data[idx].output_ids)):
                    label_mask_tensor[idx, pos, batch_data[idx].output_ids[pos]] = 1
                    # if batch_data[idx].output_ids[pos] == config.label2idx["O"] and is_train:
                    if ("MISC" in config.idx2labels[batch_data[idx].output_ids[pos]] ) and is_train and config.entity_keep_ratio < 1.0:
                        candidate_prefix = config.idx2labels[batch_data[idx].output_ids[pos]][:2]
                        label_mask_tensor[idx, pos, config.label2idx[candidate_prefix + config.new_type] ] = 1
                        # for label in config.label2idx:
                        #     if config.new_type in label and config.entity_keep_ratio < 1.0:
                        #         label_mask_tensor[idx, pos, config.label2idx[label]] = 1
                    ## To ensure we won't get Nan during training, flip to 0, you will get nan error
                label_mask_tensor[idx, word_seq_len[idx]:, :] = 1
            else:
                label_mask_tensor[idx, :word_seq_len[idx]] = torch.LongTensor(batch_data[idx].output_ids)

        if config.context_emb != ContextEmb.none:
            context_emb_tensor[idx, :word_seq_len[idx], :] = torch.from_numpy(batch_data[idx].elmo_vec)

        for word_idx in range(word_seq_len[idx]):
            char_seq_tensor[idx, word_idx, :char_seq_len[idx, word_idx]] = torch.LongTensor(batch_data[idx].char_ids[word_idx])
        for wordIdx in range(word_seq_len[idx], max_seq_len):
            char_seq_tensor[idx, wordIdx, 0: 1] = torch.LongTensor([config.char2idx[PAD]])   ###because line 119 makes it 1, every single character should have a id. but actually 0 is enough

    word_seq_tensor = word_seq_tensor.to(config.device)
    # label_seq_tensor = label_seq_tensor.to(config.device)
    label_mask_tensor = label_mask_tensor.to(config.device)
    char_seq_tensor = char_seq_tensor.to(config.device)
    word_seq_len = word_seq_len.to(config.device)
    char_seq_len = char_seq_len.to(config.device)

    ## for purpose of typing model below, obtain the typing mask.
    typing_mask = None
    if config.typing_model and batch_data[0].output_extraction != None:
        """
        The mask is to mask out values that are not valid in that position 
        during training and decoding for unlabeled network.
        For example, if we know a position is "B-something", then it must be
        "B-per", "B-loc" or "B-org", etc.
        """
        typing_mask = torch.zeros((batch_size, max_seq_len, config.label_size))
        for idx in range(batch_size):
            for pos in range(word_seq_len[idx]):
                one_valid_label = ""
                for item_label in config.label2idx.keys():
                    if item_label[:2] == batch_data[idx].output_extraction[pos][:2]:
                        one_valid_label = item_label
                        break
                valid_label_idxs = config.typing_map[config.label2idx[one_valid_label]]
                typing_mask[idx, pos, valid_label_idxs] = 1
            typing_mask[idx, word_seq_len[idx]:, :] = 1e-10 ## if we dont't do this, the objective will have NaN issue.
        typing_mask = typing_mask.to(config.device)
    elif config.typing_model:
        typing_mask = torch.zeros((batch_size, max_seq_len, config.label_size))
        for idx in range(batch_size):
            for pos in range(word_seq_len[idx]):
                valid_label_idxs = config.typing_map[batch_data[idx].output_ids[pos]]
                typing_mask[idx, pos, valid_label_idxs] = 1
            typing_mask[idx, word_seq_len[idx]:, :] = 1e-10 ## if we dont't do this, the objective will have NaN issue.
        typing_mask = typing_mask.to(config.device)

    return word_seq_tensor, word_seq_len, context_emb_tensor, char_seq_tensor, char_seq_len, typing_mask, label_mask_tensor


def lr_decay(config, optimizer: optim.Optimizer, epoch: int) -> optim.Optimizer:
    """
    Method to decay the learning rate
    :param config: configuration
    :param optimizer: optimizer
    :param epoch: epoch number
    :return:
    """
    lr = config.learning_rate / (1 + config.lr_decay * (epoch - 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    print('learning rate is set to: ', lr)
    return optimizer


def load_elmo_vec(file: str, insts: List[Instance]):
    """
    Load the elmo vectors and the vector will be saved within each instance with a member `elmo_vec`
    :param file: the vector files for the ELMo vectors
    :param insts: list of instances
    :return:
    """
    f = open(file, 'rb')
    all_vecs = pickle.load(f)  # variables come out in the order you put them in
    f.close()
    size = 0
    for vec, inst in zip(all_vecs, insts):
        inst.elmo_vec = vec
        size = vec.shape[1]
        assert(vec.shape[0] == len(inst.input.words))
    return size



def get_optimizer(config: Config, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "sgd":
        print(
            colored("Using SGD: lr is: {}, L2 regularization is: {}".format(config.learning_rate, config.l2), 'yellow'))
        return optim.SGD(params, lr=config.learning_rate, weight_decay=float(config.l2))
    elif config.optimizer.lower() == "adam":
        print(colored("Using Adam", 'yellow'))
        return optim.Adam(params)
    else:
        print("Illegal optimizer: {}".format(config.optimizer))
        exit(1)



def write_results(filename: str, insts):
    f = open(filename, 'w', encoding='utf-8')
    for inst in insts:
        for i in range(len(inst.input)):
            words = inst.input.words
            output = inst.output
            prediction = inst.prediction
            assert len(output) == len(prediction)
            f.write("{}\t{}\t{}\t{}\n".format(i, words[i], output[i], prediction[i]))
        f.write("\n")
    f.close()


def get_metric(p_num: int, total_num: int, total_predicted_num: int) -> Tuple[float, float, float]:
    """
    Return the metrics of precision, recall and f-score, based on the number
    (We make this small piece of function in order to reduce the code effort and less possible to have typo error)
    :param p_num:
    :param total_num:
    :param total_predicted_num:
    :return:
    """
    precision = p_num * 1.0 / total_predicted_num * 100 if total_predicted_num != 0 else 0
    recall = p_num * 1.0 / total_num * 100 if total_num != 0 else 0
    fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0
    return precision, recall, fscore



def remove_data(trains: List[Instance], conf: Config, type: str, change_to_type:str = "O"):
    print("[Data Info] Removing the entities")
    span_set = remove_entites(trains, conf, type, change_to_type)
    # print(f"entities removed: {span_set}")
    conf.map_insts_ids(trains)
    random.shuffle(trains)
    for inst in trains:
        inst.is_prediction = [False] * len(inst.input)
        for pos, label in enumerate(inst.output):
            if label == conf.O:
                inst.is_prediction[pos] = True


def remove_entites(train_insts: List[Instance], config: Config, type: str, change_to_type: str = "O") -> Set:
    """
    Remove certain number of entities and make them become O label
    :param train_insts:
    :param config:
    :return:
    """
    all_spans = []
    for inst in train_insts:
        output = inst.output
        start = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                start = i
            if output[i].startswith("E-"):
                end = i
                if output[i][2:] == type:
                    all_spans.append(Span(start, end, output[i][2:], inst_id=inst.id))
            if output[i].startswith("S-"):
                if output[i][2:] == type:
                    all_spans.append(Span(i, i, output[i][2:], inst_id=inst.id))
    random.shuffle(all_spans)

    span_set = set()
    num_entity_removed = round(len(all_spans) * (1 - config.entity_keep_ratio))
    for i in range(num_entity_removed):
        span = all_spans[i]
        id = span.inst_id
        output = train_insts[id].output
        if change_to_type == config.O:
            for j in range(span.left, span.right + 1):
                output[j] = change_to_type
        else:
            if span.left == span.right:
                output[span.left] = config.S + change_to_type
            else:
                output[span.left] = config.B  + change_to_type
                output[span.right] = config.E + change_to_type
                for j in range(span.left + 1, span.right):
                    output[j] = config.I +  change_to_type
        span_str = ' '.join(train_insts[id].input.words[span.left:(span.right + 1)])
        span_str = span.type + " " + span_str
        span_set.add(span_str)
    return span_set
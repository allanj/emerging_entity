
import torch
import torch.nn as nn

from typing import List
from config import Config, ContextEmb, START, STOP,PAD
from model.charbilstm import CharBiLSTM
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from overrides import overrides
import numpy as np
from enum import Enum
from itertools import combinations

class IF(Enum):
    sum = 0
    max = 1
    softmax = 2


def lse(vec: torch.Tensor) -> torch.Tensor:
    """
    Calculate the log_sum_exp trick for the tensor.
    :param vec: [batchSize x sent_len x from_label x to_label].
    :return: [batchSize x sent_len * to_label]
    """
    maxScores, idx = torch.max(vec, 2)
    maxScores[maxScores == -float("Inf")] = 0
    maxScoresExpanded = maxScores.view(vec.shape[0] , vec.shape[1],1 , vec.shape[3]).expand(vec.shape[0], vec.shape[1], vec.shape[2], vec.shape[3])
    return maxScores + torch.log(torch.sum(torch.exp(vec - maxScoresExpanded), 2))

class BiLSTMEncoder(nn.Module):

    def __init__(self, config: Config, print_info: bool = True):
        super(BiLSTMEncoder, self).__init__()

        self.label_size = config.label_size
        self.fined_label_size = config.fined_label_size if config.use_fined_labels else 0
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.context_emb = config.context_emb

        self.label2idx = config.label2idx
        self.labels = config.idx2labels

        self.use_fined_labels = config.use_fined_labels
        self.use_end2end = config.use_end2end
        self.fined_label2idx = config.fined_label2idx
        self.fined_labels = config.idx2fined_labels

        self.input_size = config.embedding_dim
        if self.context_emb != ContextEmb.none:
            self.input_size += config.context_emb_size
        if self.use_char:
            self.char_feature = CharBiLSTM(config, print_info=print_info)
            self.input_size += config.charlstm_hidden_dim

        self.word_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(config.word_embedding), freeze=False).to(self.device)
        self.word_drop = nn.Dropout(config.dropout).to(self.device)

        if print_info:
            print("[Model Info] Input size to LSTM: {}".format(self.input_size))
            print("[Model Info] LSTM Hidden Size: {}".format(config.hidden_dim))

        self.lstm = nn.LSTM(self.input_size, config.hidden_dim // 2, num_layers=1, batch_first=True, bidirectional=True).to(self.device)

        self.drop_lstm = nn.Dropout(config.dropout).to(self.device)

        final_hidden_dim = config.hidden_dim

        if print_info:
            print("[Model Info] Final Hidden Size: {}".format(final_hidden_dim))

        tag_size = self.fined_label_size if self.use_fined_labels else self.label_size
        self.hidden2tag = nn.Linear(final_hidden_dim, tag_size).to(self.device)
        if self.use_fined_labels:

            auxilary_labels = set([config.START_TAG, config.STOP_TAG, config.PAD])
            self.num_linear_layer = len(set(label[2:] if len(label) > 2 else label for label in self.label2idx if label not in auxilary_labels))## number of entity labels

            self.fined2labels = nn.Linear(self.fined_label_size, 1 * self.label_size, bias=False).to(self.device)
            label_mapping_weight = self.init_dense_label_mapping_weight()

            # self.filter = nn.Parameter(torch.from_numpy(label_mapping_weight).to(self.device).float(), requires_grad=False)
            # self.inference_method = IF[config.inference_method]

            self.fined2labels.weight.data.copy_(torch.from_numpy(label_mapping_weight))
            self.fined2labels.weight.requires_grad = False  # not updating the weight.
            self.fined2labels.zero_grad()
            ### initialize the weight
            ### add transition constraints for not all labels. (probably in CRF layer)

    def init_label_mapping_weight(self) -> np.ndarray:
        """mapping_weight = np.zeros((self.fined_label_size, self.label_size))
        for fined_label in self.fined_label2idx:
            ##enumerating fined_labels
            if fined_label in self.label2idx:
                mapping_weight[self.fined_label2idx[fined_label], self.label2idx[fined_label]] = 1.0
            else:
                for coarse_idx in self.find_other_coarse_idx(fined_label=fined_label):
                    mapping_weight[self.fined_label2idx[fined_label], coarse_idx] = 1.0
        return mapping_weight"""
        mapping_weight = np.zeros((self.label_size, self.fined_label_size))


        for fined_label in self.fined_label2idx:
            if fined_label in self.label2idx:
                mapping_weight[self.label2idx[fined_label], self.fined_label2idx[fined_label]] = 1.0
            else:
                for coarse_idx in self.find_other_coarse_idx(fined_label=fined_label):
                    mapping_weight[coarse_idx, self.fined_label2idx[fined_label]] = 1.0
        return mapping_weight

    def init_dense_label_mapping_weight(self) -> np.ndarray:
        layers = []
        ## change the start back to one.
        for num in range(self.num_linear_layer, self.num_linear_layer + 1):  ## the number of children in an hyperedge., starting from 1.
            mapping_weight = np.zeros((self.label_size, self.fined_label_size))
            for coarse_label in self.label2idx:
                orig_fined_label_idx = self.fined_label2idx[coarse_label]
                mapping_weight[self.label2idx[coarse_label], orig_fined_label_idx] = 1.0
                valid_indexs = self.find_other_fined_idx(coarse_label=coarse_label)  ## excluding coarse_label itself
                for combination in combinations(valid_indexs, num):
                    mapping_weight[self.label2idx[coarse_label], combination] = 1.0
            layers.append(mapping_weight)
        # return np.concatenate(layers)
        return layers[-1]

    def find_other_fined_idx(self, coarse_label:str) -> List[int]:
        """
        According to the coarse labels in the CRF layer, find a list of valid fined label indexs.
        :param coarse_label:
        :return: a list of valid indexes
        """

        valid_fined_label_idxs = []
        for fined_label in self.fined_label2idx:
            if fined_label.endswith("_NOT"):
                if (coarse_label[:2] == fined_label[:2] and fined_label[:-4] != coarse_label[:-4]) or coarse_label == "O":
                    valid_fined_label_idxs.append(self.fined_label2idx[fined_label])
            else:
                if len(fined_label) == 2 and coarse_label[:2] == fined_label[:2]:
                    assert (len(fined_label) == 2 and '-' in fined_label)
                    valid_fined_label_idxs.append(self.fined_label2idx[fined_label])
        return valid_fined_label_idxs


    def find_other_coarse_idx(self, fined_label:str):
        """
        Acoording to the fined labels right after LSTM layer and find a list of valid coarse labels.
        :param fined_label:
        :return:
        """
        ## fined label must be "_NOT"
        assert  fined_label.endswith("_NOT") or (len(fined_label) == 2 and '-' in fined_label)
        for coarse_label in self.label2idx:
            if fined_label.endswith("_NOT"):
                if (coarse_label[:2] == fined_label[:2] and fined_label[:-4] != coarse_label) or coarse_label == "O":
                    yield self.label2idx[coarse_label]
            elif self.use_end2end:
                if coarse_label[:2] == fined_label:
                    yield self.label2idx[coarse_label]

    @overrides
    def forward(self, word_seq_tensor: torch.Tensor,
                       word_seq_lens: torch.Tensor,
                       batch_context_emb: torch.Tensor,
                       char_inputs: torch.Tensor,
                       char_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Encoding the input with BiLSTM
        :param word_seq_tensor: (batch_size, sent_len)   NOTE: The word seq actually is already ordered before come here.
        :param word_seq_lens: (batch_size, 1)
        :param batch_context_emb: (batch_size, sent_len, context embedding) ELMo embedings
        :param char_inputs: (batch_size * sent_len * word_length)
        :param char_seq_lens: numpy (batch_size * sent_len , 1)
        :return: emission scores (batch_size, sent_len, hidden_dim)
        """

        word_emb = self.word_embedding(word_seq_tensor)
        if self.context_emb != ContextEmb.none:
            word_emb = torch.cat((word_emb, batch_context_emb.to(self.device)), 2)
        if self.use_char:
            char_features = self.char_feature(char_inputs, char_seq_lens)
            word_emb = torch.cat((word_emb, char_features), 2)

        word_rep = self.word_drop(word_emb)


        sorted_seq_len, permIdx = word_seq_lens.sort(0, descending=True)
        _, recover_idx = permIdx.sort(0, descending=False)
        sorted_seq_tensor = word_rep[permIdx]

        packed_words = pack_padded_sequence(sorted_seq_tensor, sorted_seq_len, True)
        lstm_out, _ = self.lstm(packed_words, None)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)  ## CARE: make sure here is batch_first, otherwise need to transpose.
        feature_out = self.drop_lstm(lstm_out)

        outputs = self.hidden2tag(feature_out)
        if self.use_fined_labels:
            outputs = self.fined2labels(outputs)
            #batch_size, sent_len, num_fined_labels = outputs.size()
            #outputs = outputs.view(batch_size, sent_len, num_fined_labels, 1).expand(batch_size, sent_len, num_fined_labels, self.label_size)
            #outputs = outputs * self.filter
            #if self.inference_method == IF.max:
                #outputs, _ = outputs.max(dim=-2)
            #elif self.inference_method == IF.sum:
                #outputs = outputs.sum(dim=-2)
            #else:
                #outputs = lse(outputs)

            ## in the new model, outputs is with size: batch_size x sent_len x ( coarse_label x #linear layer)
            # batch_size, sent_len, num_all = outputs.size()
            # outputs = outputs.view(batch_size, sent_len, 1, num_all).view(batch_size, sent_len, -1, self.label_size)
            # if self.inference_method == IF.max:
            #     outputs, _ = outputs.max(dim=-2)
            # elif self.inference_method == IF.sum:
            #     outputs = outputs.sum(dim=-2)
            # else:
            #     outputs = lse(outputs)

        return outputs[recover_idx]



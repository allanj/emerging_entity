# 
# @author: Allan
#

import numpy as np
from tqdm import tqdm
from typing import List, Tuple, Dict, Union
from common import Instance
import torch
from enum import Enum
import os

from termcolor import colored


START = "<START>"
STOP = "<END>"
PAD = "<PAD>"


class ContextEmb(Enum):
    none = 0
    elmo = 1
    bert = 2 # not support yet
    flair = 3 # not support yet


class Config:
    def __init__(self, args) -> None:
        """
        Construct the arguments and some hyperparameters
        :param args:
        """

        # Predefined label string.
        self.PAD = PAD
        self.B = "B-"
        self.I = "I-"
        self.S = "S-"
        self.E = "E-"
        self.O = "O"
        self.START_TAG = START
        self.STOP_TAG = STOP
        self.UNK = "<UNK>"
        self.unk_id = -1
        self.general = "general"

        # Model hyper parameters
        self.embedding_file = args.embedding_file
        self.embedding_dim = args.embedding_dim
        self.context_emb = ContextEmb[args.context_emb]
        self.context_emb_size = 0
        self.embedding, self.embedding_dim = self.read_pretrain_embedding()
        self.word_embedding = None
        self.seed = args.seed
        self.digit2zero = args.digit2zero
        self.hidden_dim = args.hidden_dim
        self.use_brnn = True
        self.num_layers = 1
        self.dropout = args.dropout
        self.char_emb_size = 25
        self.charlstm_hidden_dim = 50
        self.use_char_rnn = args.use_char_rnn
        self.use_crf_layer = args.use_crf_layer

        self.use_fined_labels = args.use_fined_labels
        self.add_label_constraint = args.add_label_constraint
        self.use_boundary = args.use_boundary
        self.use_neg_labels = args.use_neg_labels
        self.new_type = args.new_type
        self.choose_by_new_type = args.choose_by_new_type
        self.typing_model = args.typing_model
        self.hard_model = args.model_strict
        self.extraction_model = args.extraction_model
        self.inference_method = args.inference_method
        self.use_hypergraph = args.use_hypergraph
        self.heuristic = args.heuristic

        self.latent_base = args.latent_base
        self.latent_labels: List[str] = ["Latent_1", "Latent_2", "Latent_3", "Latent_4"]

        self.dataset = args.dataset
        self.dev_extraction = "results/extraction_dev.results"
        self.test_extraction = "results/extraction_test.results"
        self.train_extraction = "results/extraction_train.results"
        self.train_file = "data/" + self.dataset + "/train.txt"
        self.dev_file = "data/" + self.dataset + "/dev.txt"
        self.test_file = "data/" + self.dataset + "/test.txt"
        self.label2idx = {}
        self.idx2labels = []
        self.fined_label2idx = {}
        self.idx2fined_labels = []
        self.char2idx = {}
        self.idx2char = []
        self.num_char = 0
        self.train_num = args.train_num
        self.dev_num = args.dev_num
        self.test_num = args.test_num
        self.start_num = args.start_num

        # Training hyperparameter
        self.model_folder = args.model_folder
        self.optimizer = args.optimizer.lower()
        self.learning_rate = args.learning_rate
        self.momentum = args.momentum
        self.l2 = args.l2
        self.num_epochs = args.num_epochs
        self.use_dev = True
        self.batch_size = args.batch_size
        self.clip = 5
        self.lr_decay = args.lr_decay
        self.device = torch.device(args.device)

        self.entity_keep_ratio = args.entity_keep_ratio

    def read_pretrain_embedding(self) -> Tuple[Union[Dict[str, np.array], None], int]:
        """
        Read the pretrained word embeddings, return the complete embeddings and the embedding dimension
        :return:
        """
        print("reading the pretraing embedding: %s" % (self.embedding_file))
        if self.embedding_file is None:
            print("pretrain embedding in None, using random embedding")
            return None, self.embedding_dim
        else:
            exists = os.path.isfile(self.embedding_file)
            if not exists:
                print(colored("[Warning] pretrain embedding file not exists, using random embedding",  'red'))
                return None, self.embedding_dim
                # raise FileNotFoundError("The embedding file does not exists")
        embedding_dim = -1
        embedding = dict()
        with open(self.embedding_file, 'r', encoding='utf-8') as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                if len(line) == 0:
                    continue
                tokens = line.split()
                if len(tokens) == 2:
                    continue
                if embedding_dim < 0:
                    embedding_dim = len(tokens) - 1
                else:
                    # print(tokens)
                    # print(embedding_dim)
                    # assert (embedding_dim + 1 == len(tokens))
                    if (embedding_dim + 1) != len(tokens):
                        continue
                embedd = np.empty([1, embedding_dim])
                embedd[:] = tokens[1:]
                first_col = tokens[0]
                embedding[first_col] = embedd
        return embedding, embedding_dim

    def build_word_idx(self, train_insts: List[Instance], dev_insts: List[Instance], test_insts: List[Instance]) -> None:
        """
        Build the vocab 2 idx for all instances
        :param train_insts:
        :param dev_insts:
        :param test_insts:
        :return:
        """
        self.word2idx = dict()
        self.idx2word = []
        self.word2idx[self.PAD] = 0
        self.idx2word.append(self.PAD)
        self.word2idx[self.UNK] = 1
        self.unk_id = 1
        self.idx2word.append(self.UNK)

        self.char2idx[self.PAD] = 0
        self.idx2char.append(self.PAD)
        self.char2idx[self.UNK] = 1
        self.idx2char.append(self.UNK)

        # extract char on train, dev, test
        for inst in train_insts + dev_insts + test_insts:
            for word in inst.input.words:
                if word not in self.word2idx:
                    self.word2idx[word] = len(self.word2idx)
                    self.idx2word.append(word)
        # extract char only on train (doesn't matter for dev and test)
        for inst in train_insts:
            for word in inst.input.words:
                for c in word:
                    if c not in self.char2idx:
                        self.char2idx[c] = len(self.idx2char)
                        self.idx2char.append(c)
        self.num_char = len(self.idx2char)

    def build_emb_table(self) -> None:
        """
        build the embedding table with pretrained word embeddings (if given otherwise, use random embeddings)
        :return:
        """
        print("Building the embedding table for vocabulary...")
        scale = np.sqrt(3.0 / self.embedding_dim)
        if self.embedding is not None:
            print("[Info] Use the pretrained word embedding to initialize: %d x %d" % (len(self.word2idx), self.embedding_dim))
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                if word in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word]
                elif word.lower() in self.embedding:
                    self.word_embedding[self.word2idx[word], :] = self.embedding[word.lower()]
                else:
                    # self.word_embedding[self.word2idx[word], :] = self.embedding[self.UNK]
                    self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])
            self.embedding = None
        else:
            self.word_embedding = np.empty([len(self.word2idx), self.embedding_dim])
            for word in self.word2idx:
                self.word_embedding[self.word2idx[word], :] = np.random.uniform(-scale, scale, [1, self.embedding_dim])

    def build_label_idx(self, insts: List[Instance]) -> None:
        """
        Build the mapping from label to index and index to labels.
        :param insts: list of instances.
        :return:
        """
        self.label2idx[self.PAD] = len(self.label2idx)
        self.idx2labels.append(self.PAD)
        if self.use_fined_labels or self.latent_base:
            self.fined_label2idx[self.PAD] = len(self.fined_label2idx)
            self.idx2fined_labels.append(self.PAD)
        for inst in insts:
            for label in inst.output:
                if self.extraction_model and label != self.O:
                    label = label[:2] + self.general
                if label not in self.label2idx:
                    self.idx2labels.append(label)
                    self.label2idx[label] = len(self.label2idx)

                if self.use_fined_labels and label not in self.fined_label2idx:
                    self.idx2fined_labels.append(label)
                    self.fined_label2idx[label] = len(self.fined_label2idx)
                    if label != "O" and self.use_neg_labels: ##B-per, B-ORG
                        negative_label = label + "_NOT"
                        self.idx2fined_labels.append(negative_label)
                        self.fined_label2idx[negative_label] = len(self.fined_label2idx)
                    if label != "O" and self.use_boundary:
                        prefix_label = label[:2]
                        if prefix_label not in self.fined_label2idx:
                            self.idx2fined_labels.append(prefix_label)
                            self.fined_label2idx[prefix_label] = len(self.fined_label2idx)

                if self.latent_base and label not in self.fined_label2idx:
                    self.idx2fined_labels.append(label)
                    self.fined_label2idx[label] = len(self.fined_label2idx)

        if self.use_fined_labels:
            if self.B + self.new_type not in self.label2idx:
                self.label2idx[self.B + self.new_type] = len(self.label2idx)
                self.idx2labels.append(self.B + self.new_type)
            if self.I + self.new_type not in self.label2idx:
                self.label2idx[self.I + self.new_type] = len(self.label2idx)
                self.idx2labels.append(self.I + self.new_type)
            if self.E + self.new_type not in self.label2idx:
                self.label2idx[self.E + self.new_type] = len(self.label2idx)
                self.idx2labels.append(self.E + self.new_type)
            if self.S+ self.new_type not in self.label2idx:
                self.label2idx[self.S + self.new_type] = len(self.label2idx)
                self.idx2labels.append(self.S + self.new_type)

        if self.latent_base:
            for latent_label in self.latent_labels: ## add L1, L2, L3, L4
                if latent_label not in self.fined_label2idx:
                    self.idx2fined_labels.append(latent_label)
                    self.fined_label2idx[latent_label] = len(self.fined_label2idx)



        self.label2idx[self.START_TAG] = len(self.label2idx)
        self.idx2labels.append(self.START_TAG)
        self.label2idx[self.STOP_TAG] = len(self.label2idx)
        self.idx2labels.append(self.STOP_TAG)
        self.label_size = len(self.label2idx)
        print("#labels: {}".format(self.label_size))
        print("label 2idx: {}".format(self.label2idx))

        if self.use_fined_labels or self.latent_base:
            self.fined_label2idx[self.START_TAG] = len(self.fined_label2idx)
            self.idx2fined_labels.append(self.START_TAG)
            self.fined_label2idx[self.STOP_TAG] = len(self.fined_label2idx)
            self.idx2fined_labels.append(self.STOP_TAG)
            self.fined_label_size = len(self.fined_label2idx)
            print("#fined labels: {}".format(self.fined_label_size))
            print("fined label 2idx: {}".format(self.fined_label2idx))


    def use_iobes(self, insts: List[Instance]) -> None:
        """
        Use IOBES tagging schema to replace the IOB tagging schema in the instance
        :param insts:
        :return:
        """
        for inst in insts:
            output = inst.output
            for pos in range(len(inst)):
                curr_entity = output[pos]
                if pos == len(inst) - 1:
                    if curr_entity.startswith(self.B):
                        output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        output[pos] = curr_entity.replace(self.I, self.E)
                else:
                    next_entity = output[pos + 1]
                    if curr_entity.startswith(self.B):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.B, self.S)
                    elif curr_entity.startswith(self.I):
                        if next_entity.startswith(self.O) or next_entity.startswith(self.B):
                            output[pos] = curr_entity.replace(self.I, self.E)

    def map_insts_ids(self, insts: List[Instance]):
        """
        Create id for word, char and label in each instance.
        :param insts:
        :return:
        """
        for inst in insts:
            words = inst.input.words
            inst.word_ids = []
            inst.char_ids = []
            inst.output_ids = [] if inst.output else None
            for word in words:
                if word in self.word2idx:
                    inst.word_ids.append(self.word2idx[word])
                else:
                    inst.word_ids.append(self.word2idx[self.UNK])
                char_id = []
                for c in word:
                    if c in self.char2idx:
                        char_id.append(self.char2idx[c])
                    else:
                        char_id.append(self.char2idx[self.UNK])
                inst.char_ids.append(char_id)
            if inst.output:
                for label in inst.output:
                    if self.extraction_model and label != self.O:
                        label = label[:2] + self.general
                    inst.output_ids.append(self.label2idx[label])

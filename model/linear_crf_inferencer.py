

import torch.nn as nn
import torch

from config import log_sum_exp_pytorch, START, STOP, PAD
from typing import Tuple
from overrides import overrides

class LinearCRF(nn.Module):


    def __init__(self, config, print_info: bool = True):
        super(LinearCRF, self).__init__()

        self.label_size = config.label_size
        self.device = config.device
        self.use_char = config.use_char_rnn
        self.context_emb = config.context_emb

        self.label2idx = config.label2idx
        self.labels = config.idx2labels
        self.start_idx = self.label2idx[START]
        self.end_idx = self.label2idx[STOP]
        self.pad_idx = self.label2idx[PAD]

        # initialize the following transition (anything never -> start. end never -> anything. Same thing for the padding label)
        init_transition = torch.randn(self.label_size, self.label_size).to(self.device)
        init_transition[:, self.start_idx] = -10000.0
        init_transition[self.end_idx, :] = -10000.0
        init_transition[:, self.pad_idx] = -10000.0
        init_transition[self.pad_idx, :] = -10000.0

        if config.add_label_constraint:
            ## adding the label constraints.
            self.add_contraints(transition=init_transition)

        self.transition = nn.Parameter(init_transition)


    def add_contraints(self, transition: torch.Tensor) -> None:
        for l1 in range(self.label_size):
            previous_label = self.labels[l1]
            if l1 == self.start_idx or l1 == self.end_idx or l1 == self.pad_idx:
                continue
            for l2 in range(self.label_size):
                if l2 == self.start_idx or l2 == self.end_idx or l2 == self.pad_idx:
                    continue
                next_label = self.labels[l2]
                if previous_label.startswith("O") and (next_label.startswith("I-") or next_label.startswith("E-")):
                    transition[l1, l2] = -10000.0
                if previous_label.startswith("B-") or previous_label.startswith("I-"):
                    if next_label.startswith("B-") or next_label.startswith("S-") or next_label.startswith(
                            "O") or next_label[2:] != previous_label[2:]:
                        transition[l1, l2] = -10000.0
                if previous_label.startswith("S-") or previous_label.startswith("E-"):
                    if next_label.startswith("I-") or next_label.startswith("E-"):
                        transition[l1, l2] = -10000.0

    @overrides
    def forward(self, lstm_scores, word_seq_lens, tags, mask):
        """
        Calculate the negative log-likelihood
        :param lstm_scores:
        :param word_seq_lens:
        :param tags:
        :param mask:
        :return:
        """
        all_scores=  self.calculate_all_scores(lstm_scores= lstm_scores)
        unlabed_score = self.forward_unlabeled(all_scores, word_seq_lens)
        labeled_score = self.forward_labeled(all_scores, word_seq_lens, tags, mask)
        return unlabed_score, labeled_score

    def forward_unlabeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> torch.Tensor:
        """
        Calculate the scores with the forward algorithm. Basically calculating the normalization term
        :param all_scores: (batch_size x max_seq_len x num_labels x num_labels) from (lstm scores + transition scores).
        :param word_seq_lens: (batch_size)
        :return: (batch_size) for the normalization scores
        """
        batch_size = all_scores.size(0)
        seq_len = all_scores.size(1)
        alpha = torch.zeros(batch_size, seq_len, self.label_size).to(self.device)

        alpha[:, 0, :] = all_scores[:, 0,  self.start_idx, :] ## the first position of all labels = (the transition from start - > all labels) + current emission.

        for word_idx in range(1, seq_len):
            ## batch_size, self.label_size, self.label_size
            before_log_sum_exp = alpha[:, word_idx-1, :].view(batch_size, self.label_size, 1).expand(batch_size, self.label_size, self.label_size) + all_scores[:, word_idx, :, :]
            alpha[:, word_idx, :] = log_sum_exp_pytorch(before_log_sum_exp)

        ### batch_size x label_size
        last_alpha = torch.gather(alpha, 1, word_seq_lens.view(batch_size, 1, 1).expand(batch_size, 1, self.label_size)-1).view(batch_size, self.label_size)
        last_alpha += self.transition[:, self.end_idx].view(1, self.label_size).expand(batch_size, self.label_size)
        last_alpha = log_sum_exp_pytorch(last_alpha.view(batch_size, self.label_size, 1)).view(batch_size)

        return torch.sum(last_alpha)

    def forward_labeled(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor, tags: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        '''
        Calculate the scores for the gold instances.
        :param all_scores: (batch, seq_len, label_size, label_size)
        :param word_seq_lens: (batch, seq_len)
        :param tags: (batch, seq_len)
        :param masks: batch, seq_len
        :return: sum of score for the gold sequences Shape: (batch_size)
        '''
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]

        ## all the scores to current labels: batch, seq_len, all_from_label?
        currentTagScores = torch.gather(all_scores, 3, tags.view(batchSize, sentLength, 1, 1).expand(batchSize, sentLength, self.label_size, 1)).view(batchSize, -1, self.label_size)
        if sentLength != 1:
            tagTransScoresMiddle = torch.gather(currentTagScores[:, 1:, :], 2, tags[:, : sentLength - 1].view(batchSize, sentLength - 1, 1)).view(batchSize, -1)
        tagTransScoresBegin = currentTagScores[:, 0, self.start_idx]
        endTagIds = torch.gather(tags, 1, word_seq_lens.view(batchSize, 1) - 1)
        tagTransScoresEnd = torch.gather(self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size), 1,  endTagIds).view(batchSize)
        score = torch.sum(tagTransScoresBegin) + torch.sum(tagTransScoresEnd)
        if sentLength != 1:
            score += torch.sum(tagTransScoresMiddle.masked_select(masks[:, 1:]))
        return score

    def calculate_all_scores(self, lstm_scores: torch.Tensor) -> torch.Tensor:
        """
        Calculate all scores by adding up the transition scores and emissions (from lstm).
        Basically, compute the scores for each edges between labels at adjacent positions.
        This score is later be used for forward-backward inference
        :param lstm_scores: emission scores.
        :return:
        """
        batch_size = lstm_scores.size(0)
        seq_len = lstm_scores.size(1)
        scores = self.transition.view(1, 1, self.label_size, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size) + \
                 lstm_scores.view(batch_size, seq_len, 1, self.label_size).expand(batch_size, seq_len, self.label_size, self.label_size)
        return scores

    def decode(self, features, wordSeqLengths) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decode the batch input
        :param batchInput:
        :return:
        """
        all_scores = self.calculate_all_scores(features)
        bestScores, decodeIdx = self.viterbi_decode(all_scores, wordSeqLengths)
        return bestScores, decodeIdx

    def viterbi_decode(self, all_scores: torch.Tensor, word_seq_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Use viterbi to decode the instances given the scores and transition parameters
        :param all_scores: (batch_size x max_seq_len x num_labels)
        :param word_seq_lens: (batch_size)
        :return: the best scores as well as the predicted label ids.
               (batch_size) and (batch_size x max_seq_len)
        """
        batchSize = all_scores.shape[0]
        sentLength = all_scores.shape[1]
        # sent_len =
        scoresRecord = torch.zeros([batchSize, sentLength, self.label_size]).to(self.device)
        idxRecord = torch.zeros([batchSize, sentLength, self.label_size], dtype=torch.int64).to(self.device)
        mask = torch.ones_like(word_seq_lens, dtype=torch.int64).to(self.device)
        startIds = torch.full((batchSize, self.label_size), self.start_idx, dtype=torch.int64).to(self.device)
        decodeIdx = torch.LongTensor(batchSize, sentLength).to(self.device)

        scores = all_scores
        # scoresRecord[:, 0, :] = self.getInitAlphaWithBatchSize(batchSize).view(batchSize, self.label_size)
        scoresRecord[:, 0, :] = scores[:, 0, self.start_idx, :]  ## represent the best current score from the start, is the best
        idxRecord[:,  0, :] = startIds
        for wordIdx in range(1, sentLength):
            ### scoresIdx: batch x from_label x to_label at current index.
            scoresIdx = scoresRecord[:, wordIdx - 1, :].view(batchSize, self.label_size, 1).expand(batchSize, self.label_size,
                                                                                  self.label_size) + scores[:, wordIdx, :, :]
            idxRecord[:, wordIdx, :] = torch.argmax(scoresIdx, 1)  ## the best previous label idx to crrent labels
            scoresRecord[:, wordIdx, :] = torch.gather(scoresIdx, 1, idxRecord[:, wordIdx, :].view(batchSize, 1, self.label_size)).view(batchSize, self.label_size)

        lastScores = torch.gather(scoresRecord, 1, word_seq_lens.view(batchSize, 1, 1).expand(batchSize, 1, self.label_size) - 1).view(batchSize, self.label_size)  ##select position
        lastScores += self.transition[:, self.end_idx].view(1, self.label_size).expand(batchSize, self.label_size)
        decodeIdx[:, 0] = torch.argmax(lastScores, 1)
        bestScores = torch.gather(lastScores, 1, decodeIdx[:, 0].view(batchSize, 1))

        for distance2Last in range(sentLength - 1):
            lastNIdxRecord = torch.gather(idxRecord, 1, torch.where(word_seq_lens - distance2Last - 1 > 0, word_seq_lens - distance2Last - 1, mask).view(batchSize, 1, 1).expand(batchSize, 1, self.label_size)).view(batchSize, self.label_size)
            decodeIdx[:, distance2Last + 1] = torch.gather(lastNIdxRecord, 1, decodeIdx[:, distance2Last].view(batchSize, 1)).view(batchSize)

        return bestScores, decodeIdx
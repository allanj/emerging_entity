
import numpy as np
from overrides import overrides
from typing import List
from common import Instance
import torch


class Span:
    """
    A class of `Span` where we use it during evaluation.
    We construct spans for the convenience of evaluation.
    """
    def __init__(self, left: int, right: int, type: str):
        """
        A span compose of left, right (inclusive) and its entity label.
        :param left:
        :param right: inclusive.
        :param type:
        """
        self.left = left
        self.right = right
        self.type = type

    def __eq__(self, other):
        return self.left == other.left and self.right == other.right and self.type == other.type

    def __hash__(self):
        return hash((self.left, self.right, self.type))


def evaluate_batch_insts(batch_insts: List[Instance],
                         batch_pred_ids: torch.LongTensor,
                         batch_gold_ids: torch.LongTensor,
                         word_seq_lens: torch.LongTensor,
                         idx2label: List[str],
                         use_crf_layer: bool = True) -> np.ndarray:
    """
    Evaluate a batch of instances and handling the padding positions.
    :param batch_insts:  a batched of instances.
    :param batch_pred_ids: Shape: (batch_size, max_length) prediction ids from the viterbi algorithm.
    :param batch_gold_ids: Shape: (batch_size, max_length) gold ids.
    :param word_seq_lens: Shape: (batch_size) the length for each instance.
    :param idx2label: The idx to label mapping.
    :return: numpy array containing (number of true positive, number of all positive, number of true positive + number of false negative)
             You can also refer as (number of correctly predicted entities, number of entities predicted, number of entities in the dataset)
    """
    p = 0
    total_entity = 0
    total_predict = 0
    p_per = 0
    total_entity_per = 0
    total_predict_per = 0
    p_loc = 0
    total_entity_loc = 0
    total_predict_loc = 0
    p_org = 0
    total_entity_org = 0
    total_predict_org = 0
    p_misc = 0
    total_entity_misc = 0
    total_predict_misc = 0

    word_seq_lens = word_seq_lens.tolist()
    for idx in range(len(batch_pred_ids)):
        length = word_seq_lens[idx]
        output = batch_gold_ids[idx][:length].tolist()
        prediction = batch_pred_ids[idx][:length].tolist()
        prediction = prediction[::-1] if use_crf_layer else prediction
        output = [idx2label[l] for l in output]
        prediction =[idx2label[l] for l in prediction]
        batch_insts[idx].prediction = prediction
        #convert to span
        output_spans = set()
        output_spans_per = set()
        output_spans_loc = set()
        output_spans_org = set()
        output_spans_misc = set()
        start = -1
        start_per = -1
        start_loc = -1
        start_org = -1
        start_misc = -1
        for i in range(len(output)):
            if output[i].startswith("B-"):
                if output[i].endswith("PER"):
                    start_per = i
                if output[i].endswith("LOC"):
                    start_loc = i
                if output[i].endswith("ORG"):
                    start_org = i
                if output[i].endswith("MISC"):
                    start_misc = i
                start = i
            if output[i].startswith("E-"):
                if output[i].endswith("PER"):
                    end_per = i
                    output_spans_per.add(Span(start_per, end_per, output[i][2:]))
                if output[i].endswith("LOC"):
                    end_loc = i
                    output_spans_loc.add(Span(start_loc, end_loc, output[i][2:]))
                if output[i].endswith("ORG"):
                    end_org = i
                    output_spans_org.add(Span(start_org, end_org, output[i][2:]))
                if output[i].endswith("MISC"):
                    end_misc = i
                    output_spans_misc.add(Span(start_misc, end_misc, output[i][2:]))
                end = i
                output_spans.add(Span(start, end, output[i][2:]))
            if output[i].startswith("S-"):
                if output[i].endswith("PER"):
                    output_spans_per.add(Span(i, i, output[i][2:]))
                if output[i].endswith("LOC"):
                    output_spans_loc.add(Span(i, i, output[i][2:]))
                if output[i].endswith("ORG"):
                    output_spans_org.add(Span(i, i, output[i][2:]))
                if output[i].endswith("MISC"):
                    output_spans_misc.add(Span(i, i, output[i][2:]))
                output_spans.add(Span(i, i, output[i][2:]))
        predict_spans = set()
        predict_spans_per = set()
        predict_spans_loc = set()
        predict_spans_org = set()
        predict_spans_misc = set()
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                if prediction[i].endswith("PER"):
                    start_per = i
                if prediction[i].endswith("LOC"):
                    start_loc = i
                if prediction[i].endswith("ORG"):
                    start_org = i
                if prediction[i].endswith("MISC"):
                    start_misc = i
                start = i
            if prediction[i].startswith("E-"):
                if prediction[i].endswith("PER"):
                    end_per = i
                    predict_spans_per.add(Span(start_per, end_per, prediction[i][2:]))
                if prediction[i].endswith("LOC"):
                    end_loc = i
                    predict_spans_loc.add(Span(start_loc, end_loc, prediction[i][2:]))
                if prediction[i].endswith("ORG"):
                    end_org = i
                    predict_spans_org.add(Span(start_org, end_org, prediction[i][2:]))
                if prediction[i].endswith("MISC"):
                    end_misc = i
                    predict_spans_misc.add(Span(start_misc, end_misc, prediction[i][2:]))
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
            if prediction[i].startswith("S-"):
                if prediction[i].endswith("PER"):
                    predict_spans_per.add(Span(i, i, prediction[i][2:]))
                if prediction[i].endswith("LOC"):
                    predict_spans_loc.add(Span(i, i, prediction[i][2:]))
                if prediction[i].endswith("ORG"):
                    predict_spans_org.add(Span(i, i, prediction[i][2:]))
                if prediction[i].endswith("MISC"):
                    predict_spans_misc.add(Span(i, i, prediction[i][2:]))
                predict_spans.add(Span(i, i, prediction[i][2:]))

        total_entity += len(output_spans)
        total_predict += len(predict_spans)
        p += len(predict_spans.intersection(output_spans))

        total_entity_per += len(output_spans_per)
        total_predict_per += len(predict_spans_per)
        p_per += len(predict_spans_per.intersection(output_spans_per))

        total_entity_loc += len(output_spans_loc)
        total_predict_loc += len(predict_spans_loc)
        p_loc += len(predict_spans_loc.intersection(output_spans_loc))

        total_entity_org += len(output_spans_org)
        total_predict_org += len(predict_spans_org)
        p_org += len(predict_spans_org.intersection(output_spans_org))
        
        total_entity_misc += len(output_spans_misc)
        total_predict_misc += len(predict_spans_misc)
        p_misc += len(predict_spans_misc.intersection(output_spans_misc))

    # In case you need the following code for calculating the p/r/f in a batch.
    # (When your batch is the complete dataset)
    # precision = p * 1.0 / total_predict * 100 if total_predict != 0 else 0
    # recall = p * 1.0 / total_entity * 100 if total_entity != 0 else 0
    # fscore = 2.0 * precision * recall / (precision + recall) if precision != 0 or recall != 0 else 0

    return np.asarray([p, total_predict, total_entity, p_per, total_predict_per, total_entity_per, \
                    p_loc, total_predict_loc, total_entity_loc, p_org, total_predict_org, total_entity_org, \
                    p_misc, total_predict_misc, total_entity_misc], dtype=int)

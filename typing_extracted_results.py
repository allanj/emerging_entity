from typing import List
from common import Instance, Sentence
from tqdm import tqdm
import re
from ner_predictor import NERPredictor
from config.eval import Span
from collections import defaultdict
from config import START, STOP, PAD
from config.utils import write_results, get_metric

import argparse

"""
This file contains the code for using the typing model to make prediction 
on the predicted results from the extraction model

"""


def read_extraction_results(file: str, number: int = -1, digit2zero: bool = True) -> List[Instance]:
    print("Reading file: " + file)
    insts = []
    with open(file, 'r', encoding='utf-8') as f:
        words = []
        labels = []
        ground_truth = []
        for line in tqdm(f.readlines()):
            line = line.rstrip()
            if line == "":
                inst = Instance(Sentence(words), labels)
                inst.ground_truth = ground_truth
                insts.append(inst)
                words = []
                labels = []
                ground_truth = []
                if len(insts) == number:
                    break
                continue
            _, word, gold_label, predicted_segment_label = line.split()
            if digit2zero:
                word = re.sub('\d', '0', word)  # replace digit with 0.
            words.append(word)
            labels.append(predicted_segment_label)
            ground_truth.append(gold_label)
    print("number of sentences: {}".format(len(insts)))
    return insts


def evaluate(insts: List[Instance], predictions: List[List[str]], print_details: bool = False) -> None:
    """
        Evaluation
    """
    p_dict = defaultdict(int)
    total_entity_dict = defaultdict(int)
    total_predict_dict = defaultdict(int)
    for i, inst in enumerate(insts):
        ground_truth = inst.ground_truth
        prediction = predictions[i]
        inst.prediction = prediction
        output_spans = set()
        start = -1
        for i in range(len(ground_truth)):
            if ground_truth[i].startswith("B-"):
                start = i
            if ground_truth[i].startswith("E-"):
                end = i
                output_spans.add(Span(start, end, ground_truth[i][2:]))
                total_entity_dict[ground_truth[i][2:]] += 1
            if ground_truth[i].startswith("S-"):
                output_spans.add(Span(i, i, ground_truth[i][2:]))
                total_entity_dict[ground_truth[i][2:]] += 1
        predict_spans = set()
        start = -1
        for i in range(len(prediction)):
            if prediction[i].startswith("B-"):
                start = i
            if prediction[i].startswith("E-"):
                end = i
                predict_spans.add(Span(start, end, prediction[i][2:]))
                total_predict_dict[prediction[i][2:]] += 1
            if prediction[i].startswith("S-"):
                predict_spans.add(Span(i, i, prediction[i][2:]))
                total_predict_dict[prediction[i][2:]] += 1

        correct_spans = predict_spans.intersection(output_spans)
        for span in correct_spans:
            p_dict[span.type] += 1

    if print_details:
        for key in total_entity_dict:
            precision, recall, fscore = get_metric(p_num=p_dict[key], total_predicted_num=total_predict_dict[key], total_num=total_entity_dict[key])
            print("[%s] Prec.: %.2f, Rec.: %.2f, F1: %.2f" % (key, precision, recall, fscore))

    total_p = sum(list(p_dict.values()))
    total_predict = sum(list(total_predict_dict.values()))
    total_entity = sum(list(total_entity_dict.values()))
    precision, recall, fscore = get_metric(p_num=total_p, total_predicted_num=total_predict, total_num=total_entity)
    print("[Total] Prec.: %.2f, Rec.: %.2f, F1: %.2f" % (precision, recall, fscore), flush=True)


def parse_arguments(parser):
    ###Training Hyperparameters
    parser.add_argument('--typing_model', type=str, default="typing", help="typing model folder name")
    parser.add_argument('--extraction_result', type=str, default="extraction.results", help="result file name from extraction")
    args = parser.parse_args()
    for k in args.__dict__:
        print(k + ": " + str(args.__dict__[k]))
    return args

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="LSTM CRF implementation")
    opt = parse_arguments(parser)

    model_folder_name = opt.typing_model
    predicted_file = f"results/{opt.extraction_result}"
    model_path = f"{model_folder_name}/{model_folder_name}.tar.gz"
    model = NERPredictor(model_path)
    config = model.conf


    """
    ### PREPROCESSING Step: find some stamp labels
    ### Reason: the predicted file is the results of extraction model, where the label is {B-general, I-general, S-general} and so on
    ### However, the `label2idx` map is like {B-per: [B-per, B-org, B-loc]}. So we simply need to replace the B-general with either one of {B-per, B-org, B-loc}
    """
    stamp_labels = {}  ## For exmaple: {B- : B-per, I-: I-org}  we don't care the type. Because it doesn't matter during prediction
    for label in config.label2idx:
        if label != "O" and label != START and label != STOP and label != PAD:
            if label[:2] not in stamp_labels:
                stamp_labels[label[:2]] = label

    ## replace B-general to either one of {B-per, B-org, B-loc}
    insts = read_extraction_results(predicted_file)
    for inst in insts:
        for i in range(len(inst.output)):
            if inst.output[i][:2] in stamp_labels:
                inst.output[i] = stamp_labels[inst.output[i][:2]]

    ####

    results = model.predict_insts(insts)
    evaluate(insts=insts, predictions=results, print_details=True)
    write_results(filename="results/pipeline.results", insts=insts)




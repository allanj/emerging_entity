from tqdm import tqdm
from typing import Set
import random
import os
from shutil import copyfile

random.seed(1234)
languages = ["conll2003"]
num_new_type_sents_neededs = [50, 100]
new_types = ["MISC"]
num_data = 10

for language in languages:
    for num_new_type_sents_needed in num_new_type_sents_neededs:
        for new_type in new_types:
            sents = []
            word_labels = []
            has_new_type = False
            with open("../data/"+language+"/train.txt", 'r', encoding='utf-8') as f:
                for line in tqdm(f.readlines()):
                    line = line.rstrip()
                    if line == "":
                        sents.append((word_labels, has_new_type))
                        word_labels = []
                        has_new_type = False
                        continue
                    if new_type in line:
                        has_new_type = True
                    word_labels.append(line)

            random.shuffle(sents)

            previous_num = 0

            for n in range(num_data):

                folder_name = f"../data/{language}/{new_type}/few_random_{num_new_type_sents_needed}_{n}"
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)
                num_new_type_written = 0
                new_type_id = 0
                f_train = open(folder_name+"/train.txt", 'w', encoding='utf-8')
                for (sent, has_new_type) in sents:
                    if has_new_type:
                        if num_new_type_written < num_new_type_sents_needed and new_type_id >= previous_num:
                            for line in sent:
                                f_train.write(line + '\n')
                            f_train.write('\n')
                            num_new_type_written += 1
                        new_type_id += 1
                    if not has_new_type:
                        for line in sent:
                            f_train.write(line + '\n')
                        f_train.write('\n')

                f_train.close()
                previous_num += num_new_type_sents_needed



                orig_dev = f"../data/{language}/dev.txt"
                orig_test = f"../data/{language}/test.txt"

                copyfile(orig_dev, folder_name + "/dev.txt")
                copyfile(orig_test, folder_name + "/test.txt")
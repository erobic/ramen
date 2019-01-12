from __future__ import print_function
import os
import sys
import json
import numpy as np
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset import Dictionary


def create_dictionary(dataroot, dataset, old_dictionary=None, args=None):
    dictionary = Dictionary()
    if old_dictionary is not None:
        print("Copying old dictionary to new dictionary")
        dictionary.word2idx = old_dictionary.word2idx
        dictionary.idx2word = old_dictionary.idx2word

    file_names = [
        'train_questions.json',
        'val_questions.json',
        'test_questions.json'
    ]

    if dataset.lower() == 'vqa2':
        file_names.append('test_dev_questions.json')

    files = []
    for f in file_names:
        files.append(os.path.join(dataroot, 'vqa2', f))

    if args.combine_with is not None:
        for cs in args.combine_with_splits:
            files.append(os.path.join(args.combine_with_dataroot, 'vqa2', cs+"_questions.json"))

    print("files to process {}".format(files))

    for question_path in files:
        qs = json.load(open(question_path))['questions']
        for q in qs:
            dictionary.tokenize(q['question'], True)

    return dictionary

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--old_dictionary_file', type=str, required=False)
    parser.add_argument('--out_dictionary_file', type=str, required=False)
    parser.add_argument('--out_glove_file', type=str, required=False)
    parser.add_argument('--combine_with', type=str, required=False, default=None)
    parser.add_argument('--combine_with_splits', type=str, required=False)
    args = parser.parse_args()
    args.dataroot = args.root + "/" + args.dataset
    if args.combine_with is not None:
        args.combine_with_dataroot = args.root + '/' + args.combine_with
    else:
        args.combine_with_dataroot = None
    if args.combine_with_splits is not None:
        args.combine_with_splits = args.combine_with_splits.split(",")
    args.emb_dim = 300

    args.out_dictionary_json_file = args.dataroot + '/bottom-up-attention/combined_and_individual.json'


    return args


if __name__ == '__main__':
    args = parse_args()
    dataroot = args.dataroot
    if args.old_dictionary_file is not None:
        old_dictionary = Dictionary.load_from_file(args.old_dictionary_file)
    else:
        old_dictionary = None

    d = create_dictionary(dataroot, args.dataset, old_dictionary=old_dictionary,
                          args=args)

    with open(os.path.join(dataroot, 'bottom-up-attention', 'dictionary.json')) as f:
        combined_dict = json.load(f)
        combined_word_to_ix = combined_dict['word_to_ix']

    idx2word = {}
    combined_ix_to_curr_ix = {}
    combined_ix_to_word = {}
    curr_ix_to_combined_ix = {}
    for ix, w in enumerate(d.idx2word):
        if w in combined_word_to_ix:
            combined_ix = combined_word_to_ix[w]
            combined_ix_to_curr_ix[combined_ix] = ix
            curr_ix_to_combined_ix[ix] = combined_ix
            combined_ix_to_word[combined_ix] = w

    json_dict = {
        'combined_ix_to_curr_ix': combined_ix_to_curr_ix,
        'curr_ix_to_combined_ix': curr_ix_to_combined_ix,
        'combined_ix_to_word': combined_ix_to_word,
    }
    with open(args.out_dictionary_json_file, 'w') as jf:
        json.dump(json_dict, jf)

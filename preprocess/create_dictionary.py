from __future__ import print_function
import os
import sys
import json
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from preprocess.dataset import Dictionary
import argparse


def create_dictionary(dataroot):
    dictionary = Dictionary()
    questions = []
    files = [
        'train_questions.json',
        'val_questions.json',
        'test_questions.json',
        'test_dev_questions.json'
    ]
    for path in files:
        question_path = os.path.join(dataroot, path)
        if os.path.exists(question_path):
            qs = json.load(open(question_path))
            if 'questions' in qs:
                qs = qs['questions']
            for q in qs:
                dictionary.tokenize(q['question'], True)
    return dictionary


def create_glove_embedding_init(idx2word, glove_file):
    word2emb = {}
    with open(glove_file, 'r') as f:
        entries = f.readlines()
    emb_dim = len(entries[0].split(' ')) - 1
    print('embedding dim is %d' % emb_dim)
    weights = np.zeros((len(idx2word), emb_dim), dtype=np.float32)

    for entry in entries:
        vals = entry.replace('\n', '').split(' ')

        word = vals[0]
        vals = np.asarray([float(v) for v in vals[1:]])
        word2emb[word] = vals
    for idx, word in enumerate(idx2word):
        if word not in word2emb:
            continue
        weights[idx] = word2emb[word]
    return weights, word2emb


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root')
    args = parser.parse_args()
    data_root = args.data_root
    print("Creating dictionary...")
    d = create_dictionary(os.path.join(data_root, 'questions'))
    features_path = os.path.join(data_root, 'features')
    d.dump_json(os.path.join(features_path, 'dictionary.json'))
    d.dump_to_file(os.path.join(features_path, 'dictionary.pkl'))

    d = Dictionary.load_from_file(os.path.join(features_path, 'dictionary.json'))
    emb_dim = 300
    glove_file = os.path.join(data_root, 'glove', 'glove.6B.%dd.txt' % emb_dim)
    weights, word2emb = create_glove_embedding_init(d.idx2word, glove_file)
    np.save(os.path.join(features_path, 'glove6b_init_%dd.npy' % emb_dim), weights)
    print("Created dictionary...")

from __future__ import print_function
import os
import json
# import cPickle
import six;
from six.moves import cPickle

import numpy as np
import utils
import h5py
import torch
from torch.utils.data import Dataset
import time
import re
from vqa_utils import VqaUtils


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # the least frequent word (`bebe`) as UNK for Visual Genome dataset
                tokens.append(self.word2idx.get(w, self.padding_idx - 1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


def _create_entry(img, question, answer):
    if None != answer:
        answer.pop('image_id')
        answer.pop('question_id')
    entry = {
        'question_id': question['question_id'],
        'image_id': question['image_id'],
        'image': img,
        'question': question['question'],
        'answer': answer}
    return entry


def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features_path
    data_root: root path of dataset
    name: 'train', 'val', 'test-dev2015', test2015'
    """
    question_path = os.path.join(
        dataroot, 'questions/%s_questions.json' % (name))
    questions = json.load(open(question_path))
    if 'questions' in questions:
        questions = questions['questions']
    questions = sorted(questions,
                       key=lambda x: x['question_id'])
    if 'test' not in name and 'test_dev' not in name:
        qn_id_to_ans = {}
        answer_path = os.path.join(dataroot, 'features', '%s_target.json' % name)
        answers = json.load(open(answer_path, 'r'))
        for answer in answers:
            qn_id_to_ans[str(answer['question_id'])] = answer

        entries = []
        for question in questions:
            answer = qn_id_to_ans[str(question['question_id'])].copy()
            utils.assert_eq(question['question_id'], answer['question_id'])
            utils.assert_eq(question['image_id'], answer['image_id'])
            img_id = question['image_id']
            entries.append(_create_entry(img_id2val[str(img_id)], question, answer))
    else:  # test2015
        entries = []
        for question in questions:
            img_id = question['image_id']
            entries.append(_create_entry(img_id2val[str(img_id)], question, None))

    return entries


class VQAFeatureDataset(Dataset):
    def __init__(self, name, dictionary, data_root, adaptive=False, args=None):
        super(VQAFeatureDataset, self).__init__()

        with open(os.path.join(args.vocab_dir, 'answer_ix_map.json')) as af:
            self.answer_ix_map = json.load(af)
            self.ans2label = self.answer_ix_map['answer_to_ix']
            self.label2ans = self.answer_ix_map['ix_to_answer']
        self.num_ans_candidates = len(self.ans2label)

        self.dictionary = dictionary
        self.adaptive = adaptive
        self.args = args

        if args.h5_prefix == 'all':
            h5_name = 'all'
        else:
            h5_name = name

        with open(os.path.join(args.feature_dir, '{}_ids_map.json'.format(h5_name))) as f:
            self.img_id2idx = json.load(f)['image_id_to_ix']
        h5_path = os.path.join(args.feature_dir, '%s%s.hdf5' % (h5_name, '' if self.adaptive else ''))
        self.h5_path = h5_path

        print('loading features_path from h5 file')
        hf = h5py.File(h5_path, 'r')
        if 'image_features' in hf:
            features = hf['image_features']
        else:
            features = hf['features']
        if 'spatial_features' in hf:
            spatials = hf['spatial_features']
        else:
            spatials = hf['boxes']
        self.entries = _load_dataset(data_root, name, self.img_id2idx, self.label2ans)
        self.tokenize(args.token_length)
        print("token length {}".format(args.token_length))
        self.tensorize()
        print("self.features_path.size() {}".format(features.shape))
        self.v_dim = features.shape[1 if self.adaptive else 2] + VqaUtils.get_spatial_length(
            args.spatial_feature_type,
            args.spatial_feature_length)
        self.s_dim = spatials.shape[1 if self.adaptive else 2]
        self.printed = False
        with open(os.path.join(args.data_root, 'questions', name + "_questions.json")) as qf:
            print("Loading questions...")
            qns = json.load(qf)
            if 'questions' in qns:
                qns = qns['questions']
            self.question_map = {}
            for q in qns:
                self.question_map[q['question_id']] = q

    def tokenize(self, max_length):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if None != answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def close_h5_file(self):
        self.hf.close()

    def load_h5(self):
        if not hasattr(self, 'hf'):
            self.hf = h5py.File(self.h5_path, 'r')
            if 'image_features' in self.hf:
                self.features = self.hf['image_features']
            else:
                self.features = self.hf['features']
            if 'spatial_features' in self.hf:
                self.spatials = self.hf['spatial_features']
            else:
                self.spatials = self.hf['boxes']

            if self.adaptive:
                self.pos_boxes = self.hf.get('pos_boxes')

    def __getitem__(self, index):
        self.load_h5()
        entry = self.entries[index]
        feature_ix = entry['image']
        features = self.features[int(feature_ix)]
        # if not self.args.do_not_normalize_image_feats:
        #     features = VqaUtils.normalize_features(features)
        spatials = self.spatials[int(feature_ix)]
        curr_entry = VqaUtils.get_image_features(features, spatials,
                                                 self.args.spatial_feature_type,
                                                 self.args.spatial_feature_length,
                                                 self.args.num_objects)
        question = entry['q_token']
        # invert question
        q_len = len(question)
        question = question.index_select(0, torch.arange(q_len-1, -1, -1).long())
        question_id = entry['question_id']
        full_question = self.question_map[question_id]

        question_type = VqaUtils.get_question_type(full_question)
        answer = entry['answer']
        target = torch.zeros(self.num_ans_candidates)

        if None != answer:
            labels = answer['labels']
            scores = answer['scores']
            if labels is not None:
                target.scatter_(0, labels, scores)
        self.printed = True
        return curr_entry, spatials, question, target, question_type, question_id

    def __len__(self):
        return len(self.entries)
        # return 3000

from __future__ import print_function

import json
import os
from six.moves import cPickle


class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None, ix_to_word=None):
        self.ix_to_word = {}
        if idx2word is not None:
            for i, word in enumerate(idx2word):
                self.ix_to_word[i] = word
        if ix_to_word is not None:
            self.ix_to_word = ix_to_word
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
                tokens.append(self.word2idx[w])
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    def dump_json(self, path):
        j = {'ix_to_word': self.ix_to_word, 'word_to_ix': self.word2idx}
        with  open(path, 'w') as f:
            json.dump(j, f)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        #word2idx, idx2word = cPickle.load(open(path, 'rb'))
        with open(os.path.join(path)) as f:
            c = json.load(f)

        d = cls(c['word_to_ix'], c['ix_to_word'])
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.ix_to_word[len(self.idx2word) - 1] = word
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

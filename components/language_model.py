import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from torch.nn.utils.rnn import pack_padded_sequence


class WordEmbedding(nn.Module):
    """Word Embedding

    The ntoken-th dim is used for padding_idx, which agrees *implicitly*
    with the definition in Dictionary.
    """

    def __init__(self, ntoken, emb_dim, dropout=0):
        super(WordEmbedding, self).__init__()
        self.emb = nn.Embedding(ntoken + 1, emb_dim, padding_idx=ntoken)
        self.dropout = nn.Dropout(dropout)
        self.ntoken = ntoken
        self.emb_dim = emb_dim

    def init_embedding(self, np_file):
        weight_init = torch.from_numpy(np.load(np_file))
        assert weight_init.shape == (self.ntoken, self.emb_dim)
        self.emb.weight.data[:self.ntoken] = weight_init

    def forward(self, x):
        emb = self.emb(x)
        emb = self.dropout(emb)
        return emb


class UpDnQuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers, bidirect, dropout=0, rnn_type='GRU'):
        """Module for question embedding
        """
        super(UpDnQuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU

        self.rnn = rnn_cls(
            in_dim, num_hid, nlayers,
            bidirectional=bidirect,
            dropout=dropout,
            batch_first=True)

        self.in_dim = in_dim
        self.num_hid = num_hid
        self.nlayers = nlayers
        self.rnn_type = rnn_type
        self.ndirections = 1 + int(bidirect)

    def init_hidden(self, batch):
        # just to get the type of tensor
        weight = next(self.parameters()).data
        hid_shape = (self.nlayers * self.ndirections, batch, self.num_hid)
        if self.rnn_type == 'LSTM':
            return (Variable(weight.new(*hid_shape).zero_()),
                    Variable(weight.new(*hid_shape).zero_()))
        else:
            return Variable(weight.new(*hid_shape).zero_())

    def forward(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)

        if self.ndirections == 1:
            return output[:, -1]

        forward_ = output[:, -1, :self.num_hid]
        backward = output[:, 0, self.num_hid:]
        return torch.cat((forward_, backward), dim=1)

    def forward_all(self, x):
        # x: [batch, sequence, in_dim]
        batch = x.size(0)
        hidden = self.init_hidden(batch)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(x, hidden)
        return output


class QuestionEmbedding(nn.Module):
    def __init__(self, in_dim, num_hid, nlayers=1, bidirect=True, dropout=0, rnn_type='GRU'):
        """Module for question embedding
        """
        super(QuestionEmbedding, self).__init__()
        assert rnn_type == 'LSTM' or rnn_type == 'GRU'
        rnn_cls = nn.LSTM if rnn_type == 'LSTM' else nn.GRU
        self.in_dim = in_dim
        num_hid = int(num_hid/2) if bidirect and rnn_type == 'LSTM' else num_hid
        self.num_hid = num_hid
        self.rnn_type = rnn_type
        self.rnn = rnn_cls(in_dim, num_hid, nlayers, bidirectional=bidirect, dropout=dropout, batch_first=True)

    def forward(self, q, qlen=None):
        # x: [batch, sequence, in_dim]
        sorted_qlen_ixs = torch.argsort(qlen, descending=True)
        sorted_q = q[sorted_qlen_ixs]
        sorted_qlen = qlen[sorted_qlen_ixs]
        q_packed = pack_padded_sequence(sorted_q, sorted_qlen, batch_first=True)

        out, (hid, c) = self.rnn(q_packed)
        if self.rnn_type == 'LSTM':
            hid = torch.transpose(hid, 0, 1)
        hid = torch.flatten(hid, start_dim=1)
        hid[sorted_qlen_ixs] = hid[0:len(hid)]
        return hid

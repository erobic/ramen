from copy import deepcopy
import itertools
import os
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
from bootstrap.lib.options import Options
from bootstrap.lib.logger import Logger
import block
from block.models.networks.vqa_net import factory_text_enc
from block.models.networks.mlp import MLP
from components.language_model import WordEmbedding, UpDnQuestionEmbedding
import skipthoughts


def mask_softmax(x, lengths):  # , dim=1)
    mask = torch.zeros_like(x).to(device=x.device, non_blocking=True)
    t_lengths = lengths[:, :, None].expand_as(mask)
    arange_id = torch.arange(mask.size(1)).to(device=x.device, non_blocking=True)
    arange_id = arange_id[None, :, None].expand_as(mask)

    mask[arange_id < t_lengths] = 1
    # https://stackoverflow.com/questions/42599498/numercially-stable-softmax
    # https://stackoverflow.com/questions/34968722/how-to-implement-the-softmax-function-in-python
    # exp(x - max(x)) instead of exp(x) is a trick
    # to improve the numerical stability while giving
    # the same outputs
    x2 = torch.exp(x - torch.max(x))
    x3 = x2 * mask
    epsilon = 1e-5
    x3_sum = torch.sum(x3, dim=1, keepdim=True) + epsilon
    x4 = x3 / x3_sum.expand_as(x3)
    return x4


class RUBiBaseline(nn.Module):

    def __init__(self, config):
        super().__init__()
        # self.self_q_att = self_q_att
        self.agg = {'type': 'max'}
        assert self.agg['type'] in ['max', 'mean']
        self.classif = {'mlp': {'input_dim': 2048, 'dimensions': [2048, 2048, config.num_ans_candidates]}}
        self.fusion = {
            'type': 'block',
            'input_dims': [config.q_emb_dim, 2048],
            'output_dim': 2048,
            'mm_dim': 1000,
            'chunks': 20,
            'rank': 15,
            'dropout_input': 0.,
            'dropout_pre_lin': 0.
        }
        self.residual = False

        # Modules
        txt_enc = {
            'name': 'skipthoughts',
            'type': 'BayesianUniSkip',
            'dropout': 0.25,
            'fixed_emb': False,
            'dir_st': '/hdd/robik/skip-thoughts'
        }
        self.wid_to_word = {i: w for i, w in enumerate(config.dictionary.idx2word)}
        self.txt_enc = self.get_text_enc(self.wid_to_word, txt_enc)
        self.self_q_att = True
        if self.self_q_att:
            self.q_att_linear0 = nn.Linear(config.q_emb_dim//2, 512)
            self.q_att_linear1 = nn.Linear(512, 2)

        self.fusion_module = block.factory_fusion(self.fusion)

        # if self.classif['mlp']['dimensions'][-1] != len(self.aid_to_ans):
        #     Logger()(f"Warning, the classif_mm output dimension ({self.classif['mlp']['dimensions'][-1]})"
        #              f"doesn't match the number of answers ({len(self.aid_to_ans)}). Modifying the output dimension.")
        #     self.classif['mlp']['dimensions'][-1] = len(self.aid_to_ans)

        self.classif_module = MLP(**self.classif['mlp'])

        # Logger().log_value('nparams',
        #                    sum(p.numel() for p in self.parameters() if p.requires_grad),
        #                    should_print=True)
        #
        # Logger().log_value('nparams_txt_enc',
        #                    self.get_nparams_txt_enc(),
        #                    should_print=True)

    def factory_text_enc(self, vocab_words, opt):
        list_words = [vocab_words[i] for i in range(len(vocab_words))]
        if opt['name'] == 'skipthoughts':
            st_class = getattr(skipthoughts, opt['type'])
            seq2vec = st_class(opt['dir_st'],
                               list_words,
                               dropout=opt['dropout'],
                               fixed_emb=opt['fixed_emb'])
        else:
            raise NotImplementedError
        return seq2vec

    def get_text_enc(self, vocab_words, options):
        """
        returns the text encoding network.
        """
        return self.factory_text_enc(vocab_words, options)

    # def get_nparams_txt_enc(self):
    #     params = [p.numel() for p in self.txt_enc.parameters() if p.requires_grad]
    #     if self.self_q_att:
    #         params += [p.numel() for p in self.q_att_linear0.parameters() if p.requires_grad]
    #         params += [p.numel() for p in self.q_att_linear1.parameters() if p.requires_grad]
    #     return sum(params)

    def process_fusion(self, q, mm):
        bsize = mm.shape[0]
        n_regions = mm.shape[1]

        mm = mm.contiguous().view(bsize * n_regions, -1)
        mm = self.fusion_module([q, mm])
        mm = mm.view(bsize, n_regions, -1)
        return mm

    def forward(self, v, b, q, labels, qlen):
        l = qlen
        bsize = v.shape[0]
        n_regions = v.shape[1]

        out = {}

        q = self.process_question(q, l, )
        out['q_emb'] = q
        q_expand = q[:, None, :].expand(bsize, n_regions, q.shape[1])
        q_expand = q_expand.contiguous().view(bsize * n_regions, -1)

        mm = self.process_fusion(q_expand, v, )

        if self.residual:
            mm = v + mm

        if self.agg['type'] == 'max':
            mm, mm_argmax = torch.max(mm, 1)
        elif self.agg['type'] == 'mean':
            mm = mm.mean(1)

        out['mm'] = mm
        out['mm_argmax'] = mm_argmax

        logits = self.classif_module(mm)
        out['logits'] = logits
        return out

    def process_question(self, q, l, txt_enc=None, q_att_linear0=None, q_att_linear1=None):
        if txt_enc is None:
            txt_enc = self.txt_enc
        if q_att_linear0 is None:
            q_att_linear0 = self.q_att_linear0
        if q_att_linear1 is None:
            q_att_linear1 = self.q_att_linear1
        q_emb = txt_enc.embedding(q)

        q, _ = txt_enc.rnn(q_emb)
        if len(l.shape)==1:
            l = l.unsqueeze(1)
        l = l.cuda()

        if self.self_q_att:
            q_att = q_att_linear0(q)
            q_att = F.relu(q_att)
            q_att = q_att_linear1(q_att)
            q_att = mask_softmax(q_att, l)
            # self.q_att_coeffs = q_att
            if q_att.size(2) > 1:
                q_atts = torch.unbind(q_att, dim=2)
                q_outs = []
                for q_att in q_atts:
                    q_att = q_att.unsqueeze(2)
                    q_att = q_att.expand_as(q)
                    q_out = q_att * q
                    q_out = q_out.sum(1)
                    q_outs.append(q_out)
                q = torch.cat(q_outs, dim=1)
            else:
                q_att = q_att.expand_as(q)
                q = q_att * q
                q = q.sum(1)
        else:
            # l contains the number of words for each question
            # in case of multi-gpus it must be a Tensor
            # thus we convert it into a list during the forward pass
            l = list(l.data[:, 0])
            q = txt_enc._select_last(q, l)

        return q

    def process_answers(self, out, key=''):
        batch_size = out[f'logits{key}'].shape[0]
        _, pred = out[f'logits{key}'].data.max(1)
        pred.squeeze_()
        if batch_size != 1:
            out[f'answers{key}'] = [self.aid_to_ans[pred[i].item()] for i in range(batch_size)]
            out[f'answer_ids{key}'] = [pred[i].item() for i in range(batch_size)]
        else:
            out[f'answers{key}'] = [self.aid_to_ans[pred.item()]]
            out[f'answer_ids{key}'] = [pred.item()]
        return out

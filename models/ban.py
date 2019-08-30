"""
Bilinear Attention Networks
Jin-Hwa Kim, Jaehyun Jun, Byoung-Tak Zhang
https://arxiv.org/abs/1805.07932

This code is adapted from: https://github.com/jnhwkim/ban-vqa (written by Jin-Hwa Kim)
"""
import torch.nn as nn

from components.attention import BiAttention
from components.classifier import SimpleClassifier
from components.counting import Counter
from components.fc import FCNet, BCNet
from components.language_model import WordEmbedding, UpDnQuestionEmbedding


class Ban(nn.Module):
    def __init__(self, config):
        super(Ban, self).__init__()
        self.config = config
        self.w_emb = WordEmbedding(config.w_emb_size, 300, .0)
        self.q_emb = UpDnQuestionEmbedding(300, config.q_emb_dim, 1, False, .0)
        self.v_att = BiAttention(config.v_dim, config.num_hid, config.num_hid, config.glimpse)
        self.b_net = []
        self.q_prj = []
        self.c_prj = []
        self.objects = 10  # minimum number of boxes
        for i in range(config.glimpse):
            self.b_net.append(BCNet(config.v_dim, config.num_hid, config.num_hid, None, k=1))
            self.q_prj.append(FCNet([config.num_hid, config.num_hid], '', .2))
            self.c_prj.append(FCNet([self.objects + 1, config.num_hid], 'ReLU', .0))

        self.b_net = nn.ModuleList(self.b_net)
        self.q_prj = nn.ModuleList(self.q_prj)
        self.c_prj = nn.ModuleList(self.c_prj)

        self.classifier = SimpleClassifier(
            config.num_hid, config.num_hid * 2, config.num_ans_candidates, .5)
        self.counter = Counter(self.objects)
        self.drop = nn.Dropout(.5)
        self.tanh = nn.Tanh()

    def forward(self, v, b, q, labels, qlen):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        w_emb = self.w_emb(q)
        q_emb = self.q_emb.forward_all(w_emb)  # [batch, q_len, q_dim]
        boxes = b[:, :, :4].transpose(1, 2)

        b_emb = [0] * self.config.glimpse
        att, logits = self.v_att.forward_all(v, q_emb)  # b x g x v x q

        for g in range(self.config.glimpse):
            b_emb[g] = self.b_net[g].forward_with_weights(v, q_emb, att[:, g, :, :])  # b x l x h

            atten, _ = logits[:, g, :, :].max(2)
            embed = self.counter(boxes, atten)

            q_emb = self.q_prj[g](b_emb[g].unsqueeze(1)) + q_emb
            q_emb = q_emb + self.c_prj[g](embed).unsqueeze(1)

        logits = self.classifier(q_emb.sum(1))
        return {'logits': logits, 'q_emb': q_emb.sum(1)}

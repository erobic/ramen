"""
Bottom-up and top-down attention for image captioning and vqa.
Anderson, Peter, et al.
http://openaccess.thecvf.com/content_cvpr_2018/CameraReady/1163.pdf

This code is adapted from: https://github.com/hengyuan-hu/bottom-up-attention-vqa
"""
import torch.nn as nn

from components.attention import UpDnAttention
from components.classifier import SimpleClassifier
from components.fc import FCNet
from components.language_model import WordEmbedding, UpDnQuestionEmbedding


class UpDn(nn.Module):
    def __init__(self, config):
        super(UpDn, self).__init__()
        self.w_emb = WordEmbedding(config.w_emb_size, 300, 0.0)
        self.q_emb = UpDnQuestionEmbedding(300, config.q_emb_dim, 1, False, 0.0)
        self.v_att = UpDnAttention(config.v_dim, self.q_emb.num_hid, config.num_hid)
        self.q_net = FCNet([self.q_emb.num_hid, config.num_hid])
        self.v_net = FCNet([config.v_dim, config.num_hid])
        self.classifier = SimpleClassifier(
            config.num_hid, config.num_hid * 2, config.num_ans_candidates, 0.5)

    def forward(self, v, b, q, labels, qlen):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # print("q = {}".format(q))
        w_emb = self.w_emb(q)
        # print("w_emb = {}".format(w_emb))
        q_emb = self.q_emb(w_emb)  # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1)  # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return {'logits': logits, 'q_emb': q_emb}

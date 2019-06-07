"""
@inproceedings{santoro2017simple,
  title={A simple neural network module for relational reasoning},
  author={Santoro, Adam and Raposo, David and Barrett, David G and Malinowski, Mateusz and Pascanu, Razvan and Battaglia, Peter and Lillicrap, Timothy},
  booktitle={Advances in neural information processing systems},
  pages={4967--4976},
  year={2017}
}

Code based on: https://github.com/mesnico/RelationNetworks-CLEVR
"""
import torch
import torch.nn as nn

from components.language_model import QuestionEmbedding
from components.language_model import WordEmbedding
from components.pairwise_relations import PairwiseRelationModule


class RelationNetwork(nn.Module):
    def __init__(self, config):
        super(RelationNetwork, self).__init__()
        self.config = config
        self.w_emb = WordEmbedding(config.w_emb_size, 300)
        self.w_emb.init_embedding(config.glove_file)
        self.q_emb = QuestionEmbedding(300, self.config.q_emb_dim, 1, bidirect=False, dropout=0, rnn_type='GRU')
        self.relation_module = PairwiseRelationModule(config.v_dim + config.q_emb_dim, config.interactor_sizes, config.aggregator_sizes)
        self.classifier = nn.Linear(config.aggregator_sizes[-1], config.num_ans_candidates)

    def forward(self, v, b, q, a=None, qlen=None):
        """Forward

        v: [batch, num_objs, v_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits
        """
        q = self.w_emb(q)
        q_words_emb, q_emb = self.q_emb(q)
        q_emb_repeated = q_emb.unsqueeze(1)
        q_emb_repeated = q_emb_repeated.repeat(1, v.shape[1], 1)
        vq_paired = torch.cat((v, q_emb_repeated), dim=2)
        rel, _ = self.relation_module(vq_paired)
        logits = self.classifier(rel)
        return logits

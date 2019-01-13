import torch.nn as nn

from components import nonlinearity
from components.language_model import QuestionEmbedding
from components.language_model import WordEmbedding
from components.multi_modal_core import MultiModalCore


class Ramen(nn.Module):
    def __init__(self, config):
        super(Ramen, self).__init__()
        self.config = config
        self.mmc_net = MultiModalCore(config)
        self.w_emb = WordEmbedding(config.w_emb_size, 300)
        self.w_emb.init_embedding(config.glove_file)
        self.q_emb = QuestionEmbedding(300, self.config.q_emb_dim, 1, bidirect=True, dropout=0, rnn_type='GRU')

        clf_in_size = config.mmc_aggregator_dim * 2
        classifier_layers = []
        for ix, size in enumerate(config.classifier_sizes):
            in_s = clf_in_size if ix == 0 else config.classifier_sizes[ix - 1]
            out_s = size
            lin = nn.Linear(in_s, out_s)
            classifier_layers.append(lin)
            classifier_layers.append(getattr(nonlinearity, config.classifier_nonlinearity)())
            classifier_layers.append(nn.Dropout(p=config.classifier_dropout))

        self.pre_classification_layers = nn.Sequential(*classifier_layers)
        self.classifier = nn.Linear(out_s, config.num_ans_candidates)

    def forward(self, v, b, q, a=None):
        """Forward

        v: [batch, num_objs, v_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits
        """
        batch_size, num_objs, v_emb_dim = v.size()
        b = b[:, :, :4]
        q = self.w_emb(q)
        q_words_emb, q_emb = self.q_emb(q)
        mmc, mmc_aggregated = self.mmc_net(v, b, q_emb)  # B x num_objs x num_hid and B x num_hid
        final_emb = self.pre_classification_layers(mmc_aggregated)
        logits = self.classifier(final_emb)
        return logits

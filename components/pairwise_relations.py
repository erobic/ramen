import torch
import torch.nn as nn


def make_pairs(v):
    """
    Create object pairs from the provided object list
    :param v: Batch size (B) X num. objects (o) X Embedding size
    :return: Batch-wise object pairs: B X o**2 x (Embedding size *2)
    """
    b, num_objs, vis_emb_size = v.size()

    # First repeat objects in dim = 1
    v_i = torch.unsqueeze(v, 1)  # (B x 1 x num_objs x vis_emb_size)
    v_i = v_i.repeat(1, num_objs, 1, 1)  # (B x num_objs x num_objs x vis_emb_size)

    # Second repeat objects in dim = 2
    v_j = torch.unsqueeze(v, 2)  # (B x num_objs x 1 x vis_emb_size)
    v_j = v_j.repeat(1, 1, num_objs, 1)  # (B x num_objs x num_objs x vis_emb_size)

    # concatenate all together
    x_full = torch.cat([v_i, v_j], 3)  # (B x num_objs x num_objs x 2 * vis_emb_size)
    obj_pairs = x_full.view(b, num_objs ** 2, 2, vis_emb_size)
    return obj_pairs


class PairwiseRelationModule(nn.Module):
    def __init__(self, obj_dim, interactor_sizes, aggregator_sizes):
        """
        Computes pairwise relations between objects
        interaction layers refer to 'g' function of original relation network paper
        aggregator layers refer to 'f' function of original relation  network paper
        :param obj_dim:
        :param interactor_sizes:
        :param aggregator_sizes:
        """
        super(PairwiseRelationModule, self).__init__()
        self.interaction_sizes, self.aggregation_sizes = interactor_sizes, aggregator_sizes
        self.interaction_layers = []
        for gix, g_size in enumerate(interactor_sizes):
            in_s = obj_dim * 2 if gix == 0 else interactor_sizes[gix - 1]
            out_s = g_size
            self.interaction_layers.append(nn.Linear(in_s, out_s))
            self.interaction_layers.append(nn.ReLU())
        self.interaction_layers = nn.Sequential(*self.interaction_layers)
        self.sum_bnorm = nn.BatchNorm1d(interactor_sizes[-1])
        self.aggregation_layers = []

        if aggregator_sizes is not None and len(aggregator_sizes) > 0:
            for fix, f_size in enumerate(aggregator_sizes):
                in_s = g_size if fix == 0 else aggregator_sizes[-1]
                out_s = f_size
                self.aggregation_layers.append(nn.Linear(in_s, out_s))
                self.aggregation_layers.append(nn.ReLU())
            self.aggregation_layers = nn.Sequential(*self.aggregation_layers)

    def forward(self, objs):
        """

        :param objs: B x num_objs x obj_dim
        :return:
        """
        b, num_objs, obj_dim = objs.size()
        x_pairs = make_pairs(objs).reshape(b, num_objs ** 2,
                                           obj_dim * 2)  # B objs num_objs objs
        x_pairs = self.interaction_layers(x_pairs)

        interaction_sum = x_pairs.sum(1).squeeze()
        interaction_sum = self.sum_bnorm(interaction_sum)
        if self.aggregation_sizes is not None and len(self.aggregation_sizes) > 0:
            aggregation_sum = self.aggregation_layers(interaction_sum)
        else:
            aggregation_sum = interaction_sum
        return aggregation_sum, x_pairs

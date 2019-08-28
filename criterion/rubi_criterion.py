import torch.nn as nn


class RUBiCriterion(nn.Module):

    def __init__(self, question_loss_weight=1.0):
        super().__init__()

        self.question_loss_weight = question_loss_weight
        # self.fusion_loss = nn.CrossEntropyLoss()
        # self.question_loss = nn.CrossEntropyLoss()
        self.fusion_loss = nn.BCEWithLogitsLoss()
        self.question_loss = nn.BCEWithLogitsLoss()

    def forward(self, net_out, labels):
        out = {}
        # logits = net_out['logits']
        logits_q = net_out['logits_q']
        logits_rubi = net_out['logits_rubi']
        # class_id = labels.squeeze(1)
        fusion_loss = self.fusion_loss(logits_rubi, labels)
        question_loss = self.question_loss(logits_q, labels)
        loss = fusion_loss + self.question_loss_weight * question_loss

        out['loss'] = loss
        out['loss_rubi'] = fusion_loss
        out['loss_q'] = question_loss
        return out

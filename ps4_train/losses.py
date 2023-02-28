import torch
from torch import nn


class CrossEntropyTimeDistributedLoss(nn.Module):
    """loss function for multi-timsetep model output"""
    def __init__(self):
        super(CrossEntropyTimeDistributedLoss, self).__init__()

        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, y_hat, y, mask=None):

        _y_hat = y_hat.squeeze(0)
        _y = y.squeeze(0)

        # Loss from one sequence
        if mask is not None:
            mask -= 1
            mask *= -1

            indices = torch.nonzero(mask)
            loss = self.loss_func(_y_hat[indices].view(len(indices), -1), _y[indices].view(len(indices)))
        else:
            loss = self.loss_func(_y_hat, _y)

        loss = torch.mean(loss)

        return loss


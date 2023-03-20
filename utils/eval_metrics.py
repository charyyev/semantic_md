import torch
import numpy as np


def depth_metrics(pred, target):
    assert pred.shape == target.shape

    # accuracy with threshold t = 1.25, 1.25^2, 1.25^3
    thresh = torch.max((target / pred), (pred / target))
    delta1 = torch.sum(thresh < 1.25).float() / len(thresh)
    delta2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    delta3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    # mean relative error, root mean squared error, log10 error
    diff = pred - target
    abs_rel = torch.mean(torch.abs(diff) / target)
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))

    return {'delta1': delta1.item(), 'delta2': delta2.item(), 'delta3': delta3.item(),
            'abs_rel': abs_rel.item(), 'rmse': rmse.item(), 'log10':log10.item()}

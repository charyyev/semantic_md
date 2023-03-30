import torch
import numpy as np


def depth_metrics(pred, target):
    assert pred.shape == target.shape

    # accuracy with threshold t = 1.25, 1.25^2, 1.25^3
    thresh = torch.max((target / pred), (pred / target))
    delta1 = torch.mean(thresh < 1.25)
    delta2 = torch.mean(thresh < 1.25 ** 2)
    delta3 = torch.mean(thresh < 1.25 ** 3)

    # mean absolute relative error, mean squared relative error, root mean squared error, rsme log, log10 error
    diff = pred - target
    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(torch.log(pred) - torch.log(target), 2)))
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))

    return {'delta1': delta1.item(), 'delta2': delta2.item(), 'delta3': delta3.item(),
            'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item()}

import torch
import numpy as np


def depth_metrics(pred, target):
    assert pred.shape == target.shape

    # accuracy with threshold t = 1.25, 1.25^2, 1.25^3
    thresh = torch.max((target / pred), (pred / target))
    delta1 = torch.mean((thresh < 1.25).float())
    delta2 = torch.mean((thresh < 1.25 ** 2).float())
    delta3 = torch.mean((thresh < 1.25 ** 3).float())

    # mean absolute relative error, mean squared relative error, root mean squared error, rsme log, log10 error
    diff = pred - target
    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)
    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(torch.log(pred) - torch.log(target), 2)))
    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))

    return {'delta1': delta1.item(), 'delta2': delta2.item(), 'delta3': delta3.item(),
            'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(), 'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item()}


def test():

    epsilon = 1e-18
    # create an example 3D tensor with predicted and true values
    y_pred = torch.tensor([
        [[1.2, 2.4], [3.6, -4.8]],
        [[5.1, 3.2], [7.3, 8.4]],
        [[9.5, 10.6], [11.7, -12.8]]
    ])
    y_pred = torch.clamp(y_pred, min=epsilon, max=None)

    y_target = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]]
    ])
    print(y_pred.size())

    metrics = depth_metrics(y_pred, y_target)

    # print the loss value
    print(metrics)


if __name__ == '__main__':
    test()

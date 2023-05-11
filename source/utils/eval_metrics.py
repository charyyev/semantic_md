import numpy as np
import torch

# from torchmetrics.classification import MulticlassJaccardIndex, MulticlassAccuracy
import sklearn
from sklearn.metrics import accuracy_score, confusion_matrix, jaccard_score
from torchmetrics.functional.classification import (
    multiclass_accuracy,
    multiclass_jaccard_index,
)
from utils.configs import Config


def unnormalize(inp, min_depth, max_depth):
    inp = inp * (max_depth - min_depth) + min_depth
    return inp


def depth_metrics(pred, target, epsilon, config):
    pred = torch.clamp(pred, min=epsilon, max=None)
    assert pred.shape == target.shape
    depth = config["transformations"]["depth_range"]
    min_depth, max_depth = depth["min"], depth["max"]
    target = unnormalize(target, min_depth, max_depth)
    pred = unnormalize(pred, min_depth, max_depth)

    # accuracy with threshold t = 1.25, 1.25^2, 1.25^3
    thresh = torch.max((target / pred), (pred / target))
    delta1 = torch.mean((thresh < 1.25).float())
    delta2 = torch.mean((thresh < 1.25**2).float())
    delta3 = torch.mean((thresh < 1.25**3).float())

    # mean absolute relative error, mean squared relative error, root mean squared error, rsme log, log10 error
    diff = pred - target
    abs_rel = torch.nanmean(torch.abs(diff) / target)
    sq_rel = torch.nanmean(torch.pow(diff, 2) / target)
    rmse = torch.sqrt(torch.nanmean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(
        torch.nanmean(torch.pow(torch.log(pred) - torch.log(target), 2))
    )
    log10 = torch.nanmean(torch.abs(torch.log10(pred) - torch.log10(target)))

    return {
        "delta1": delta1.item(),
        "delta2": delta2.item(),
        "delta3": delta3.item(),
        "abs_rel": abs_rel.item(),
        "sq_rel": sq_rel.item(),
        "rmse": rmse.item(),
        "rmse_log": rmse_log.item(),
        "log10": log10.item(),
    }


def seg_metrics(pred, target, epsilon, config):
    pred = torch.squeeze(torch.argmax(pred, dim=1))
    mask = torch.ne(target, -1)
    target = torch.masked_select(target, mask).cpu().numpy()
    pred = torch.masked_select(pred, mask).cpu().numpy()
    meanIoU = jaccard_score(target, pred, average="macro")
    pixelAcc = accuracy_score(target, pred)
    cm_acc = confusion_matrix(target, pred)
    meanAcc = np.mean(cm_acc.diagonal() / (cm_acc.sum(axis=1) + epsilon))

    return {"meanIoU": meanIoU, "meanAcc": meanAcc, "pixelAcc": pixelAcc}


def border_metrics(pred, target, epsilon, config):
    tp = ((pred == 1) & (target == 1)).sum().item()
    tn = ((pred == 0) & (target == 0)).sum().item()
    fp = ((pred == 1) & (target == 0)).sum().item()
    fn = ((pred == 0) & (target == 1)).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1_score = (2 * precision * recall) / (precision + recall + epsilon)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "F1": f1_score,
    }


def test():
    epsilon = 1e-4
    # create an example 3D tensor with predicted and true values
    y_pred = torch.tensor(
        [
            [[1.2, 2.4], [3.6, -4.8]],
            [[5.1, 3.2], [7.3, 8.4]],
            [[9.5, 10.6], [11.7, -12.8]],
        ]
    )
    y_pred = torch.clamp(y_pred, min=epsilon, max=None)

    y_target = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[5.0, 6.0], [7.0, 8.0]],
            [[9.0, 10.0], [11.0, 12.0]],
        ]
    )
    # print(y_pred)
    # print(y_pred.size())

    s_pred = torch.tensor(
        [
            [0, 0, 0],
            [1, 1, 1],
            [2, 2, 2],
        ]
    )

    s_target = torch.tensor(
        [
            [0, 1, 0],
            [1, -1, 1],
            [0, 1, 2],
        ]
    )

    metrics = seg_metrics(s_pred, s_target, epsilon)
    print(metrics)


if __name__ == "__main__":
    test()

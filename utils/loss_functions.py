import torch
import torch.nn as nn


class ShiftInvariantLoss(nn.Module):
    """
    https://proceedings.neurips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf
    """

    def __init__(self, lam=.5):
        super(ShiftInvariantLoss, self).__init__()
        self.lam = lam

    def forward(self, y_pred, y_true):
        n = y_true.size()[-1] * y_true.size()[-2]
        diff = y_pred - y_true

        left_term = torch.mean(diff ** 2)
        right_term = torch.sum(diff, dim=[1, 2]) ** 2
        loss_per_item = left_term - (self.lam / (n ** 2)) * right_term
        return torch.mean(loss_per_item)


class BerHuLoss(nn.Module):
    """
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7785097
    """

    def __init__(self, contains_nan=True):
        super(BerHuLoss, self).__init__()
        self.contains_nan = contains_nan

    def forward(self, y_pred, y_true):
        if self.contains_nan:
            mask = (~torch.isnan(y_pred)) & (~torch.isnan(y_true))
            y_pred = torch.masked_select(y_pred, mask)
            y_true = torch.masked_select(y_true, mask)

        diff = torch.abs(y_pred - y_true)
        c = .2 * torch.max(diff)

        # Conditional statement to determine how to calculate loss element-wise
        loss = torch.zeros_like(diff)
        mask = diff <= c
        loss[mask] = diff[mask]
        loss[~mask] = (diff[~mask] ** 2 + c * +2) / (2 * c)

        return loss.mean()


def test():
    # create an example 3D tensor with predicted and true values
    y_pred = torch.tensor([
        [[1.2, 2.4], [3.6, 4.8]],
        [[5.1, 6.2], [7.3, 8.4]],
        [[9.5, 10.6], [11.7, 12.8]]
    ])
    y_true = torch.tensor([
        [[1.0, 2.0], [3.0, 4.0]],
        [[5.0, 6.0], [7.0, 8.0]],
        [[9.0, 10.0], [11.0, 12.0]]
    ])
    print(y_pred.size())

    # instantiate your custom loss function
    criterion = BerHuLoss()

    # compute the loss between y_pred and y_true
    loss = criterion(y_pred, y_true)

    # print the loss value
    print("Loss:", loss.item())


if __name__ == '__main__':
    test()

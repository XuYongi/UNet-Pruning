import torch
from torch.autograd import Function
from torch import nn,optim

import torch
import torch.nn as nn
# --------------------------- BINARY LOSSES ---------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, weight=None, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.bce_fn = nn.BCEWithLogitsLoss(weight=self.weight)

    def forward(self, preds, labels):
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            labels = labels[mask]
            preds = preds[mask]

        logpt = -self.bce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss
# --------------------------- MULTICLASS LOSSES ---------------------------
class MFocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2, weight=None, ignore_index=255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
        self.ce_fn = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index)

    def forward(self, preds, labels):
        logpt = -self.ce_fn(preds, labels)
        pt = torch.exp(logpt)
        loss = -((1 - pt) ** self.gamma) * self.alpha * logpt
        return loss


class AddNet(nn.Module):
    def __init__(self):
        super(AddNet, self).__init__()
        self.input  = nn.Conv2d(in_channels=1,out_channels=8,kernel_size=3,dilation=2, padding=2,bias=True)
        self.conv2d = nn.Conv2d(in_channels=8, out_channels=8, kernel_size=3, dilation=2, padding=2,bias=True)
        self.out = nn.Conv2d(in_channels=8, out_channels=1, kernel_size=3, dilation=2, padding=2,bias=True)
        self.lrelu = nn.ReLU()
        self.inorm2d = nn.InstanceNorm2d(8)
        self.sigmoid = nn.Sigmoid()
    def forward(self, input):
        out = self.input(input)
        out = self.lrelu(out)
        # out = self.inorm2d(out)
        out = self.conv2d(out)
        out = self.lrelu(out)
        # out = self.inorm2d(out)
        out = self.out(out)
        out = self.sigmoid(out)
        return out

from dice_loss import dice_loss
if __name__ == '__main__':
    add = AddNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(add.parameters(), lr=0.01, weight_decay=1e-8)
    a = torch.full([1, 1, 7, 7], 1, requires_grad=True)
    b= torch.rand([1, 1, 7, 7],requires_grad=True)
    # a =  a+b
    for i in range(20):
        c = add(b)
        print(c)
        criterion = FocalLoss()
        # criterion = nn.MSELoss()

        # criterion = dice_loss
        dice = criterion(c,a)
        print('loss***:',dice)
        optimizer.zero_grad()
        dice.backward()
        optimizer.step()
    print(b,'\n',a)
    print('c:',add(b))

import torch
from torch.autograd import Function
from torch import nn,optim

class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t =  (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

class DiceLoss(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t =  1 - (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target

def dice_loss(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceLoss().forward(c[0], c[1])

    return s / (i + 1)



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
if __name__ == '__main__':
    add = AddNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(add.parameters(), lr=0.01, weight_decay=1e-8)
    a = torch.full([1, 1, 7, 7], 1, requires_grad=True)
    b = torch.full([1, 1, 7, 7], 0, requires_grad=True)
    for i in range(20):
        c = add(b)
        print(c)
        dice = dice_loss(c,a)
        # dice = criterion(c,a)
        print('loss***:',dice)
        optimizer.zero_grad()
        dice.backward()
        optimizer.step()
    print(b,'\n',a)
    print(add(b))

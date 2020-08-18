from unet.model_level3_dl_dilationCov_transConv import Modified2DUNet as UNet
from nni.compression.torch import LevelPruner, SlimPruner, FPGMPruner, L1FilterPruner, \
    L2FilterPruner, AGP_Pruner, ActivationMeanRankFilterPruner, ActivationAPoZRankFilterPruner, \
    TaylorFOWeightFilterPruner, NaiveQuantizer, QAT_Quantizer, DoReFaQuantizer, BNNQuantizer
from torchvision.models import vgg19_bn
import torch
from models.cifar10.vgg import VGG

from torch import nn
from torchvision.models.resnet import resnet18 #as testNet

class Pad(nn.Module):
    """Upscaling then double conv"""

    def __init__(self):
        super().__init__()

    def forward(self, x1, x2):
        # x1 = self.up(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        # x = torch.cat([x2, x1], dim=1)
        return x1

class testNet(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1, base_n_filter = 8):
        super(testNet, self).__init__()
        self.Pad = Pad()
        self.in_channels = n_channels
        self.n_channels= n_channels
        self.n_classes = n_classes
        self.bilinear = False
        self.base_n_filter = base_n_filter

        self.lrelu = nn.LeakyReLU()
        self.dropout2d = nn.Dropout2d(p=0.6)
        self.upsacle1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsacle2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.softmax = nn.Softmax(dim=1)

        # Level 1 context pathway
        self.conv2d_c1_1 = nn.Conv2d(self.in_channels, self.base_n_filter, kernel_size=3, stride=1, dilation=2, padding=2, bias=False)
        self.conv2d_c1_2 = nn.Conv2d(self.base_n_filter, self.base_n_filter, kernel_size=3, stride=1,dilation=2, padding=2, bias=False)
        self.lrelu_conv_c1 = self.lrelu_conv(self.base_n_filter, self.base_n_filter)
        self.inorm2d_c1 = nn.InstanceNorm2d(self.base_n_filter)

        # Level 2 context pathway
        self.conv2d_c2 = nn.Conv2d(self.base_n_filter, self.base_n_filter*2, kernel_size=3, stride=2, dilation=2, padding=2, bias=False)
        self.norm_lrelu_conv_c2_1 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
        self.norm_lrelu_conv_c2_2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
        self.inorm2d_c2 = nn.InstanceNorm2d(self.base_n_filter*2)

        # Level 3 context pathway
        self.conv2d_c3 = nn.Conv2d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, dilation=2, padding=2, bias=False)
        self.norm_lrelu_conv_c3_1 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
        self.norm_lrelu_conv_c3_2 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
        self.inorm2d_c3 = nn.InstanceNorm2d(self.base_n_filter*4)

        # Level 4 context pathway
        # self.conv2d_c4 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
        # self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
        # self.inorm2d_c4 = nn.InstanceNorm2d(self.base_n_filter*8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv2d_c5 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, dilation=2, padding=2, bias=False)
        self.norm_lrelu_conv_c5_1 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
        self.norm_lrelu_conv_c5_2 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

        self.conv2d_l0 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*4, kernel_size = 1, stride=1, padding=0, bias=False)
        self.inorm2d_l0 = nn.InstanceNorm2d(self.base_n_filter*4)

        # # Level 1 localization pathway
        # self.conv_norm_lrelu_l1 = self.conv_norm_lrelu(self.base_n_filter*16, self.base_n_filter*16)
        # self.conv2d_l1 = nn.Conv2d(self.base_n_filter*16, self.base_n_filter*8, kernel_size=1, stride=1, padding=0, bias=False)
        # self.norm_lrelu_upscale_conv_norm_lrelu_l1 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = self.conv_norm_lrelu(self.base_n_filter*8, self.base_n_filter*8)
        self.conv2d_l2 = nn.Conv2d(self.base_n_filter*8, self.base_n_filter*4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = self.conv_norm_lrelu(self.base_n_filter*4, self.base_n_filter*4)
        self.conv2d_l3 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = self.norm_lrelu_upscale_conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = self.conv_norm_lrelu(self.base_n_filter*2, self.base_n_filter*2)
        self.conv2d_l4 = nn.Conv2d(self.base_n_filter*2, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds2_1x1_conv2d = nn.Conv2d(self.base_n_filter*8, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv2d = nn.Conv2d(self.base_n_filter*4, self.n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.sigmoid = nn.Sigmoid()


    def conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1,dilation=2, padding=2, bias=False),
            nn.InstanceNorm2d(feat_out),
            nn.LeakyReLU())

    def norm_lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm2d(feat_in),
            nn.LeakyReLU(),
            nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, dilation=2, padding=2, bias=False))

    def lrelu_conv(self, feat_in, feat_out):
        return nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, dilation=2, padding=2, bias=False))

    def norm_lrelu_upscale_conv_norm_lrelu(self, feat_in, feat_out):
        return nn.Sequential(
            nn.InstanceNorm2d(feat_in),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(feat_in,feat_in,kernel_size=3, stride=2, padding=1, bias=False),
            # nn.Upsample(scale_factor=2),
            # should be feat_in*2 or feat_in
            nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, dilation=2, padding=2, bias=False),
            nn.InstanceNorm2d(feat_out),
            nn.LeakyReLU())

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv2d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv2d_c1_2(out)
        out = self.dropout2d(out)
        out = self.lrelu_conv_c1(out)
        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm2d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv2d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2_1(out)
        out = self.dropout2d(out)
        out = self.norm_lrelu_conv_c2_2(out)
        out += residual_2
        out = self.inorm2d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv2d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3_1(out)
        out = self.dropout2d(out)
        out = self.norm_lrelu_conv_c3_2(out)
        # out += residual_3
        out = self.inorm2d_c3(out)
        out = self.lrelu(out)
        context_3 = out


        # Level 5
        out = self.conv2d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5_1(out)
        out = self.dropout2d(out)
        out = self.norm_lrelu_conv_c5_2(out)
        # out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        out = self.conv2d_l0(out)
        out = self.inorm2d_l0(out)
        out = self.lrelu(out)

        # Level 2 localization pathway
        out = self.Pad(out, context_3)
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv2d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = self.Pad(out, context_2)
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv2d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = self.Pad(out, context_1)
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv2d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv2d(ds2)
        ds1_ds2_sum_upscale = self.upsacle1(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv2d(ds3)
        ds1_ds2_sum_upscale = self.Pad(ds1_ds2_sum_upscale, ds3_1x1_conv)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsacle2(ds1_ds2_sum_upscale_ds3_sum)
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.Pad(ds1_ds2_sum_upscale_ds3_sum_upscale, out_pred)
        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale
        out = self.sigmoid(out)

        return out

# class testNet(nn.Module):

#     def __init__(self, num_classes=10,bias = False):
#         super(testNet, self).__init__()
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=bias),
#             nn.InstanceNorm2d(64),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             # nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias),

#             # nn.BatchNorm2d(64),

#         )
#         self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias)
#         self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.conv4 = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=bias)
#         self.transconv = nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2, padding=1, bias=bias)

#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.Pad = Pad()
#         self.classifier = nn.Sequential(
#             nn.Linear(64 , 10),
#         )

#     def forward(self, x):
#         x = self.features(x)
#         xt = x
#         x = self.conv2(x)
#         x += xt

#         x = self.Pad(x,xt) 
#         x = self.conv3(x)
#         x = self.transconv(x)
#         x= torch.cat([x,xt],dim=1)
#         x = self.conv4(x)
#         x = self.avgpool(x)

#         x = torch.flatten(x, 1)
#         x = self.classifier(x)
#         return x
# def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                      padding=dilation, groups=groups, bias=False, dilation=dilation)


# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



# class testNet(nn.Module):
#     expansion = 1

#     def __init__(self, inplanes=64, planes=64, stride=1, downsample=None, groups=1,
#                  base_width=64, dilation=1, norm_layer=None):
#         super(testNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1 or base_width != 64:
#             raise ValueError('BasicBlock only supports groups=1 and base_width=64')
#         if dilation > 1:
#             raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes, stride)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = conv3x3(planes, planes)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride

#     def forward(self, x):

#         out = self.conv1(x)
#         # out = self.bn1(out)
#         identity = out
#         # out = self.relu(out)

#         out = self.conv2(out)
#         # out = self.bn2(out)

#         # if self.downsample is not None:
#         #     identity = self.downsample(x)

#         out += identity
#         # out = self.relu(out)

#         return out


model = testNet()
# model.load_state_dict(torch.load('./savedmodel/CP_epoch13Trainloss0.06215980889821293ValDice0.9336298553337536.pth',
#                                 map_location='cuda'))

config_list = [{ 
            'sparsity': 0.5,
            'op_types': ['Conv2d'],
            # 'op_names': ['features.0'],
}]
pruner = L2FilterPruner(model, config_list)
pruner.compress()
for epoch in range(10):
    pruner.update_epoch(epoch)

pruner.export_model("./save_pruner/pruned_mode_test.pth","./save_pruner/pruned_mask_test.pth")
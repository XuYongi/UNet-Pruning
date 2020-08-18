import torch.nn as nn
import torch

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

class Modified2DUNet_forpruner(nn.Module):
    def __init__(self, n_channels = 3, n_classes = 1, base_n_filter = 8):
        super(Modified2DUNet_forpruner, self).__init__()
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


class Modified2DUNet(nn.Module):
	def __init__(self, n_channels, n_classes, base_n_filter = 8):
		super(Modified2DUNet, self).__init__()
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
		self.norm_lrelu_conv_c2 = self.norm_lrelu_conv(self.base_n_filter*2, self.base_n_filter*2)
		self.inorm2d_c2 = nn.InstanceNorm2d(self.base_n_filter*2)

		# Level 3 context pathway
		self.conv2d_c3 = nn.Conv2d(self.base_n_filter*2, self.base_n_filter*4, kernel_size=3, stride=2, dilation=2, padding=2, bias=False)
		self.norm_lrelu_conv_c3 = self.norm_lrelu_conv(self.base_n_filter*4, self.base_n_filter*4)
		self.inorm2d_c3 = nn.InstanceNorm2d(self.base_n_filter*4)

		# Level 4 context pathway
		# self.conv2d_c4 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, padding=1, bias=False)
		# self.norm_lrelu_conv_c4 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
		# self.inorm2d_c4 = nn.InstanceNorm2d(self.base_n_filter*8)

		# Level 5 context pathway, level 0 localization pathway
		self.conv2d_c5 = nn.Conv2d(self.base_n_filter*4, self.base_n_filter*8, kernel_size=3, stride=2, dilation=2, padding=2, bias=False)
		self.norm_lrelu_conv_c5 = self.norm_lrelu_conv(self.base_n_filter*8, self.base_n_filter*8)
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
			nn.ConvTranspose2d(feat_in,feat_in,kernel_size=3, stride=2, padding=1),
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
		out = self.norm_lrelu_conv_c2(out)
		out = self.dropout2d(out)
		out = self.norm_lrelu_conv_c2(out)
		out += residual_2
		out = self.inorm2d_c2(out)
		out = self.lrelu(out)
		context_2 = out

		# Level 3 context pathway
		out = self.conv2d_c3(out)
		residual_3 = out
		out = self.norm_lrelu_conv_c3(out)
		out = self.dropout2d(out)
		out = self.norm_lrelu_conv_c3(out)
		out += residual_3
		out = self.inorm2d_c3(out)
		out = self.lrelu(out)
		context_3 = out


		# Level 5
		out = self.conv2d_c5(out)
		residual_5 = out
		out = self.norm_lrelu_conv_c5(out)
		out = self.dropout2d(out)
		out = self.norm_lrelu_conv_c5(out)
		out += residual_5
		out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

		out = self.conv2d_l0(out)
		out = self.inorm2d_l0(out)
		out = self.lrelu(out)

		# Level 1 localization pathway
		# out = self.Pad(out, context_4)
		# out = torch.cat([out, context_4], dim=1)
		# out = self.conv_norm_lrelu_l1(out)
		# out = self.conv2d_l1(out)
		# out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

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
		# seg_layer = out
		# out = out.permute(0, 2, 3, 4, 1).contiguous().view(-1, self.n_classes)
		# out = out.view(-1, self.n_classes)
		# out = self.softmax(out)
		return out

if __name__ == '__main__':
	Net = Modified2DUNet(3,1)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	Net = Net.to(device)

	from torchsummary import summary
	summary(Net, (3, 950,1510))
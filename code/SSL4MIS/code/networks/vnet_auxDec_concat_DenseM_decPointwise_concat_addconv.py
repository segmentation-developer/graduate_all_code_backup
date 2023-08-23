import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions.uniform import Uniform
import numpy as np

class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i==0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Pointwise(nn.Module):
    def __init__(self, n_filters_in, n_filters_out):
        super(Pointwise, self).__init__()

        ops = []
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, 1, padding=0))
        ops.append(nn.BatchNorm3d(n_filters_out))
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class ConvBlock_for_perturb(nn.Module):
    def __init__(self, n_stages, feature_stages, n_filters, normalization='none'):
        super(ConvBlock_for_perturb, self).__init__()

        ops = []
        for i in range(n_stages):
            input_channel = n_filters * feature_stages

            ops.append(nn.Conv3d(input_channel,input_channel, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(input_channel))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=input_channel))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(input_channel))
            elif normalization != 'none':
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization='none'):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != 'none':
                assert False

            if i != n_stages-1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = (self.conv(x) + x)
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != 'none':
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))
            if normalization == 'batchnorm':
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == 'groupnorm':
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == 'instancenorm':
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(nn.ConvTranspose3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride))

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization='none'):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(nn.Upsample(scale_factor=stride, mode='trilinear',align_corners=False))
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == 'batchnorm':
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == 'groupnorm':
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == 'instancenorm':
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != 'none':
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x



class Encoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=False, drop_rate=0.5):
        super(Encoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_one = ConvBlock(1, n_channels, n_filters, normalization=normalization)
        self.out_one = nn.Conv3d(n_filters + n_channels, n_filters, 1, padding=0)
        self.block_one_dw = DownsamplingConvBlock(n_filters, 2 * n_filters, normalization=normalization)

        self.block_two = ConvBlock(2, n_filters * 2, n_filters * 2, normalization=normalization)
        self.out_two = nn.Conv3d(n_filters * 4, n_filters * 2, 1, padding=0)
        self.block_two_dw = DownsamplingConvBlock(n_filters * 2, n_filters * 4, normalization=normalization)

        self.block_three = ConvBlock(3, n_filters * 4, n_filters * 4, normalization=normalization)
        self.out_three = nn.Conv3d(n_filters * 8, n_filters * 4, 1, padding=0)
        self.block_three_dw = DownsamplingConvBlock(n_filters * 4, n_filters * 8, normalization=normalization)

        self.block_four = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.out_four = nn.Conv3d(n_filters * 16, n_filters * 8, 1, padding=0)
        self.block_four_dw = DownsamplingConvBlock(n_filters * 8, n_filters * 16, normalization=normalization)

        self.block_five = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.out_five = nn.Conv3d(n_filters * 32, n_filters * 16, 1, padding=0)
        self.dropout = nn.Dropout3d(p=drop_rate, inplace=False)


    def forward(self, input):
        x1 = self.block_one(input)
        x1 = self.out_one(torch.cat([input, x1], 1))
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2 = self.out_two(torch.cat([x1_dw, x2], 1))
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3 = self.out_three(torch.cat([x2_dw, x3], 1))
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4 = self.out_four(torch.cat([x3_dw, x4], 1))
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)
        x5 = self.out_five(torch.cat([x4_dw, x5], 1))
        # x5 = F.dropout3d(x5, p=0.5, training=True)
        if self.has_dropout:
            x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res

class Decoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=False, drop_rate=0.5):
        super(Decoder, self).__init__()
        self.has_dropout = has_dropout

        self.block_five_up = UpsamplingDeconvBlock(n_filters * 16, n_filters * 8, normalization=normalization)

        self.block_six = ConvBlock(3, n_filters * 16, n_filters * 16, normalization=normalization)
        self.pointwise_six = Pointwise(n_filters * 16, n_filters * 8)
        self.out_six = nn.Conv3d((n_filters * 16 + n_filters * 8), n_filters * 8, 1, padding=0)
        self.block_six_up = UpsamplingDeconvBlock(n_filters * 8, n_filters * 4, normalization=normalization)

        self.block_seven = ConvBlock(3, n_filters * 8, n_filters * 8, normalization=normalization)
        self.pointwise_seven = Pointwise(n_filters * 8, n_filters * 4)
        self.out_seven = nn.Conv3d((n_filters * 8 + n_filters * 4), n_filters * 4, 1, padding=0)
        self.block_seven_up = UpsamplingDeconvBlock(n_filters * 4, n_filters * 2, normalization=normalization)

        self.block_eight = ConvBlock(2, n_filters * 4, n_filters * 4, normalization=normalization)
        self.pointwise_eight = Pointwise(n_filters * 4, n_filters * 2)
        self.out_eight = nn.Conv3d((n_filters * 4 + n_filters * 2), n_filters * 2, 1, padding=0)
        self.block_eight_up = UpsamplingDeconvBlock(n_filters * 2, n_filters, normalization=normalization)

        self.block_nine = ConvBlock(1, n_filters * 2, n_filters * 2, normalization=normalization)
        self.pointwise_nine = Pointwise(n_filters * 2, n_filters * 1)
        self.out_nine = nn.Conv3d((n_filters * 2 + n_filters), n_filters, 1, padding=0)

        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)

        self.dropout = nn.Dropout3d(p=drop_rate, inplace=False)

    def forward(self, features):
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)          #128

        x5_up = torch.cat([x5_up, x4], 1)       #128+128=256
        x6 = self.block_six(x5_up)              #256  3번
        x6 = self.pointwise_six(x6)             #128
        x6 = self.out_six(torch.cat([x5_up, x6], 1))            #256+128
        x6_up = self.block_six_up(x6)

        x6_up = torch.cat([x6_up, x3], 1)  # 4,128,28,28,20
        x7 = self.block_seven(x6_up)  # 4,64,28,28,20
        x7 = self.pointwise_seven(x7)
        x7 = self.out_seven(torch.cat([x6_up, x7], 1))
        x7_up = self.block_seven_up(x7)  # 4,32,56,56,40

        x7_up = torch.cat([x7_up, x2], 1)  # 4,64,56,56,40   #64
        x8 = self.block_eight(x7_up)  # 4,32,56,56,40
        x8 = self.pointwise_eight(x8)
        x8 = self.out_eight(torch.cat([x7_up, x8], 1))
        x8_up = self.block_eight_up(x8)  # 4,16,112,,112,80

        x8_up = torch.cat([x8_up, x1], 1)  # 4,32,112,,112,80        #32
        x9 = self.block_nine(x8_up)  # 4,16,112,,112,80
        x9 = self.pointwise_nine(x9)
        x9 = self.out_nine(torch.cat([x8_up, x9], 1))
        # x9 = F.dropout3d(x9, p=0.5, training=True)
        if self.has_dropout:
            x9 = self.dropout(x9)
        out = self.out_conv(x9)
        return out



class FeatureNoise(nn.Module):
    def __init__(self, uniform_range=0.3):
        super(FeatureNoise, self).__init__()
        self.uni_dist = Uniform(-uniform_range, uniform_range)

    def feature_based_noise(self, x):
        noise_vector = self.uni_dist.sample(x.shape[1:]).to(x.device).unsqueeze(0)
        x_noise = x.mul(noise_vector) + x
        return x_noise

    def forward(self, x):
        x = self.feature_based_noise(x)
        return x





class VNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, n_filters=16, normalization='batchnorm', has_dropout=True, drop_rate=0.3):
        super(VNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.n_filters = n_filters
        self.normalization = normalization
        self.has_dropout = has_dropout
        self.drop_rate = drop_rate

        self.encoder = Encoder(n_channels = self.n_channels,
                               n_classes = self.n_classes,
                               n_filters = self.n_filters,
                               normalization = self.normalization,
                               has_dropout = self.has_dropout,
                               drop_rate = self.drop_rate)
        self.main_decoder = Decoder(n_channels = self.n_channels,
                                   n_classes = self.n_classes,
                                   n_filters = self.n_filters,
                                   normalization = self.normalization,
                                   has_dropout = self.has_dropout,
                                   drop_rate = self.drop_rate)
        #self.aux_decoder1 = Decoder(has_dropout =has_dropout,drop_rate=0.3)
        #self.aux_decoder2 = Decoder(has_dropout =has_dropout,drop_rate=0.0)

        ## perturbations을 진행한 enc output을 conv 3번 진행
        self.pbenc_conv1 = ConvBlock_for_perturb(3, 1, n_filters = n_filters, normalization=normalization)
        self.pbenc_conv2 = ConvBlock_for_perturb(3, 2, n_filters=n_filters, normalization=normalization)
        self.pbenc_conv3 = ConvBlock_for_perturb(3, 4, n_filters=n_filters, normalization=normalization)
        self.pbenc_conv4 = ConvBlock_for_perturb(3, 8, n_filters=n_filters, normalization=normalization)
        self.pbenc_conv5 = ConvBlock_for_perturb(3, 16, n_filters=n_filters, normalization=normalization)

        ## enc output, FD output,noise output 모두 concat한 후의 1*1*1 conv 진행
        self.point_conv5 = nn.Conv3d((n_filters * 16) * 3, (n_filters * 16), 1, padding=0)
        self.point_conv4 = nn.Conv3d((n_filters * 8) * 3, (n_filters * 8) , 1, padding=0)
        self.point_conv3 = nn.Conv3d((n_filters * 4) * 3 , (n_filters * 4), 1, padding=0)
        self.point_conv2 = nn.Conv3d((n_filters * 2) * 3, (n_filters * 2), 1, padding=0)
        self.point_conv1 = nn.Conv3d((n_filters) * 3, (n_filters), 1, padding=0)

    def FeatureDropout(self,x):
        attention = torch.mean(x, dim=1, keepdim=True)
        max_val, _ = torch.max(attention.view(x.size(0), -1), dim=1, keepdim=True)
        threshold = max_val * np.random.uniform(0.7, 0.9)
        threshold = threshold.view(x.size(0), 1, 1, 1, 1).expand_as(attention)
        drop_mask = (attention < threshold).float()
        x = x.mul(drop_mask)
        return x

    def forward(self, input):
        features = self.encoder(input)

        aux1_feature = [FeatureNoise()(i) for i in features]  # perturbation :x1,x2,x3,x4,x5
        aux2_feature = [self.FeatureDropout(i) for i in features]  # perturbation :x1,x2,x3,x4,x5

        aux1_feature[0] = self.pbenc_conv1(aux1_feature[0])
        aux1_feature[1] = self.pbenc_conv2(aux1_feature[1])
        aux1_feature[2] = self.pbenc_conv3(aux1_feature[2])
        aux1_feature[3] = self.pbenc_conv4(aux1_feature[3])
        aux1_feature[4] = self.pbenc_conv5(aux1_feature[4])

        aux2_feature[0] = self.pbenc_conv1(aux2_feature[0])
        aux2_feature[1] = self.pbenc_conv2(aux2_feature[1])
        aux2_feature[2] = self.pbenc_conv3(aux2_feature[2])
        aux2_feature[3] = self.pbenc_conv4(aux2_feature[3])
        aux2_feature[4] = self.pbenc_conv5(aux2_feature[4])

        features_concat = []
        for i in range(5):
            features_concat.append(torch.cat([features[i], aux1_feature[i], aux2_feature[i]], 1))

        features_concat[4] = self.point_conv5(features_concat[4])
        features_concat[3] = self.point_conv4(features_concat[3])
        features_concat[2] = self.point_conv3(features_concat[2])
        features_concat[1] = self.point_conv2(features_concat[1])
        features_concat[0] = self.point_conv1(features_concat[0])


        main_seg = self.main_decoder(features_concat)
        return main_seg

    # def __init_weight(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv3d):
    #             torch.nn.init.kaiming_normal_(m.weight)
    #         elif isinstance(m, nn.BatchNorm3d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()
if __name__ == '__main__':
    # compute FLOPS & PARAMETERS
    from thop import profile
    from thop import clever_format
    model = VNet(n_channels=1, n_classes=2)
    input = torch.randn(4, 1, 112, 112, 80)
    flops, params = profile(model, inputs=(input,))
    print(flops, params)
    macs, params = clever_format([flops, params], "%.3f")
    print(macs, params)
    print("VNet have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
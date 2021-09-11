import torch
import torch.nn as nn
import torch.nn.functional as F

class SKConv(nn.Module):
    def __init__(self, features=3, M=2, r=2, L=32):

        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        print(d)
        self.M = M
        self.features = features
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, y):
        feas = torch.cat([x.unsqueeze_(dim=1), y.unsqueeze_(dim=1)], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class ConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, dilation = 1, bn=False):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, dilation= dilation)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        if self.bn:
            x = self.bn(out)
        out = self.relu(out)
        return out

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(Upsample, self).__init__()
      reflection_padding = kernel_size // 2
      self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)
      self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return self.relu(out)

class MSRB(nn.Module):  # Residual block

    def __init__(self, inplanes):
        super(MSRB, self).__init__()

        self.pading = nn.ReflectionPad2d(1)
        self.conv1 = nn.Conv2d(inplanes, inplanes, kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(inplanes, inplanes, kernel_size=3)

    def forward(self, x):

        out = self.conv1(self.pading(x))
        out = self.relu(out)
        out = self.conv2(self.pading(out))
        out += x
        out = self.relu(out)

        return out

class fusion(nn.Module):
    def __init__(self, in_channels):
        super(fusion, self).__init__()
        self.downsample = ConvLayer(in_channels, 2*in_channels, stride=2, kernel_size=3)
        self.middle = ConvLayer(2 * in_channels, 2 * in_channels, kernel_size=3, stride=1)
        self.upsample = Upsample(2 * in_channels, in_channels, kernel_size=3, stride=2)
        self.softmax = nn.Softmax(dim=-1)
        self.sk = SKConv(features=2*in_channels, M=2, r=2, L=64)

    def forward(self, x, y):
        out = self.downsample(x)
        out = self.sk(out.clone(), y.clone())
        out = self.middle(out)
        out = self.upsample(out)
        out = F.interpolate(out, x.size()[2:], mode='bilinear', align_corners=False)

        return torch.add(out, x)

class dehaze(nn.Module):
    def __init__(self, n=2, n_feats=32):
        super(dehaze, self).__init__()
        self.n_feats = n_feats
        self.pading = nn.ReflectionPad2d(2)
        self.conv2_init_1 = nn.Conv2d(3, 8 * n, kernel_size=5)
        self.conv2_init_2 = nn.Conv2d(8 * n, 16 * n, kernel_size=5)

        self.c1_1 = MSRB(16 * n)
        self.f1_1 = fusion(16 * n)
        self.d1_1 = ConvLayer(16*n, 32*n, kernel_size=3, stride=2)
        self.c1_2 = MSRB(32 * n)
        self.f1_2 = fusion(32 * n)
        self.d1_2 = ConvLayer(32 * n, 64 * n, kernel_size=3, stride=2)
        self.c1_3 = MSRB(64 * n)
        self.f1_3 = fusion(64 * n)
        self.d1_3 = ConvLayer(64 * n, 128 * n, kernel_size=3, stride=2)
        self.c1_4 = MSRB(128 * n)
        self.f1_4 = fusion(128 * n)
        self.d1_4 = ConvLayer(128 * n, 256 * n, kernel_size=3, stride=2)
        self.c1_5 = MSRB(256 * n)

        self.c2_1 = MSRB(16 * n)
        self.f2_1 = fusion(16 * n)
        self.d2_1 = ConvLayer(16 * n, 32 * n, kernel_size=3, stride=2) # not used
        self.c2_2 = MSRB(32 * n)
        self.f2_2 = fusion(32 * n)
        self.d2_2 = ConvLayer(32 * n, 64 * n, kernel_size=3, stride=2) # not used
        self.c2_3 = MSRB(64 * n)
        self.f2_3 = fusion(64 * n)
        self.d2_3 = ConvLayer(64 * n, 128 * n, kernel_size=3, stride=2) # not used
        self.c2_4 = MSRB(128 * n)

        self.c3_1 = MSRB(16 * n)
        self.f3_1 = fusion(16 * n)
        self.d3_1 = ConvLayer(16 * n, 32 * n, kernel_size=3, stride=2) # not used
        self.c3_2 = MSRB(32 * n)
        self.f3_2 = fusion(32 * n)
        self.d3_2 = ConvLayer(32 * n, 64 * n, kernel_size=3, stride=2) # not used
        self.c3_3 = MSRB(64 * n)

        self.c4_1 = MSRB(16 * n)
        self.f4_1 = fusion(16 * n)
        self.d4_1 = ConvLayer(16 * n, 32 * n, kernel_size=3, stride=2) # not used
        self.c4_2 = MSRB(32 * n)

        self.c5_1 = MSRB(16 * n)

        self.Output = ConvLayer(16 * n, 3, kernel_size=3, stride=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        out = self.relu(self.conv2_init_1(self.pading(x)))
        out = self.relu(self.conv2_init_2(self.pading(out)))

        out1 = self.c1_1(out)
        out2 = self.c1_2(self.d1_1(out1))
        out3 = self.c1_3(self.d1_2(out2))
        out4 = self.c1_4(self.d1_3(out3))
        out5 = self.c1_5(self.d1_4(out4))
        out1 = self.f1_1(out1, out2)
        out2 = self.f1_2(out2, out3)
        out3 = self.f1_3(out3, out4)
        out4 = self.f1_4(out4, out5)
        del out5

        out1 = self.c2_1(out1)
        out2 = self.c2_2(out2)
        out3 = self.c2_3(out3)
        out4 = self.c2_4(out4)
        out1 = self.f2_1(out1, out2)
        out2 = self.f2_2(out2, out3)
        out3 = self.f2_3(out3, out4)
        del out4

        out1 = self.c3_1(out1)
        out2 = self.c3_2(out2)
        out3 = self.c3_3(out3)
        out1 = self.f3_1(out1, out2)
        out2 = self.f3_2(out2, out3)
        del out3

        out1 = self.c4_1(out1)
        out2 = self.c4_2(out2)
        out1 = self.f4_1(out1, out2)
        del out2

        out1 = self.c5_1(out1)
        out1 = self.Output(out1)

        return out1
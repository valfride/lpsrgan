import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace
from models import register

from basicsr.archs.rrdbnet_arch import RRDBNet

def ConvBlock(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding='same',
               is_bn=True, is_relu=True, n=2):
    """ Custom function for conv2d:
        Apply 3*3 convolutions with BN and ReLU.
    """
    layers = []
    for i in range(1, n + 1):
        conv = nn.Conv2d(in_channels=in_channels if i == 1 else out_channels, 
                         out_channels=out_channels,
                         kernel_size=kernel_size,
                         stride=stride,
                         padding=padding if padding != 'same' else 'same',
                         bias=not is_bn)  # Disable bias when using BatchNorm
        layers.append(conv)
        
        if is_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        
        if is_relu:
            layers.append(nn.ReLU(inplace=True))
        
    return nn.Sequential(*layers)

def dot_product(seg, cls):
    b, n, h, w = seg.shape
    seg = seg.view(b, n, -1)
    cls = cls.unsqueeze(-1)  # Add an extra dimension for broadcasting
    final = torch.einsum("bik,bi->bik", seg, cls)
    final = final.view(b, n, h, w)
    return final

class UNet3Plus(nn.Module):
    def __init__(self, input_shape, output_channels, deep_supervision=False, cgm=False, training=False):
        super(UNet3Plus, self).__init__()
        self.deep_supervision = deep_supervision
        self.CGM = deep_supervision and cgm
        self.training = training

        self.filters = [64, 128, 256, 512, 1024]
        self.cat_channels = self.filters[0]
        self.cat_blocks = len(self.filters)
        self.upsample_channels = self.cat_blocks * self.cat_channels

        # Encoder
        self.e1 = ConvBlock(input_shape[0], self.filters[0])
        self.e2 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[0], self.filters[1])
        )
        self.e3 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[1], self.filters[2])
        )
        self.e4 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[2], self.filters[3])
        )
        self.e5 = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(self.filters[3], self.filters[4])
        )

        # Classification Guided Module
        self.cgm = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(self.filters[4], 2, kernel_size=1, padding=0),
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Sigmoid()
        ) if self.CGM else None

        # Decoder
        self.d4 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.filters[2], self.cat_channels, n=1),
            ConvBlock(self.filters[3], self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d4_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d3 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.filters[2], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d3_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d2 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.filters[1], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d2_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.d1 = nn.ModuleList([
            ConvBlock(self.filters[0], self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.upsample_channels, self.cat_channels, n=1),
            ConvBlock(self.filters[4], self.cat_channels, n=1)
        ])
        self.d1_conv = ConvBlock(self.upsample_channels, self.upsample_channels, n=1)

        self.final = nn.Conv2d(self.upsample_channels, output_channels, kernel_size=1)

        # Deep Supervision
        self.deep_sup = nn.ModuleList([
                ConvBlock(self.upsample_channels, output_channels, n=1, is_bn=False, is_relu=False)
                for _ in range(3)
            ] + [ConvBlock(self.filters[4], output_channels, n=1, is_bn=False, is_relu=False)]
        ) if self.deep_supervision else None

    def forward(self, x) -> torch.Tensor:
        training = self.training
        # Encoder
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)

        # Classification Guided Module
        if self.CGM:
            cls = self.cgm(e5)
            cls = torch.argmax(cls, dim=1).float()

        # Decoder
        d4 = [
            F.max_pool2d(e1, 8),
            F.max_pool2d(e2, 4),
            F.max_pool2d(e3, 2),
            e4,
            F.interpolate(e5, scale_factor=2, mode='bilinear', align_corners=True)
        ]
        d4 = [conv(d) for conv, d in zip(self.d4, d4)]
        d4 = torch.cat(d4, dim=1)
        d4 = self.d4_conv(d4)

        d3 = [
            F.max_pool2d(e1, 4),
            F.max_pool2d(e2, 2),
            e3,
            F.interpolate(d4, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(e5, scale_factor=4, mode='bilinear', align_corners=True)
        ]
        d3 = [conv(d) for conv, d in zip(self.d3, d3)]
        d3 = torch.cat(d3, dim=1)
        d3 = self.d3_conv(d3)

        d2 = [
            F.max_pool2d(e1, 2),
            e2,
            F.interpolate(d3, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d4, scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(e5, scale_factor=8, mode='bilinear', align_corners=True)
        ]
        d2 = [conv(d) for conv, d in zip(self.d2, d2)]
        d2 = torch.cat(d2, dim=1)
        d2 = self.d2_conv(d2)

        d1 = [
            e1,
            F.interpolate(d2, scale_factor=2, mode='bilinear', align_corners=True),
            F.interpolate(d3, scale_factor=4, mode='bilinear', align_corners=True),
            F.interpolate(d4, scale_factor=8, mode='bilinear', align_corners=True),
            F.interpolate(e5, scale_factor=16, mode='bilinear', align_corners=True)
        ]
        d1 = [conv(d) for conv, d in zip(self.d1, d1)]
        d1 = torch.cat(d1, dim=1)
        d1 = self.d1_conv(d1)
        d1 = self.final(d1)

        outputs = [d1]

        # Deep Supervision
        if self.deep_supervision and training:
            outputs.extend([
                F.interpolate(self.deep_sup[0](d2), scale_factor=2, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[1](d3), scale_factor=4, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[2](d4), scale_factor=8, mode='bilinear', align_corners=True),
                F.interpolate(self.deep_sup[3](e5), scale_factor=16, mode='bilinear', align_corners=True)
            ])

        # Classification Guided Module
        if self.CGM:
            outputs = [dot_product(out, cls) for out in outputs]
        
        outputs = [F.sigmoid(out) for out in outputs]
        
        if self.deep_supervision and training:
            return torch.cat(outputs, dim=0)
        else:
            return outputs[0]

# class UNet3Plus(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # Encoder (Downsample)
#         self.encoder = nn.Sequential(
#             nn.Conv2d(3, 64, 4, stride=2, padding=1),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(128, 256, 4, stride=2, padding=1),
#             nn.InstanceNorm2d(256),
#             nn.LeakyReLU(0.2),
#         )
#         # Decoder (Upsample with skip connections)
#         self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
#             nn.InstanceNorm2d(128),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
#             nn.InstanceNorm2d(64),
#             nn.LeakyReLU(0.2),
#             nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1),
#         )

#     def forward(self, x):
#         x = self.encoder(x)
#         x = self.decoder(x)
#         return x

class ResidualDenseBlock(nn.Module):
    def __init__(self, channels=64, growth_channels=32):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, growth_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels + growth_channels, growth_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(channels + 2*growth_channels, growth_channels, 3, padding=1)
        self.conv4 = nn.Conv2d(channels + 3*growth_channels, growth_channels, 3, padding=1)
        self.conv_out = nn.Conv2d(growth_channels, 2*growth_channels, 3, padding=1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat([x, x1], 1)))
        x3 = self.lrelu(self.conv3(torch.cat([x, x1, x2], 1)))
        x4 = self.lrelu(self.conv4(torch.cat([x, x1, x2, x3], 1)))
        x_out = self.conv_out(x4) + x 
        
        return x_out

class RRDBplus(nn.Module):  
    def __init__(self, channels=32, num_blocks=3):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualDenseBlock(channels=64) for _ in range(num_blocks)])
        self.conv_mid = nn.Conv2d(64, 64, 3, padding=1)
        
    def forward(self, x):
        # x = self.conv_first(x)
        for block in self.blocks:
            x = block(x) + x  # Residual connection
        x = self.conv_mid(x) + x
        
        return x
        
class RRDBNetPlus(nn.Module):
    def __init__(self, num_blocks=23, dropout_prob=0.5):
        super().__init__()
        self.conv_first = nn.Conv2d(3, 64, 3, padding=1)
        self.blocks = nn.ModuleList([RRDBplus() for _ in range(num_blocks+1)])
        self.conv_mid = nn.Conv2d(64, 64, 3, padding=1)
        self.outBlock = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear'),
                nn.Conv2d(64, 64, 3, padding=1),
                nn.Dropout2d(p=dropout_prob),
                nn.Conv2d(64, 3, 3, padding=1)
            )    
        # self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x):
        x = self.conv_first(x)
        conv_first = x
        for block in self.blocks:
            x = block(x) + x  # Residual connection
        x = self.conv_mid(x) + conv_first
        x = self.outBlock(x)
        
        return x

@register('lpsrgan')        
def make(num_blocks, dropout_prob=0.5):
    return RRDBNetPlus(num_blocks=num_blocks, dropout_prob=dropout_prob), UNet3Plus([3, 32, 96], 3, deep_supervision=False, cgm=False,training=True)
    
if __name__ == '__main__':
    x = torch.randn((1, 3, 16, 48))
    netG = RRDBNetPlus(23, 0.5)
    print(netG)
    # netD = UNet3Plus()
    
    netD = UNet3Plus([3, 16, 48], 3, deep_supervision=False, cgm=False,training=True)

    print(netG(x).shape)
    print(netD(x).shape)
    
    
    print('done')

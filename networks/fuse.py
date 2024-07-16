from __future__ import absolute_import, division, print_function

import numpy as np
import torch
import torch.nn as nn
import torchvision

from .convnext import *
from .layers import ConvBlock, upsample, Concat, upsample4
from .dmlpv2 import DMLPv2

from collections import OrderedDict
class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
class Fuse(nn.Module):
    def __init__(self, equi_h, equi_w, invalid_ids=[], pretrained=False, num_classes=13, init_bias=0.0):
        super(Fuse, self).__init__()
        self.num_classes =  num_classes
        self.equi_h = equi_h
        self.equi_w = equi_w
        self.invalid_ids = invalid_ids

        self.bias = nn.Parameter(torch.full([1, num_classes, 1, 1], init_bias))
        # encoder
        self.equi_encoder = convnext_base(pretrained)
        self.proj = nn.Conv2d(3, 128, kernel_size=4, stride=4)##exp
        self.layernorm = LayerNorm(128, eps=1e-6)

        self.num_ch_enc = np.array([128, 128, 256, 512, 1024])  #
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])
        self.equi_dec_convs = OrderedDict()

        self.dmlp = DMLPv2()

        self.equi_dec_convs["deconv_5"] = ConvBlock(self.num_ch_enc[4] + self.num_ch_enc[4], self.num_ch_enc[4])
        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])
        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])#self.num_ch_enc[3]
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])
        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])#self.num_ch_enc[2]
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])
        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])#self.num_ch_enc[1]
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])
        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])#self.num_ch_enc[0]
        # self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[1])##structerd3d   (2)
        # self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[1])##structerd3d   (2)
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])##structerd3d   (1)  stanford
        self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])##structerd3d   (1)   stanford
        self.equi_dec_convs["segconv_0"] = nn.Conv2d(self.num_ch_dec[0], 13,1)##stanford2d3d
        # self.equi_dec_convs["segconv_0"] = nn.Conv2d(self.num_ch_dec[0], 40,1)##structerd3d   (1)
        # self.equi_dec_convs["segconv_0"] = nn.Conv2d(self.num_ch_dec[1], 40,1)##structerd3d  (2)
        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
        self.conv = ConvBlock(self.num_ch_enc[4] + self.num_ch_enc[4], self.num_ch_enc[4])

    def forward(self, input_equi_image,feat4):########
        bs, c, h, w = input_equi_image.shape
        equi_enc_feat0,equi_enc_feat1,equi_enc_feat2,equi_enc_feat3,equi_enc_feat4 = self.equi_encoder(input_equi_image)
        equi_enc_feat0, equi_enc_feat1, equi_enc_feat2, equi_enc_feat3 = self.dmlp(equi_enc_feat0,equi_enc_feat1,equi_enc_feat2,equi_enc_feat3)

        outputs = {}

        equi_x = self.conv(torch.cat([equi_enc_feat4,feat4],dim=1))##use
        equi_x = upsample(self.equi_dec_convs["upconv_5"](equi_x))
        equi_x = torch.cat([equi_x, equi_enc_feat3], 1)
        equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))
        equi_x = torch.cat([equi_x, equi_enc_feat2], 1)
        equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))
        equi_x = torch.cat([equi_x, equi_enc_feat1], 1)
        equi_x = self.equi_dec_convs["deconv_2"](equi_x)
        equi_x = self.equi_dec_convs["upconv_2"](equi_x)
        equi_x = torch.cat([equi_x, equi_enc_feat0], 1)
        equi_x = self.equi_dec_convs["deconv_1"](equi_x)
        equi_x = upsample4(self.equi_dec_convs["upconv_1"](equi_x))
        equi_x = self.equi_dec_convs["deconv_0"](equi_x)
        sem = self.equi_dec_convs["segconv_0"](equi_x)
        sem = self.bias + sem
        sem[:, self.invalid_ids] = -100
        outputs["sem"] = sem
        return outputs

    def deform_proj(self, x):
        # h, w = x.shape[2:]
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          mask=modulator,
                                          stride=4,
                                          )
        return x
import torch
import torch.nn as nn
import numpy as np
import copy

from .convnext import *
from .layers import Conv3x3, ConvBlock, upsample, subpixelconvolution, upsample4
from collections import OrderedDict
from .blocks import Transformer_Block

class Transformer_cascade(nn.Module):
    def __init__(self, emb_dims, num_patch, depth, num_heads):
        super(Transformer_cascade, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(emb_dims, eps=1e-6)
        self.pos_emb = nn.Parameter(torch.zeros(1, num_patch, emb_dims))
        nn.init.trunc_normal_(self.pos_emb, std=.02)
        for _ in range(depth):
            layer = Transformer_Block(emb_dims, num_heads=num_heads)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, x):
        hidden_states = x + self.pos_emb
        for i, layer_block in enumerate(self.layer):
            hidden_states = layer_block(hidden_states)

        encoded = self.encoder_norm(hidden_states)
        return encoded
class Equi_convnext(nn.Module):
    def __init__(self, equi_h, equi_w, pretrained=False, max_depth=10.0, **kwargs):
        super(Equi_convnext, self).__init__()


        self.equi_h = equi_h
        self.equi_w = equi_w
        # encoder
        self.equi_encoder = convnext_base(pretrained)
        self.num_ch_enc = np.array([128, 128, 256, 512, 1024])#
        self.num_ch_dec = np.array([32, 64, 128, 256, 512])
        self.equi_dec_convs = OrderedDict()

        self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])#+256

        self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])

        self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])

        self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])

        self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])

        self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])

        self.equi_dec_convs["depthconv_0"] = Conv3x3(self.num_ch_dec[0], 1)


        self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))


        self.sigmoid = nn.Sigmoid()
        self.max_depth = nn.Parameter(torch.tensor(max_depth), requires_grad=False)

    def forward(self, input_equi_image):
        bs, c, erp_h, erp_w = input_equi_image.shape
        equi_enc_feat0,equi_enc_feat1,equi_enc_feat2,equi_enc_feat3,equi_enc_feat4 = self.equi_encoder(input_equi_image)

        outputs = {}

        equi_x = equi_enc_feat4
        equi_x = upsample(self.equi_dec_convs["upconv_5"](equi_x))

        equi_x = torch.cat([equi_x, equi_enc_feat3], 1)
        equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))

        equi_x = torch.cat([equi_x, equi_enc_feat2], 1)
        equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))#(4,128,64,128)

        equi_x = torch.cat([equi_x, equi_enc_feat1], 1)
        equi_x = self.equi_dec_convs["deconv_2"](equi_x)#(4,128,64,128)
        equi_x =self.equi_dec_convs["upconv_2"](equi_x)

        equi_x = torch.cat([equi_x, equi_enc_feat0], 1)
        equi_x = self.equi_dec_convs["deconv_1"](equi_x)
        equi_x = self.equi_dec_convs["upconv_1"](equi_x)
        equi_x = upsample4(equi_x)
        equi_x = self.equi_dec_convs["deconv_0"](equi_x)
        equi_depth = self.equi_dec_convs["depthconv_0"](equi_x)
        outputs["pred_depth"] = self.max_depth * self.sigmoid(equi_depth)
        outputs["feat4"] = equi_enc_feat4
        # outputs["feat3"] = equi_enc_feat3
        # outputs["feat2"] = equi_enc_feat2
        # outputs["feat1"] = equi_enc_feat1
        # outputs["feat0"] = equi_enc_feat0

        return outputs

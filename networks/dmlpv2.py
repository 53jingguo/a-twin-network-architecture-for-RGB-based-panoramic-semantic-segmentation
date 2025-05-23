import math
from collections import OrderedDict
from functools import partial

import numpy as np

from .layers import ConvBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from timm.models.layers import trunc_normal_, DropPath
from torch import Tensor
from torch.nn import init
from torch.nn.modules.utils import _pair
from .layers import upsample,upsample4


class DWConv2d(nn.Module):
    def __init__(self, in_chans, out_chans, kernel_size=3, padding=1, bias=False):
        super(DWConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_chans, in_chans, kernel_size=kernel_size,
                                   padding=padding, groups=in_chans, bias=bias)
        self.pointwise = nn.Conv2d(in_chans, out_chans, kernel_size=1, bias=bias)

        nn.init.kaiming_uniform_(self.depthwise.weight, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.pointwise.weight, a=math.sqrt(5))

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class GroupNorm(nn.GroupNorm):
    """
    Input: tensor in shape [B, C, H, W]
    """

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class DWConvSeq(nn.Module):
    def __init__(self, dim=768):
        super(DWConvSeq, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


def make_divisible(v, divisor=8, min_value=None):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=nn.Sigmoid, divisor=1, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = nn.Sigmoid()
        reduced_chs = make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x


class SEMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., linear=False,
                 use_se=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        # self.dwconv = DWConv(hidden_features)
        self.dwconv = DWConvSeq(hidden_features)
        self.gamma = nn.Parameter(torch.ones(hidden_features), requires_grad=True)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.linear = linear
        if self.linear:
            self.relu = nn.ReLU(inplace=True)
        self.se = SqueezeExcite(out_features, se_ratio=0.25) if use_se else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        if self.linear:
            x = self.relu(x)
        # import pdb; pdb.set_trace()
        x = self.drop(self.gamma * self.dwconv(x, H, W)) + x
        x = self.fc2(x)
        x = self.drop(x)
        x = self.se(x.permute(0, 2, 1).reshape(B, C, H, W)).reshape(B, C, N).permute(0, 2, 1)
        return x


class DeformableProjEmbed(nn.Module):
    """ feature map to Projected Embedding
    """

    def __init__(self, in_chans=512, emb_chans=128):
        super().__init__()
        self.kernel_size = kernel_size = 3
        self.stride = stride = 1
        self.padding = padding = 1
        self.proj = nn.Conv2d(in_chans, emb_chans, kernel_size=kernel_size, stride=stride,
                              padding=padding)
        self.offset_conv = nn.Conv2d(in_chans, 2 * kernel_size * kernel_size, kernel_size=kernel_size,
                                     stride=stride, padding=padding)
        nn.init.constant_(self.offset_conv.weight, 0.)
        nn.init.constant_(self.offset_conv.bias, 0.)
        self.modulator_conv = nn.Conv2d(in_chans, 1 * kernel_size * kernel_size, kernel_size=kernel_size,
                                        stride=stride, padding=padding)
        nn.init.constant_(self.modulator_conv.weight, 0.)
        nn.init.constant_(self.modulator_conv.bias, 0.)
        self.norm = nn.BatchNorm2d(emb_chans)
        self.act = nn.GELU()

    def deform_proj(self, x):
        # h, w = x.shape[2:]
        max_offset = min(x.shape[-2], x.shape[-1]) // 4
        offset = self.offset_conv(x).clamp(-max_offset, max_offset)
        modulator = 2. * torch.sigmoid(self.modulator_conv(x))
        x = torchvision.ops.deform_conv2d(input=x,
                                          offset=offset,
                                          weight=self.proj.weight,
                                          bias=self.proj.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          )
        return x

    def forward(self, x):
        x = self.deform_proj(x)
        x = self.act(self.norm(x))
        return x


class DeformableMLP(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            bias: bool = True,
    ):
        super(DeformableMLP, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)

        self.offset_modulator_conv = DWConv2d(in_channels, 3 * in_channels)

        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        B, C, H, W = input.size()
        offset_modulator = self.offset_modulator_conv(input)
        offset_y, offset_x, modulator = torch.chunk(offset_modulator, 3, dim=1)
        modulator = 2. * torch.sigmoid(modulator)
        offset = torch.cat((offset_y, offset_x), dim=1)
        max_offset = max(H, W) // 4
        offset = offset.clamp(-max_offset, max_offset)
        x = torchvision.ops.deform_conv2d(input=input,
                                          offset=offset,
                                          weight=self.weight,
                                          bias=self.bias,
                                          padding=self.padding,
                                          mask=modulator,
                                          stride=self.stride,
                                          dilation=self.dilation
                                          )

        x = self.act(self.norm(x))
        return x

    def extra_repr(self) -> str:
        # s = self.__class__.__name__ + '('
        s = ''
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        # s += ')'
        return s.format(**self.__dict__)


class DeformableMLPBlock(nn.Module):
    def __init__(self, in_chans=512, emb_chans=64, drop_path=0.):
        super().__init__()
        # spatial deformable proj
        self.sdp = DeformableProjEmbed(in_chans=in_chans, emb_chans=emb_chans)
        self.dmlp = DeformableMLP(emb_chans, emb_chans)
        self.cmlp1 = SEMlp(emb_chans)
        self.cmlp2 = SEMlp(emb_chans)
        h, w = 3, 3
        self.norm1 = GroupNorm(emb_chans)
        self.pooling = nn.AvgPool2d((h, w), stride=1, padding=(h // 2, w // 2), count_include_pad=False)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = self.sdp(x)
        B, C, H, W = x.shape
        x_ = x.reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.cmlp1(x_, H, W)
        x_ = x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = x_ + self.pooling(self.norm1(x))

        x_ = x.reshape(B, C, -1).permute(0, 2, 1)
        x_ = self.cmlp2(x_, H, W)
        x_ = x_.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        x = x_ + self.drop_path(self.dmlp(x))

        return x


class DMLPv2(nn.Module):
    def __init__(self,
                 in_channels=[128, 256, 512],
                 num_classes=13,
                 dropout_ratio=0.1,
                 embed_dim=768,
                 norm_layer=nn.BatchNorm2d,
                 align_corners=False):
        super().__init__()
        self.num_classes = num_classes
        self.align_corners = align_corners

        self.in_channels = in_channels

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        c1_in_channels, c2_in_channels, c3_in_channels = self.in_channels

        embedding_dim = embed_dim
        self.head0 = DeformableMLPBlock(in_chans=c1_in_channels, emb_chans=c1_in_channels)#embedding_dim)
        self.head1 = DeformableMLPBlock(in_chans=c1_in_channels, emb_chans=c1_in_channels)#embedding_dim)
        self.head2 = DeformableMLPBlock(in_chans=c2_in_channels, emb_chans=c2_in_channels)#embedding_dim)
        self.head3 = DeformableMLPBlock(in_chans=c3_in_channels, emb_chans=c3_in_channels)#embedding_dim)
        # self.head4 = DeformableMLPBlock(in_chans=c4_in_channels, emb_chans=c4_in_channels)
        # self.head5 = DeformableMLPBlock(in_chans=c4_in_channels, emb_chans=c4_in_channels)

        # self.linear_fuse = nn.Sequential(
        #     nn.Conv2d(in_channels=embedding_dim * 4, out_channels=embedding_dim, kernel_size=1),
        #     norm_layer(embedding_dim),
        #     nn.ReLU(inplace=True)
        # )
        # self.equi_dec_convs = OrderedDict()
        # self.pred = nn.Conv2d(embedding_dim, self.num_classes, kernel_size=1)
        # # self.pred = nn.Conv2d(32, self.num_classes, kernel_size=1)
        # self.num_ch_enc = np.array([128, 128, 256, 512, 1024])  #
        # self.num_ch_dec = np.array([32, 64, 128, 256, 512])
        # self.equi_dec_convs["deconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_enc[4])
        # self.equi_dec_convs["upconv_5"] = ConvBlock(self.num_ch_enc[4], self.num_ch_dec[4])
        # self.equi_dec_convs["deconv_4"] = ConvBlock(self.num_ch_dec[4] + self.num_ch_enc[3], self.num_ch_dec[4])
        # self.equi_dec_convs["upconv_4"] = ConvBlock(self.num_ch_dec[4], self.num_ch_dec[3])
        # self.equi_dec_convs["deconv_3"] = ConvBlock(self.num_ch_dec[3] + self.num_ch_enc[2], self.num_ch_dec[3])
        # self.equi_dec_convs["upconv_3"] = ConvBlock(self.num_ch_dec[3], self.num_ch_dec[2])
        # self.equi_dec_convs["deconv_2"] = ConvBlock(self.num_ch_dec[2] + self.num_ch_enc[1], self.num_ch_dec[2])
        # self.equi_dec_convs["upconv_2"] = ConvBlock(self.num_ch_dec[2], self.num_ch_dec[1])
        # self.equi_dec_convs["deconv_1"] = ConvBlock(self.num_ch_dec[1] + self.num_ch_enc[0], self.num_ch_dec[1])
        # self.equi_dec_convs["upconv_1"] = ConvBlock(self.num_ch_dec[1], self.num_ch_dec[0])
        # self.equi_dec_convs["deconv_0"] = ConvBlock(self.num_ch_dec[0], self.num_ch_dec[0])
        # self.equi_decoder = nn.ModuleList(list(self.equi_dec_convs.values()))
        # self.conv0 = nn.Conv2d(1024, 64, 1)
        # self.conv1 = nn.Conv2d(1024, 128, 1)
        # self.conv2 = nn.Conv2d(1024, 256, 1)
        # self.conv3 = nn.Conv2d(1024, 512, 1)
        # self.conv00 = nn.Conv2d(128, 64, 1)
        # self.gelu0 = nn.GELU()
        # self.gelu1 = nn.GELU()
        # self.gelu2 = nn.GELU()
        # self.gelu3 = nn.GELU()
        # self.gelu4 = nn.GELU()
        # self.w = nn.Parameter(torch.ones(3))
        # self.q = nn.Parameter(torch.ones(3))
        # self.e = nn.Parameter(torch.ones(3))
        # self.r = nn.Parameter(torch.ones(3))
        # self.t = nn.Parameter(torch.ones(3))


    def forward(self, c0, c1, c2, c3):
        # f4 = c4
        # f3 = c3
        # f2 = c2
        # f1 = c1
        # f0 = c0
        # c5 = self.head4(c5)
        #
        # c4 = self.head4(c4)
        c3 = self.head3(c3)
        c2 = self.head2(c2)
        c1 = self.head1(c1)
        c0 = self.head0(c0)
        # equi_x = torch.cat([feat4, f4], 1)
        # equi_x = self.equi_dec_convs["deconv_5"](f4)
        # t0 = torch.exp(self.t[0] / torch.sum(torch.exp(self.t)))
        # t1 = torch.exp(self.t[1] / torch.sum(torch.exp(self.t)))
        # t2 = torch.exp(self.t[2] / torch.sum(torch.exp(self.t)))
        # equi_x = self.gelu1(t0*equi_x + t1*c4 + t2*feat4)
        # equi_x = upsample(self.equi_dec_convs["upconv_5"](equi_x))
        # equi_x = torch.cat([equi_x, feat3], 1)
        # equi_x = self.equi_dec_convs["deconv_4"](equi_x)
        # q0 = torch.exp(self.q[0] / torch.sum(torch.exp(self.q)))
        # q1 = torch.exp(self.q[1] / torch.sum(torch.exp(self.q)))
        # q2 = torch.exp(self.q[2] / torch.sum(torch.exp(self.q)))
        # equi_x = self.gelu1(q0*equi_x + q1*c3 + q2*f3)
        # equi_x = upsample(self.equi_dec_convs["upconv_4"](equi_x))
        # equi_x = torch.cat([equi_x, feat2], 1)
        # equi_x = self.equi_dec_convs["deconv_3"](equi_x)
        # w0 = torch.exp(self.w[0] / torch.sum(torch.exp(self.w)))
        # w1 = torch.exp(self.w[1] / torch.sum(torch.exp(self.w)))
        # w2 = torch.exp(self.w[2] / torch.sum(torch.exp(self.w)))
        # equi_x = self.gelu2(w0*equi_x + w1*c2 + w2*f2)
        # equi_x = upsample(self.equi_dec_convs["upconv_3"](equi_x))
        # equi_x = torch.cat([equi_x, feat1], 1)
        # equi_x = self.equi_dec_convs["deconv_2"](equi_x)
        # e0 = torch.exp(self.e[0] / torch.sum(torch.exp(self.e)))
        # e1 = torch.exp(self.e[1] / torch.sum(torch.exp(self.e)))
        # e2 = torch.exp(self.e[2] / torch.sum(torch.exp(self.e)))
        # equi_x = self.gelu3(e0*equi_x + e1*c1 + e2*f1)
        # equi_x = self.equi_dec_convs["upconv_2"](equi_x)
        # equi_x = torch.cat([equi_x, feat0], 1)
        # equi_x = self.equi_dec_convs["deconv_1"](equi_x)
        # r0 = torch.exp(self.r[0] / torch.sum(torch.exp(self.r)))
        # r1 = torch.exp(self.r[1] / torch.sum(torch.exp(self.r)))
        # r2 = torch.exp(self.r[2] / torch.sum(torch.exp(self.r)))
        # equi_x = self.gelu4(r0*equi_x + r1*c0 + r2*self.conv00(f0))
        # equi_x = upsample4(self.equi_dec_convs["upconv_1"](equi_x))
        # out = self.equi_dec_convs["deconv_0"](equi_x)

        ############## DMLP decoder on C1-C4 ###########
        # c4 = self.head4(c4)
        # c4 = F.interpolate(c4, size=(512,1024), mode='bilinear', align_corners=self.align_corners)
        #
        # c3 = self.head3(c3)
        # c3 = F.interpolate(c3, size=(512,1024), mode='bilinear', align_corners=self.align_corners)
        #
        # c2 = self.head2(c2)
        # c2 = F.interpolate(c2, size=(512,1024), mode='bilinear', align_corners=self.align_corners)
        #
        # c1 = self.head1(c1)
        #
        # c1 = F.interpolate(c1, size=(512,1024), mode='bilinear', align_corners=self.align_corners)
        # c0 = F.interpolate(c0, size=(512,1024), mode='bilinear', align_corners=self.align_corners)
        # out = self.linear_fuse(torch.cat([c1, c2, c3, c0], dim=1))
        # out = self.dropout(out)
        # out = self.pred(out)

        return c0, c1, c2, c3

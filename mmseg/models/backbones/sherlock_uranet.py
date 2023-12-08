# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import (UPSAMPLE_LAYERS, ConvModule, build_activation_layer,
                      build_norm_layer)
from mmcv.runner import BaseModule
from mmcv.utils.parrots_wrapper import _BatchNorm

from mmseg.ops import Upsample
from ..builder import BACKBONES
from ..utils import UpConvBlock, SherlockUpConvBlock

# sherlock
import torch
import torch.nn.functional as F


def vertical_slice(input_tensor):
    """
    Args:
        input_tensor (torch.Tensor): 输入特征图，大小为[n, w/2, c]
        
    Returns:
        list of torch.Tensor: 竖直切片得到的子特征图列表，每个子特征图大小为[1, w/2, c]
    """
    # 获取输入特征图的高度维度
    n, _, c = input_tensor.shape
    
    # 初始化子特征图列表
    sliced_tensors = []
    
    # 逐一切片
    for i in range(n):
        sliced_tensor = input_tensor[i:i+1, :, :]
        sliced_tensors.append(sliced_tensor)
    
    return sliced_tensors

class RAU(nn.Module):
    def __init__(self, in_channels, n, reduction=16):
        super(RAU, self).__init__()
        self.combine_conv = nn.Conv2d(in_channels * 9, in_channels, kernel_size=1)
        self.conv_out = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()
    
    def upsample_and_subtract(self, original_tensor, sliced_tensors):
        """
        Args:
            original_tensor (torch.Tensor): 原始输入特征图，大小为 [batch, channels, h, w]
            sliced_tensors (list of torch.Tensor): 竖直切片得到的子特征图列表，每个子特征图大小为 [batch, channels, h/2, w]
            
        Returns:
            torch.Tensor: 结果特征图，大小为 [batch, channels, h, w]
        """
        # 获取原始特征图的大小
        h, w = original_tensor.size(2), original_tensor.size(3)
        
        # 初始化结果特征图
        #result_tensor = torch.zeros(original_tensor.size())
        result_tensor = original_tensor
        
        # 逐一处理每个子特征图
        for sliced_tensor in sliced_tensors:
            # 上采样子特征图至原始大小
            upsampled_tensor = F.interpolate(sliced_tensor, size=(h, w), mode='bilinear', align_corners=False)
            
            # 与原始特征图相减
            diff_tensor = original_tensor - upsampled_tensor
            
            # 将差异结果累加到结果特征图上
            result_tensor = torch.cat((result_tensor,diff_tensor), 1)
        
        return result_tensor
    
    def forward(self, x):

        pooled_tensor = F.avg_pool2d(x, kernel_size=(1, 2), stride=(1, 2))
        pooled_tensor = torch.nn.functional.adaptive_avg_pool2d(pooled_tensor, output_size=(8, pooled_tensor.size(3)))
    
        b, _, _, c = pooled_tensor.shape
        sliced_tensors = []
        
        # 逐一切片
        n = 8
        sliced_tensors = pooled_tensor.chunk(8, dim=2)


        result_tensor = self.upsample_and_subtract(x, sliced_tensors)
        combined_out = self.combine_conv(result_tensor)
        return combined_out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        
        # 第一个卷积层（F(X)）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        # 第二个卷积层（Wi）
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x  # 上采样操作 WSX
        
        out = self.conv1(x)  # F(X)
        out = self.bn1(out)
        out = self.relu1(out)
        
        out = self.conv2(out)  # Wi
        out = self.bn2(out)
        
        residual = self.conv3(residual)
        residual = self.bn3(residual)
        out += residual  # 残差连接
        out = self.relu2(out)
        
        return out
    

class PositionAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(PositionAttentionModule, self).__init__()
        # 使用两个不同的1x1卷积核生成f(X)和g(X)
        self.f_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.g_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.h_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        f_x = self.f_conv(x)
        g_x = self.g_conv(x)
        h_x = self.h_conv(x)

        # 重组f(X)和g(X)为 [RC', N]
        batch_size, channels, height, width = f_x.size()
        f_x = f_x.view(batch_size, channels, -1)  # [B, C, N]
        g_x = g_x.view(batch_size, channels, -1)  # [B, C, N]

        # 计算位置注意力矩阵S
        attention_map = torch.matmul(f_x.permute(0, 2, 1), g_x)  # [B, N, N]
        attention_map = F.softmax(attention_map, dim=-1)

        # 重组得到特征O
        position_output = torch.matmul(h_x.view(batch_size, channels, -1), attention_map.permute(0, 2, 1))
        position_output = position_output.view(batch_size, channels, height, width)
        position_output += x 

        return position_output


class ChannelAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        # 通过卷积操作将特征图X映射为通道注意力矩阵S'
        batch_size, channels, height, width = x.size()
        f_x = x.view(batch_size, channels, -1)  # [B, C, N]
        g_x = x.view(batch_size, channels, -1)  # [B, C, N]
        att = torch.matmul(f_x.permute(0, 2, 1), g_x)  # [B, N, N]
        att = F.softmax(att, dim=-1)            # [B, N, N]
        outp = torch.matmul(f_x, att)
        outp = outp.view(batch_size, channels, height, width)
        # 将通道注意力矩阵S'应用于特征图X
        channel_output = x + outp  
        
        return channel_output


class DualAttentionModule(nn.Module):
    def __init__(self, in_channels):
        super(DualAttentionModule, self).__init__()
        self.conv = nn.Conv2d(2*in_channels, in_channels, kernel_size=1)
        self.position_attention = PositionAttentionModule(in_channels)
        self.channel_attention = ChannelAttentionModule(in_channels)
        

    def forward(self, x):
        position_output = self.position_attention(x)
        channel_output = self.channel_attention(x)
       
        pAdc = torch.cat((position_output,channel_output),1)
        y = self.conv(pAdc)  # 特征融合
        
        return y


class UNetEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=1, n_rau_blocks=1):
        super(UNetEncoderLayer, self).__init__()
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_channels, out_channels) for _ in range(1)
        ])
        
        self.rau_blocks = nn.ModuleList([
            RAU(out_channels, n_rau_blocks) for _ in range(1)
        ])
        
    def forward(self, x):
        # 逐一应用ResidualBlock和RAU
        for residual_block, rau_block in zip(self.residual_blocks, self.rau_blocks):
            x = residual_block(x)
            x = rau_block(x)
        return x
    
class UNetAttnEncoderLayer(nn.Module):
    def __init__(self, in_channels, out_channels, num_residual_blocks=1, n_rau_blocks=1):
        super(UNetAttnEncoderLayer, self).__init__()
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(in_channels, out_channels) for _ in range(1)
        ])
        
        self.rau_blocks = nn.ModuleList([
            RAU(out_channels, n_rau_blocks) for _ in range(1)
        ])

        self.att_blocks = nn.ModuleList([
            DualAttentionModule(out_channels) for _ in range(1)
        ])
        
    def forward(self, x):
        # 逐一应用ResidualBlock和RAU
        for residual_block, rau_block, att_block in zip(self.residual_blocks, self.rau_blocks, self.att_blocks):
            x = residual_block(x)
            x = rau_block(x)
            x = att_block(x)
        return x

class BasicConvBlock(nn.Module):
    """Basic convolutional block for UNet.

    This module consists of several plain convolutional layers.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_convs (int): Number of convolutional layers. Default: 2.
        stride (int): Whether use stride convolution to downsample
            the input feature map. If stride=2, it only uses stride convolution
            in the first convolutional layer to downsample the input feature
            map. Options are 1 or 2. Default: 1.
        dilation (int): Whether use dilated convolution to expand the
            receptive field. Set dilation rate of each convolutional layer and
            the dilation rate of the first convolutional layer is always 1.
            Default: 1.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_convs=2,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 dcn=None,
                 plugins=None):
        super(BasicConvBlock, self).__init__()
        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'

        self.with_cp = with_cp
        convs = []
        for i in range(num_convs):
            convs.append(
                ConvModule(
                    in_channels=in_channels if i == 0 else out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=stride if i == 0 else 1,
                    dilation=1 if i == 0 else dilation,
                    padding=1 if i == 0 else dilation,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))

        self.convs = nn.Sequential(*convs)

    def forward(self, x):
        """Forward function."""

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(self.convs, x)
        else:
            out = self.convs(x)
        return out

@BACKBONES.register_module()
class URANet(BaseModule):
    """UNet backbone.

    This backbone is the implementation of `U-Net: Convolutional Networks
    for Biomedical Image Segmentation <https://arxiv.org/abs/1505.04597>`_.

    Args:
        in_channels (int): Number of input image channels. Default" 3.
        base_channels (int): Number of base channels of each stage.
            The output channels of the first stage. Default: 64.
        num_stages (int): Number of stages in encoder, normally 5. Default: 5.
        strides (Sequence[int 1 | 2]): Strides of each stage in encoder.
            len(strides) is equal to num_stages. Normally the stride of the
            first stage in encoder is 1. If strides[i]=2, it uses stride
            convolution to downsample in the correspondence encoder stage.
            Default: (1, 1, 1, 1, 1).
        enc_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence encoder stage.
            Default: (2, 2, 2, 2, 2).
        dec_num_convs (Sequence[int]): Number of convolutional layers in the
            convolution block of the correspondence decoder stage.
            Default: (2, 2, 2, 2).
        downsamples (Sequence[int]): Whether use MaxPool to downsample the
            feature map after the first stage of encoder
            (stages: [1, num_stages)). If the correspondence encoder stage use
            stride convolution (strides[i]=2), it will never use MaxPool to
            downsample, even downsamples[i-1]=True.
            Default: (True, True, True, True).
        enc_dilations (Sequence[int]): Dilation rate of each stage in encoder.
            Default: (1, 1, 1, 1, 1).
        dec_dilations (Sequence[int]): Dilation rate of each stage in decoder.
            Default: (1, 1, 1, 1).
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
        conv_cfg (dict | None): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict | None): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict | None): Config dict for activation layer in ConvModule.
            Default: dict(type='ReLU').
        upsample_cfg (dict): The upsample config of the upsample module in
            decoder. Default: dict(type='InterpConv').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        dcn (bool): Use deformable convolution in convolutional layer or not.
            Default: None.
        plugins (dict): plugins for convolutional layers. Default: None.
        pretrained (str, optional): model pretrained path. Default: None
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None

    Notice:
        The input image size should be divisible by the whole downsample rate
        of the encoder. More detail of the whole downsample rate can be found
        in UNet._check_input_divisible.
    """

    def __init__(self,
                 in_channels=3,
                 base_channels=64,
                 num_stages=5,
                 strides=(1, 1, 1, 1, 1),
                 enc_num_convs=(2, 2, 2, 2, 2),
                 dec_num_convs=(2, 2, 2, 2),
                 downsamples=(True, True, True, True),
                 enc_dilations=(1, 1, 1, 1, 1),
                 dec_dilations=(1, 1, 1, 1),
                 with_cp=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 upsample_cfg=dict(type='InterpConv'),
                 norm_eval=False,
                 dcn=None,
                 plugins=None,
                 pretrained=None,
                 init_cfg=None):
        super(URANet, self).__init__(init_cfg)

        self.pretrained = pretrained
        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            if init_cfg is None:
                self.init_cfg = [
                    dict(type='Kaiming', layer='Conv2d'),
                    dict(
                        type='Constant',
                        val=1,
                        layer=['_BatchNorm', 'GroupNorm'])
                ]
        else:
            raise TypeError('pretrained must be a str or None')

        assert dcn is None, 'Not implemented yet.'
        assert plugins is None, 'Not implemented yet.'
        assert len(strides) == num_stages, \
            'The length of strides should be equal to num_stages, '\
            f'while the strides is {strides}, the length of '\
            f'strides is {len(strides)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_num_convs) == num_stages, \
            'The length of enc_num_convs should be equal to num_stages, '\
            f'while the enc_num_convs is {enc_num_convs}, the length of '\
            f'enc_num_convs is {len(enc_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_num_convs) == (num_stages-1), \
            'The length of dec_num_convs should be equal to (num_stages-1), '\
            f'while the dec_num_convs is {dec_num_convs}, the length of '\
            f'dec_num_convs is {len(dec_num_convs)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(downsamples) == (num_stages-1), \
            'The length of downsamples should be equal to (num_stages-1), '\
            f'while the downsamples is {downsamples}, the length of '\
            f'downsamples is {len(downsamples)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(enc_dilations) == num_stages, \
            'The length of enc_dilations should be equal to num_stages, '\
            f'while the enc_dilations is {enc_dilations}, the length of '\
            f'enc_dilations is {len(enc_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        assert len(dec_dilations) == (num_stages-1), \
            'The length of dec_dilations should be equal to (num_stages-1), '\
            f'while the dec_dilations is {dec_dilations}, the length of '\
            f'dec_dilations is {len(dec_dilations)}, and the num_stages is '\
            f'{num_stages}.'
        self.num_stages = num_stages
        self.strides = strides
        self.downsamples = downsamples
        self.norm_eval = norm_eval
        self.base_channels = base_channels

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        for i in range(num_stages):
            enc_conv_block = []
            if i != 0:
                if i != 4:
                    if strides[i] == 1 and downsamples[i - 1]:
                        enc_conv_block.append(nn.MaxPool2d(kernel_size=2))
                    upsample = (strides[i] != 1 or downsamples[i - 1])
                    self.decoder.append(
                        SherlockUpConvBlock(
                            conv_block=BasicConvBlock,
                            in_channels=base_channels * 2**i,
                            skip_channels=base_channels * 2**(i - 1),
                            out_channels=base_channels * 2**(i - 1),
                            num_convs=dec_num_convs[i - 1],
                            stride=1,
                            dilation=dec_dilations[i - 1],
                            with_cp=with_cp,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            upsample_cfg=upsample_cfg if upsample else None,
                            dcn=None,
                            plugins=None))
                else: 
                    if strides[i] == 1 and downsamples[i - 1]:
                        enc_conv_block.append(nn.MaxPool2d(kernel_size=2, 
                                              ))
                    upsample = (strides[i] != 1 or downsamples[i - 1])
                    self.decoder.append(
                        SherlockUpConvBlock(
                            conv_block=BasicConvBlock,
                            in_channels=base_channels * 2**i,
                            skip_channels=base_channels * 2**(i - 1),
                            out_channels=base_channels * 2**(i - 1),
                            num_convs=dec_num_convs[i - 1],
                            stride=1,
                            dilation=dec_dilations[i - 1],
                            with_cp=with_cp,
                            conv_cfg=conv_cfg,
                            norm_cfg=norm_cfg,
                            act_cfg=act_cfg,
                            upsample_cfg=upsample_cfg if upsample else None,
                            dcn=None,
                            plugins=None)
                            )
            if i != 4:
                enc_conv_block.append(
                    UNetEncoderLayer(in_channels=in_channels,
                        out_channels=base_channels * 2**i)
                )
                self.encoder.append((nn.Sequential(*enc_conv_block)))
                in_channels = base_channels * 2**i
            else: 
                enc_conv_block.append(
                    UNetAttnEncoderLayer(in_channels=in_channels,
                        out_channels=base_channels * 2**i)
                )
                self.encoder.append((nn.Sequential(*enc_conv_block)))
                in_channels = base_channels * 2**i

    def forward(self, x):
        #self._check_input_divisible(x)  # sherlock: 检查输入是否可被下采样;
        enc_outs = []
        for enc in self.encoder:
            x = enc(x)
            enc_outs.append(x)
        dec_outs = [x]
        for i in reversed(range(len(self.decoder))):
            x = self.decoder[i](enc_outs[i], x)
            dec_outs.append(x)

        return dec_outs

    def train(self, mode=True):
        """Convert the model into training mode while keep normalization layer
        freezed."""
        super(URANet, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()

    def _check_input_divisible(self, x):
        h, w = x.shape[-2:]
        whole_downsample_rate = 1
        for i in range(1, self.num_stages):
            if self.strides[i] == 2 or self.downsamples[i - 1]:
                whole_downsample_rate *= 2
        assert (h % whole_downsample_rate == 0) \
            and (w % whole_downsample_rate == 0),\
            f'The input image size {(h, w)} should be divisible by the whole '\
            f'downsample rate {whole_downsample_rate}, when num_stages is '\
            f'{self.num_stages}, strides is {self.strides}, and downsamples '\
            f'is {self.downsamples}.'

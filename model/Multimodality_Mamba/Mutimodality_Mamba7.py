import math
from typing import Sequence

import mmengine
import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import PatchEmbed
from mmengine.model import ModuleList
from mmengine.model.weight_init import trunc_normal_
import torch.nn.functional as F

from utils import seed_torch

from mmpretrain.models.utils.position_encoding import build_2d_sincos_position_embedding

from transformers.models.mamba.modeling_mamba import MambaMixer

from mmpretrain.registry import MODELS
from mmpretrain.models.utils.helpers import to_2tuple
from mmpretrain.models.utils.norm import build_norm_layer
from mmpretrain.models.backbones.base_backbone import BaseBackbone
from thop import profile
from thop import clever_format
from model.changer import SpatialExchange, ChannelExchange, ChannelInsert

class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=(1, 1), stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.sgap = nn.AvgPool2d(2)  # kernelsize

    def forward(self, x):
        # B, H, W, C = x.shape
        # x = x.view(B, C, H, W)
        B, C, H, W = x.shape

        mx = torch.max(x, 1)[0].unsqueeze(1)
        avg = torch.mean(x, 1).unsqueeze(1)
        combined = torch.cat([mx, avg], dim=1)
        fmap = self.conv(combined)
        weight_map = torch.sigmoid(fmap)
        out = (x * weight_map).mean(dim=(-2, -1))

        return out, x * weight_map

class TokenLearner(nn.Module):
    def __init__(self, S) -> None:
        super().__init__()
        self.S = S
        self.tokenizers = nn.ModuleList([SpatialAttention() for _ in range(S)])

    def forward(self, x):
        # B, _, _, C = x.shape
        B, C, _, _ = x.shape
        Z = torch.Tensor(B, self.S, C).to("cuda:0")
        for i in range(self.S):
            Ai, _ = self.tokenizers[i](x)  # [B, C]
            Z[:, i, :] = Ai
        return Z

class _NonLocalBlockND(nn.Module):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 dimension=3,
                 sub_sample=False,
                 bn_layer=True):
        super(_NonLocalBlockND, self).__init__()
        assert dimension in [1, 2, 3]

        self.dimension = dimension
        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2

            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = nn.Conv3d
            max_pool_layer = nn.MaxPool3d(kernel_size=(1, 2, 2))
            bn = nn.BatchNorm3d
        elif dimension == 2:
            conv_nd = nn.Conv2d
            max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2))
            bn = nn.BatchNorm2d
        else:
            conv_nd = nn.Conv1d
            max_pool_layer = nn.MaxPool1d(kernel_size=(2))
            bn = nn.BatchNorm1d

        self.g = conv_nd(in_channels=self.in_channels,
                         out_channels=self.inter_channels,
                         kernel_size=1,
                         stride=1,
                         padding=0)

        if bn_layer:
            self.W = nn.Sequential(
                conv_nd(in_channels=self.inter_channels,
                        out_channels=self.in_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0), bn(self.in_channels))
            nn.init.constant_(self.W[1].weight, 0)
            nn.init.constant_(self.W[1].bias, 0)
        else:
            self.W = conv_nd(in_channels=self.inter_channels,
                             out_channels=self.in_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
            nn.init.constant_(self.W.weight, 0)
            nn.init.constant_(self.W.bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels,
                             out_channels=self.inter_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0)
        self.phi = conv_nd(in_channels=self.in_channels,
                           out_channels=self.inter_channels,
                           kernel_size=1,
                           stride=1,
                           padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x, y, z, return_nl_map=False):
        batch_size = x.size(0)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_y = self.phi(y).view(batch_size, self.inter_channels, -1)

        f = torch.matmul(theta_x, phi_y)
        f_div_C = F.softmax(f, dim=-1)

        g_z = self.g(z).view(batch_size, self.inter_channels, -1)
        g_z = g_z.permute(0, 2, 1)
        o = torch.matmul(f_div_C, g_z)
        o = o.permute(0, 2, 1).contiguous()
        o = o.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(o)
        z = W_y + z
        if return_nl_map:
            return z, f_div_C
        return z

class NONLocalBlock2D(_NonLocalBlockND):
    def __init__(self,
                 in_channels,
                 inter_channels=None,
                 sub_sample=True,
                 bn_layer=True):
        super(NONLocalBlock2D, self).__init__(
            in_channels,
            inter_channels=inter_channels,
            dimension=2,
            sub_sample=sub_sample,
            bn_layer=bn_layer,
        )


class hsiMamba(BaseBackbone):
    arch_zoo = {
        **dict.fromkeys(
            ['gv1', 'globalview1'], {
                'embed_dims': 144,
                'num_layers': 1,
                'feedforward_channels': 144//2,
            }),
        **dict.fromkeys(
            ['gf1', 'globalfeature1'], {
                'embed_dims': 256,#用过768//4和768//8 正常是768//4和768//2  现在是64//2和64//4
                'num_layers': 1,#12*2   5 12 1 24
                'feedforward_channels': 256//2 #768//8  原先搞错了一直是64来着  ； 本来是64//4
            }),
        **dict.fromkeys(
            ['gv1', 'globalview2'], {
                'embed_dims': 256,
                'num_layers': 1,
                'feedforward_channels': 256 // 2,
            }),
        **dict.fromkeys(
            ['gf1', 'globalfeature2'], {
                'embed_dims': 144,  # 用过768//4和768//8 正常是768//4和768//2  现在是64//2和64//4
                'num_layers': 1,  # 12*2   5 12 1 24
                'feedforward_channels': 144 // 2  # 768//8  原先搞错了一直是64来着  ； 本来是64//4
            }),
    }
    OUT_TYPES = {'featmap', 'avg_featmap', 'cls_token', 'raw'}

    def __init__(self,
                 arch='base',
                 pe_type='learnable',#position embedding
                 # 'forward', 'forward_reverse_mean', 'forward_reverse_gate', 'forward_reverse_shuffle_gate'
                 path_type='forward_reverse_shuffle_gate',
                 cls_position='none',  # 'head', 'tail', 'head_tail', 'middle', 'none'
                 img_size=224,
                 patch_size=16,
                 in_channels=3,
                 out_indices=-1,#？？？？？
                 drop_rate=0.,
                 norm_cfg=dict(type='LN', eps=1e-6),
                 final_norm=True,
                 out_type='avg_featmap',
                 frozen_stages=-1,
                 interpolate_mode='bicubic',
                 patch_cfg=dict(),
                 layer_cfgs=dict(),
                 init_cfg=None):
        super(hsiMamba, self).__init__(init_cfg)
        # 获取架构信息
        self.arch = arch
        if isinstance(arch, str):
            arch = arch.lower()
            assert arch in set(self.arch_zoo), \
                f'Arch {arch} is not in default archs {set(self.arch_zoo)}'
            self.arch_settings = self.arch_zoo[arch]
        else:
            essential_keys = {
                'embed_dims', 'num_layers', 'feedforward_channels'
            }
            assert isinstance(arch, dict) and essential_keys <= set(arch), \
                f'Custom arch needs a dict with keys {essential_keys}'#自定义架构
            self.arch_settings = arch

        self.embed_dims = self.arch_settings['embed_dims']#786//4
        self.num_layers = self.arch_settings['num_layers']#12*2
        self.img_size = to_2tuple(img_size)#(224, 224)
        self.cls_position = cls_position#None
        self.path_type = path_type

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=patch_size,###不是stride= patchsize的一半吗，for overlap
            padding=0,
            bias=False,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(f'Unsupported `out_type` {out_type}, please '
                             f'choose from {self.OUT_TYPES}')
        self.out_type = out_type

        self.num_extra_tokens = 0
        # Set cls token
        if cls_position != 'none':
            if cls_position == 'head_tail':
                self.cls_token = nn.Parameter(torch.zeros(1, 2, self.embed_dims))
                self.num_extra_tokens = 2
            else:
                self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
                self.num_extra_tokens = 1
        else:
            self.cls_token = None

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pe_type = pe_type
        if pe_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_extra_tokens, self.embed_dims))
        elif pe_type == 'sine':
            self.pos_embed = build_2d_sincos_position_embedding(
                patches_resolution=self.patch_resolution,
                embed_dims=self.embed_dims,
                temperature=10000,
                cls_token=False)
            # TODO: add cls token
        else:
            self.pos_embed = None

        self.drop_after_pos = nn.Dropout(p=drop_rate)###

        if isinstance(out_indices, int):######？？？？？？？？？？？？
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), \
            f'"out_indices" must by a sequence or int, ' \
            f'get {type(out_indices)} instead.'
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index#23
            assert 0 <= out_indices[i] <= self.num_layers, \
                f'Invalid out_indices {index}'
        self.out_indices = out_indices

        self.layers = ModuleList()
        self.gate_layers = ModuleList()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                hidden_size=self.embed_dims,
                state_size=16,
                intermediate_size=self.arch_settings.get('feedforward_channels', self.embed_dims * 2),
                conv_kernel=4,
                time_step_rank=math.ceil(self.embed_dims / 16),
                use_conv_bias=True,
                hidden_act="silu",
                use_bias=False,
            )
            _layer_cfg.update(layer_cfgs[i])
            _layer_cfg = mmengine.Config(_layer_cfg)
            self.layers.append(MambaMixer(_layer_cfg, i))
            if 'gate' in self.path_type:
                gate_out_dim = 2
                if 'shuffle' in self.path_type:
                    gate_out_dim = 3
                if 'eight' in self.path_type:
                    gate_out_dim = 8
                if 'clock' in self.path_type:
                    gate_out_dim = 2
                self.gate_layers.append(
                    nn.Sequential(
                        nn.Linear(gate_out_dim*self.embed_dims, gate_out_dim, bias=False),
                        nn.Softmax(dim=-1)
                    )
                )

        # self.gate_layers_for_tokens = nn.Sequential(
        #     nn.Linear(49 * self.embed_dims, 49, bias=False),
        #     nn.Softmax(dim=-1)
        # )


        self.frozen_stages = frozen_stages
        self.pre_norm = build_norm_layer(norm_cfg, self.embed_dims)

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = build_norm_layer(norm_cfg, self.embed_dims)
        if self.out_type == 'avg_featmap':
            self.ln2 = build_norm_layer(norm_cfg, self.embed_dims)
        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

        self.weights = nn.Parameter(torch.zeros(1, 10, 1))
        H = self.patch_resolution[0] - 2
        self.tokenlearner = TokenLearner(H*H)
        self.ln3 = build_norm_layer(norm_cfg, self.embed_dims)

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def init_weights(self):
        super(hsiMamba, self).init_weights()

        if not (isinstance(self.init_cfg, dict) and self.init_cfg['type'] == 'Pretrained'):
            if self.pos_embed is not None:
                trunc_normal_(self.pos_embed, std=0.02)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False
        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
            if 'gate' in self.path_type:
                m = self.gate_layers[i - 1]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == 'avg_featmap':
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            if self.cls_position == 'head':
                x = torch.cat((cls_token, x), dim=1)
            elif self.cls_position == 'tail':
                x = torch.cat((x, cls_token), dim=1)
            elif self.cls_position == 'head+tail':
                x = torch.cat((cls_token[:, :1], x, cls_token[:, 1:]), dim=1)
            elif self.cls_position == 'middle':
                x = torch.cat((x[:, :x.size(1) // 2], cls_token, x[:, x.size(1) // 2:]), dim=1)
            else:
                raise ValueError(f'Invalid cls_position {self.cls_position}')

        if self.pos_embed is not None:
            x = x + self.pos_embed.to(device=x.device)

        x = self.drop_after_pos(x)

        outs = []
        for i, layer in enumerate(self.layers):
            residual = x
            if 'forward' == self.path_type:
                x = self.pre_norm(x.to(dtype=self.pre_norm.weight.dtype))
                x = layer(x)
            if 'shuffle' == self.path_type:
                rand_index = torch.randperm(x.size(1))
                x = x[:, rand_index]
                x = self.pre_norm(x.to(dtype=self.pre_norm.weight.dtype))
                x = layer(x)
                rand_index = torch.argsort(rand_index)
                x = x[:, rand_index]  # 回原顺序
            if 'eight_directions_gate' == self.path_type:  # for input_size=(b, 49, dim_embedding)
                index_vf = torch.tensor([0, 7, 14, 21, 28, 35, 42, 1, 8, 15, 22, 29, 36, 43, 2, 9, 16, \
                                         23, 30, 37, 44, 3, 10, 17, 24, 31, 38, 45, 4, 11, 18, 25, 32, \
                                         39, 46, 5, 12, 19, 26, 33, 40, 47, 6, 13, 20, 27, 34, 41, 48])  # index of direction: vertical_forward
                index_vr = torch.flip(index_vf, dims=[0])  # index of direction: vertical_reverse
                index_37df = torch.tensor([0, 1, 7, 2, 8, 14, 3, 9, 15, 21, 4, 10, 16, 22, 28, 5, 11, \
                                           17, 23, 29, 35, 6, 12, 18, 24, 30, 36, 42, 13, 19, 25, 31, \
                                           37, 43, 20, 26, 32, 38, 44, 27, 33, 39, 45, 34, 40, 46, 41, 47, 48])  # index of direction: 37diagonal_forward(signal from 1-9)
                index_37dr = torch.flip(index_37df, dims=[0])  # index of direction: 37diagonal_reverse
                index_19df = torch.tensor([6, 5, 13, 4, 12, 20, 3, 11, 19, 27, 2, 10, 18,26, 34, 1, 9, 17, \
                                           25, 33, 41, 0, 8, 16, 24, 32, 40, 48, 7, 15, 23, 31, 39, 47, 14, \
                                           22, 30, 38, 46, 21, 29, 37, 45, 28, 36, 44, 35, 43,42])  # index of direction: 19diagonal_reverse
                index_19dr = torch.flip(index_19df, dims=[0])  # index of direction: 19diagonal_reverse

                d_hf = x  # direction: horizontal_forward
                d_hr = torch.flip(x, [1])  # direction: horizontal_reverse
                d_vf = x[:, index_vf]
                d_vr = x[:, index_vr]
                d_37df = x[:, index_37df]
                d_37dr = x[:, index_37dr]
                d_19df = x[:, index_19df]
                d_19dr = x[:, index_19dr]
                x_inputs = [d_hf, d_hr, d_vf, d_vr, d_37df, d_37dr, d_19df, d_19dr]
                x_inputs = torch.cat(x_inputs, dim=0)

                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))  # 先合在一起做归一化、mamba mixer
                x_inputs = layer(x_inputs)  # （8B， 9,128）mambermixer 是按照每大组做的吗，还是所有batch的samples之间会有互动

                x_hf, x_hr, x_vf, x_vr, x_37df, x_37dr, x_19df, x_19dr = torch.split(x_inputs, B, dim=0)
                # 回去 reverse index
                index_vf = torch.argsort(index_vf)
                index_vr = torch.argsort(index_vr)
                index_37df = torch.argsort(index_37df)
                index_37dr = torch.argsort(index_37dr)
                index_19df = torch.argsort(index_19df)
                index_19dr = torch.argsort(index_19dr)

                x_hr = torch.flip(x_hr, [1])
                x_vf = x_vf[:, index_vf]
                x_vr = x_vr[:, index_vr]
                x_37df = x_37df[:, index_37df]
                x_37dr = x_37dr[:, index_37dr]
                x_19df = x_19df[:, index_19df]
                x_19dr = x_19dr[:, index_19dr]

                # mean_x_hf = torch.mean(x_hf, dim=1)  # （1，196,192）---》（1， 192）
                # mean_x_hr = torch.mean(x_hr, dim=1)
                # mean_x_vf = torch.mean(x_vf, dim=1)
                # mean_x_vr = torch.mean(x_vr, dim=1)
                # mean_x_37df = torch.mean(x_37df, dim=1)
                # mean_x_37dr = torch.mean(x_37dr, dim=1)
                # mean_x_19df = torch.mean(x_19df, dim=1)
                # mean_x_19dr = torch.mean(x_19dr, dim=1)
                #
                # gate = torch.cat(
                #     [mean_x_hf, mean_x_hr, mean_x_vf, mean_x_vr, mean_x_37df, mean_x_37dr, mean_x_19df, mean_x_19dr],
                #     dim=-1)  # （1， 192*8）
                # gate = self.gate_layers[i](gate)  # （1， 192*8）--》（1， 8）计算量有点高且不合理
                # gate = gate.unsqueeze(-1)  # （1， 8， 1）
                gate = self.weights
                x = gate[:, 0:1] * x_hf + gate[:, 1:2] * x_hr + gate[:, 2:3] * x_vf + gate[:, 3:4] * x_vr + \
                    gate[:, 4:5] * x_37df + gate[:, 5:6] * x_37dr + gate[:, 6:7] * x_19df + gate[:, 7:8] * x_19dr
            if '81twoclock' == self.path_type: # for input_size=(b, 81, dim_embedding)
                index_ltcw = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 26, 35, 44, 53, 62,
                                           71, 80, 79, 78, 77, 76, 75, 74, 73, 72, 63, 54, 45,
                                           36, 27, 18, 9, 10, 11, 12, 13, 14, 15, 16, 25, 34,
                                           43, 52, 61, 70, 69, 68, 67, 66, 65, 64, 55, 46, 37,
                                           28, 19, 20, 21, 22, 23, 24, 33, 42, 51, 60, 59, 58,
                                           57, 56, 47, 38, 29, 30, 31, 32, 41, 50, 49, 48, 39,
                                           40])  # index of direction: start from left-top(0) in clockwise
                index_ltacw = torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 72, 73, 74, 75, 76,
                                            77, 78, 79, 80, 71, 62, 53, 44, 35, 26, 17, 8, 7,
                                            6, 5, 4, 3, 2, 1, 10, 19, 28, 37, 46, 55, 64, 65,
                                            66, 67, 68, 69, 70, 61, 52, 43, 34, 25, 16, 15, 14,
                                            13, 12, 11, 20, 29, 38, 47, 56, 57, 58, 59, 60, 51,
                                            42, 33, 24, 23, 22, 21, 30, 39, 48, 49, 50, 41, 32, 31,
                                            40])  # index of direction: start from left-top(0) in anticlockwise
                # index_rtcw = torch.tensor([6, 13, 20, 27, 34, 41, 48, 47, 46, 45, 44, 43, 42, 35, \
                #                            28, 21, 14, 7, 0, 1, 2, 3, 4, 5, 12, 19, 26, 33, 40, 39, \
                #                            38, 37, 36, 29, 22, 15, 8, 9, 10, 11, 18, 25, 32, 31, 30, 23, 16, 17, 24])#index of direction: start from right-top(6) in clockwise
                # index_rtacw = torch.tensor([6, 5, 4, 3, 2, 1, 0, 7, 14, 21, 28, 35, 42, 43, 44, 45, \
                #                             46, 47, 48, 41, 34, 27, 20, 13, 12, 11, 10, 9, 8, 15, 22, \
                #                             29, 36, 37, 38, 39, 40, 33, 26, 19, 18, 17, 16, 23, 30, 31, 32, 25, 24])#index of direction: start from right-top(6) in anticlockwise
                # index_ldcw = torch.tensor([42, 35, 28, 21, 14, 7, 0, 1, 2, 3, 4, 5, 6, 13, 20, 27, \
                #                            34, 41, 48, 47, 46, 45, 44, 43, 36, 29, 22, 15, 8, 9, 10, \
                #                            11, 12, 19, 26, 33, 40, 39, 38, 37, 30, 23, 16, 17, 18, 25, 32, 31, 24])#index of direction: start from left-down(42) in clockwise
                # index_ldacw = torch.tensor([42, 43, 44, 45, 46, 47, 48, 41, 34, 27, 20, 13, 6, 5, 4, 3, \
                #                             2, 1, 0, 7, 14, 21, 28, 35, 36, 37, 38, 39, 40, 33, 26, 19, \
                #                             12, 11, 10, 9, 8, 15, 22, 29, 30, 31, 32, 25, 18, 17, 16, 23, 24])#index of direction: start from left-down(42) in anticlockwise
                # index_rdcw = torch.tensor([48, 47, 46, 45, 44, 43, 42, 35, 28, 21, 14, 7, 0, 1, 2, 3, \
                #                            4, 5, 6, 13, 20, 27, 34, 41, 40, 39, 38, 37, 36, 29, 22, 15, \
                #                            8, 9, 10, 11, 12, 19, 26, 33, 32, 31, 30, 23, 16, 17, 18, 25, 24])#index of direction: start from right-down(48) in clockwise
                # index_rdacw = torch.tensor([48, 41, 34, 27, 20, 13, 6, 5, 4, 3, 2, 1, 0, 7, 14, 21, 28, \
                #                             35, 42, 43, 44, 45, 46, 47, 40, 33, 26, 19, 12, 11, 10, 9, \
                #                             8, 15, 22, 29, 36, 37, 38, 39, 32, 25, 18, 17, 16, 23, 30, 31, 24])#index of direction: start from right-down(48) in anticlockwise

                d_ltcw = x[:, index_ltcw]
                d_ltacw = x[:, index_ltacw]
                # d_rtcw = x[:, index_rtcw]
                # d_rtacw = x[:, index_rtacw]
                # d_ldcw = x[:, index_ldcw]
                # d_ldacw = x[:, index_ldacw]
                # d_rdcw = x[:, index_rdcw]
                # d_rdacw = x[:, index_rdacw]

                # x_inputs = [d_ltcw, d_ltacw, d_rtcw, d_rtacw, d_ldcw, d_ldacw, d_rdcw, d_rdacw]
                # x_inputs = [d_ltcw, d_ltacw, d_rdcw, d_rdacw]
                x_inputs = [d_ltcw, d_ltacw]
                x_inputs = torch.cat(x_inputs, dim=0)

                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))  # 先合在一起做归一化、mamba mixer
                x_inputs = layer(x_inputs)  # （8B， 9,128）mambermixer 是按照每大组做的吗，还是所有batch的samples之间会有互动

                # x_ltcw, x_ltacw, x_rtcw, x_rtacw, x_ldcw, x_ldacw, x_rdcw, x_rdacw = torch.split(x_inputs, B, dim=0)
                # x_ltcw, x_ltacw, x_rdcw, x_rdacw = torch.split(x_inputs, B, dim=0)
                x_ltcw, x_ltacw = torch.split(x_inputs, B, dim=0)

                # 回去 reverse index
                index_ltcw = torch.argsort(index_ltcw)
                index_ltacw = torch.argsort(index_ltacw)
                # index_rtcw = torch.argsort(index_rtcw)
                # index_rtacw = torch.argsort(index_rtacw)
                # index_ldcw = torch.argsort(index_ldcw)
                # index_ldacw = torch.argsort(index_ldacw)
                # index_rdcw = torch.argsort(index_rdcw)
                # index_rdacw = torch.argsort(index_rdacw)

                x_ltcw = x_ltcw[:, index_ltcw]
                x_ltacw = x_ltacw[:, index_ltacw]
                # x_rtcw = x_rtcw[:, index_rtcw]
                # x_rtacw = x_rtacw[:, index_rtacw]
                # x_ldcw = x_ldcw[:, index_ldcw]
                # x_ldacw = x_ldacw[:, index_ldacw]
                # x_rdcw = x_rdcw[:, index_rdcw]
                # x_rdacw = x_rdacw[:, index_rdacw]

                # mean_x_hf = torch.mean(x_hf, dim=1)  # （1，196,192）---》（1， 192）
                # mean_x_hr = torch.mean(x_hr, dim=1)
                # mean_x_vf = torch.mean(x_vf, dim=1)
                # mean_x_vr = torch.mean(x_vr, dim=1)
                # mean_x_37df = torch.mean(x_37df, dim=1)
                # mean_x_37dr = torch.mean(x_37dr, dim=1)
                # mean_x_19df = torch.mean(x_19df, dim=1)
                # mean_x_19dr = torch.mean(x_19dr, dim=1)
                #
                # gate = torch.cat([mean_x_hf, mean_x_hr, mean_x_vf, mean_x_vr, mean_x_37df, mean_x_37dr, mean_x_19df, mean_x_19dr], dim=-1)  # （1， 192*8）
                # gate = self.gate_layers[i](gate)  # （1， 192*8）--》（1， 8）计算量有点高且不合理
                # gate = gate.unsqueeze(-1)  # （1， 8， 1）
                gate = self.weights
                gate = F.softmax(gate, dim=1)
                # x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw + gate[:, 2:3] * x_rtcw + gate[:, 3:4] * x_rtacw  + \
                #     gate[:, 4:5] * x_ldcw + gate[:, 5:6] * x_ldacw + gate[:, 6:7] * x_rdcw + gate[:, 7:8] * x_rdacw
                # x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw + gate[:, 2:3] * x_rdcw + gate[:, 3:4] * x_rdacw
                x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw
            if '81_2+8' == self.path_type:  # for input_size=(b, 81, dim_embedding)
                index_vf = torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 72, 73, 64, 55, 46, 37, 28, 19, 10, 1, \
                                         2, 11, 20, 29, 38, 47, 56, 65, 74, 75, 66, 57, 48, 39, 30, 21, 12, 3, \
                                         4, 13, 22, 31, 40, 49, 58, 67, 76, 77, 68, 59, 50, 41, 32, 23, 14, 5, \
                                         6, 15, 24, 33,42, 51, 60, 69, 78, 79, 70, 61, 52, 43, 34, 25, 16, 7, \
                                         8, 17, 26, 35, 44, 53, 62, 71, 80])  # index of direction: vertical_forward
                index_vr = torch.flip(index_vf, dims=[0])  # index of direction: vertical_reverse
                index_37df = torch.tensor([0, 1, 9, 18, 10, 2, 3, 11, 19, 27, 36, 28, 20, 12, 4, 5, 13, 21, 29, \
                                           37, 45, 54, 46, 38, 30, 22, 14, 6, 7, 15, 23, 31, 39, 47, 55, 63, 72, \
                                           64, 56, 48, 40, 32, 24, 16, 8, 17, 25, 33, 41, 49, 57, 65, 73, 74, 66, \
                                           58, 50, 42, 34, 26, 35, 43, 51, 59, 67, 75, 76, 68, 60, 52, 44, 53, 61, \
                                           69, 77, 78, 70, 62, 71, 79, 80])  # index of direction: 37diagonal_forward(signal from 1-9)
                index_37dr = torch.flip(index_37df, dims=[0])  # index of direction: 37diagonal_reverse
                index_19df = torch.tensor([8, 7, 17, 26, 16, 6, 5, 15, 25, 35, 44, 34, 24, 14, 4, 3, 13, 23, 33, \
                                           43, 53, 62, 52,42, 32, 22, 12, 2, 1, 11, 21, 31, 41, 51, 61, 71, 80, \
                                           70, 60, 50, 40, 30, 20, 10, 0, 9, 19, 29, 39, 49, 59, 69, 79, 78, 68, \
                                           58, 48, 38, 28, 18, 27, 37, 47, 57, 67, 77, 76, 66, 56, 46, 36, 45, 55, \
                                           65, 75, 74, 64, 54, 63, 73, 72])  # index of direction: 19diagonal_reverse
                index_19dr = torch.flip(index_19df, dims=[0])  # index of direction: 19diagonal_reverse
                index_ltcw = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 17, 26, 35, 44, 53, 62,
                                           71, 80, 79, 78, 77, 76, 75, 74, 73, 72, 63, 54, 45,
                                           36, 27, 18, 9, 10, 11, 12, 13, 14, 15, 16, 25, 34,
                                           43, 52, 61, 70, 69, 68, 67, 66, 65, 64, 55, 46, 37,
                                           28, 19, 20, 21, 22, 23, 24, 33, 42, 51, 60, 59, 58,
                                           57, 56, 47, 38, 29, 30, 31, 32, 41, 50, 49, 48, 39,
                                           40])  # index of direction: start from left-top(0) in clockwise
                index_ltacw = torch.tensor([0, 9, 18, 27, 36, 45, 54, 63, 72, 73, 74, 75, 76,
                                            77, 78, 79, 80, 71, 62, 53, 44, 35, 26, 17, 8, 7,
                                            6, 5, 4, 3, 2, 1, 10, 19, 28, 37, 46, 55, 64, 65,
                                            66, 67, 68, 69, 70, 61, 52, 43, 34, 25, 16, 15, 14,
                                            13, 12, 11, 20, 29, 38, 47, 56, 57, 58, 59, 60, 51,
                                            42, 33, 24, 23, 22, 21, 30, 39, 48, 49, 50, 41, 32, 31,
                                            40])  # index of direction: start from left-top(0) in anticlockwise

                d_hf = x  # direction: horizontal_forward
                d_hr = torch.flip(x, [1])  # direction: horizontal_reverse
                d_vf = x[:, index_vf]
                d_vr = x[:, index_vr]
                d_37df = x[:, index_37df]
                d_37dr = x[:, index_37dr]
                d_19df = x[:, index_19df]
                d_19dr = x[:, index_19dr]
                d_ltcw = x[:, index_ltcw]
                d_ltacw = x[:, index_ltacw]

                x_inputs = [d_hf, d_hr, d_vf, d_vr, d_37df, d_37dr, d_19df, d_19dr, d_ltcw, d_ltacw]
                x_inputs = torch.cat(x_inputs, dim=0)

                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))  # 先合在一起做归一化、mamba mixer
                x_inputs = layer(x_inputs)  # （8B， 9,128）mambermixer 是按照每大组做的吗，还是所有batch的samples之间会有互动

                x_hf, x_hr, x_vf, x_vr, x_37df, x_37dr, x_19df, x_19dr, x_ltcw, x_ltacw = torch.split(x_inputs, B, dim=0)

                # 回去 reverse index
                index_vf = torch.argsort(index_vf)
                index_vr = torch.argsort(index_vr)
                index_37df = torch.argsort(index_37df)
                index_37dr = torch.argsort(index_37dr)
                index_19df = torch.argsort(index_19df)
                index_19dr = torch.argsort(index_19dr)
                index_ltcw = torch.argsort(index_ltcw)
                index_ltacw = torch.argsort(index_ltacw)

                x_hr = torch.flip(x_hr, [1])
                x_vf = x_vf[:, index_vf]
                x_vr = x_vr[:, index_vr]
                x_37df = x_37df[:, index_37df]
                x_37dr = x_37dr[:, index_37dr]
                x_19df = x_19df[:, index_19df]
                x_19dr = x_19dr[:, index_19dr]
                x_ltcw = x_ltcw[:, index_ltcw]
                x_ltacw = x_ltacw[:, index_ltacw]


                # mean_x_hf = torch.mean(x_hf, dim=1)  # （1，196,192）---》（1， 192）
                # mean_x_hr = torch.mean(x_hr, dim=1)
                # mean_x_vf = torch.mean(x_vf, dim=1)
                # mean_x_vr = torch.mean(x_vr, dim=1)
                # mean_x_37df = torch.mean(x_37df, dim=1)
                # mean_x_37dr = torch.mean(x_37dr, dim=1)
                # mean_x_19df = torch.mean(x_19df, dim=1)
                # mean_x_19dr = torch.mean(x_19dr, dim=1)
                #
                # gate = torch.cat([mean_x_hf, mean_x_hr, mean_x_vf, mean_x_vr, mean_x_37df, mean_x_37dr, mean_x_19df, mean_x_19dr], dim=-1)  # （1， 192*8）
                # gate = self.gate_layers[i](gate)  # （1， 192*8）--》（1， 8）计算量有点高且不合理
                # gate = gate.unsqueeze(-1)  # （1， 8， 1）
                gate = self.weights
                gate = F.softmax(gate, dim=1)
                # x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw + gate[:, 2:3] * x_rtcw + gate[:, 3:4] * x_rtacw  + \
                #     gate[:, 4:5] * x_ldcw + gate[:, 5:6] * x_ldacw + gate[:, 6:7] * x_rdcw + gate[:, 7:8] * x_rdacw
                # x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw + gate[:, 2:3] * x_rdcw + gate[:, 3:4] * x_rdacw
                x = gate[:, 0:1] * x_hf + gate[:, 1:2] * x_hr + gate[:, 2:3] * x_vf + gate[:, 3:4] * x_vr + \
                    gate[:, 4:5] * x_37df + gate[:, 5:6] * x_37dr + gate[:, 6:7] * x_19df + gate[:, 7:8] * x_19dr + \
                    gate[:, 8:9] * x_ltcw + gate[:, 9:10] * x_ltacw
            if '49twoclock' == self.path_type: # for input_size=(b, 9, dim_embedding)
                index_ltcw = torch.tensor([0, 1, 2, 3, 4, 5, 6, 13, 20, 27, 34, 41, 48, 47, 46, 45, 44, 43, 42, \
                                         35, 28, 21, 14, 7, 8, 9, 10, 11, 12, 19, 26, 33, \
                                         40, 39, 38, 37, 36, 29, 22, 15, 16, 17, 18, 25, \
                                         32, 31, 30, 23, 24]) #index of direction: start from left-top(0) in clockwise
                index_ltacw = torch.tensor([0, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46,47, 48, 41, 34, \
                                         27, 20, 13, 6, 5, 4, 3, 2, 1, 8, 15, 22, 29, 36, 37, 38, 39, \
                                         40, 33, 26, 19, 12, 11, 10, 9, 16, 23, 30, 31, 32, 25, 18, 17, 24]) #index of direction: start from left-top(0) in anticlockwise
                # index_rtcw = torch.tensor([6, 13, 20, 27, 34, 41, 48, 47, 46, 45, 44, 43, 42, 35, \
                #                            28, 21, 14, 7, 0, 1, 2, 3, 4, 5, 12, 19, 26, 33, 40, 39, \
                #                            38, 37, 36, 29, 22, 15, 8, 9, 10, 11, 18, 25, 32, 31, 30, 23, 16, 17, 24])#index of direction: start from right-top(6) in clockwise
                # index_rtacw = torch.tensor([6, 5, 4, 3, 2, 1, 0, 7, 14, 21, 28, 35, 42, 43, 44, 45, \
                #                             46, 47, 48, 41, 34, 27, 20, 13, 12, 11, 10, 9, 8, 15, 22, \
                #                             29, 36, 37, 38, 39, 40, 33, 26, 19, 18, 17, 16, 23, 30, 31, 32, 25, 24])#index of direction: start from right-top(6) in anticlockwise
                # index_ldcw = torch.tensor([42, 35, 28, 21, 14, 7, 0, 1, 2, 3, 4, 5, 6, 13, 20, 27, \
                #                            34, 41, 48, 47, 46, 45, 44, 43, 36, 29, 22, 15, 8, 9, 10, \
                #                            11, 12, 19, 26, 33, 40, 39, 38, 37, 30, 23, 16, 17, 18, 25, 32, 31, 24])#index of direction: start from left-down(42) in clockwise
                # index_ldacw = torch.tensor([42, 43, 44, 45, 46, 47, 48, 41, 34, 27, 20, 13, 6, 5, 4, 3, \
                #                             2, 1, 0, 7, 14, 21, 28, 35, 36, 37, 38, 39, 40, 33, 26, 19, \
                #                             12, 11, 10, 9, 8, 15, 22, 29, 30, 31, 32, 25, 18, 17, 16, 23, 24])#index of direction: start from left-down(42) in anticlockwise
                # index_rdcw = torch.tensor([48, 47, 46, 45, 44, 43, 42, 35, 28, 21, 14, 7, 0, 1, 2, 3, \
                #                            4, 5, 6, 13, 20, 27, 34, 41, 40, 39, 38, 37, 36, 29, 22, 15, \
                #                            8, 9, 10, 11, 12, 19, 26, 33, 32, 31, 30, 23, 16, 17, 18, 25, 24])#index of direction: start from right-down(48) in clockwise
                # index_rdacw = torch.tensor([48, 41, 34, 27, 20, 13, 6, 5, 4, 3, 2, 1, 0, 7, 14, 21, 28, \
                #                             35, 42, 43, 44, 45, 46, 47, 40, 33, 26, 19, 12, 11, 10, 9, \
                #                             8, 15, 22, 29, 36, 37, 38, 39, 32, 25, 18, 17, 16, 23, 30, 31, 24])#index of direction: start from right-down(48) in anticlockwise

                d_ltcw = x[:, index_ltcw]
                d_ltacw = x[:, index_ltacw]
                # d_rtcw = x[:, index_rtcw]
                # d_rtacw = x[:, index_rtacw]
                # d_ldcw = x[:, index_ldcw]
                # d_ldacw = x[:, index_ldacw]
                # d_rdcw = x[:, index_rdcw]
                # d_rdacw = x[:, index_rdacw]

                # x_inputs = [d_ltcw, d_ltacw, d_rtcw, d_rtacw, d_ldcw, d_ldacw, d_rdcw, d_rdacw]
                # x_inputs = [d_ltcw, d_ltacw, d_rdcw, d_rdacw]
                x_inputs = [d_ltcw, d_ltacw]
                x_inputs = torch.cat(x_inputs, dim=0)

                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))  # 先合在一起做归一化、mamba mixer
                x_inputs = layer(x_inputs)  # （8B， 9,128）mambermixer 是按照每大组做的吗，还是所有batch的samples之间会有互动

                # x_ltcw, x_ltacw, x_rtcw, x_rtacw, x_ldcw, x_ldacw, x_rdcw, x_rdacw = torch.split(x_inputs, B, dim=0)
                # x_ltcw, x_ltacw, x_rdcw, x_rdacw = torch.split(x_inputs, B, dim=0)
                x_ltcw, x_ltacw = torch.split(x_inputs, B, dim=0)

                # 回去 reverse index
                index_ltcw = torch.argsort(index_ltcw)
                index_ltacw = torch.argsort(index_ltacw)
                # index_rtcw = torch.argsort(index_rtcw)
                # index_rtacw = torch.argsort(index_rtacw)
                # index_ldcw = torch.argsort(index_ldcw)
                # index_ldacw = torch.argsort(index_ldacw)
                # index_rdcw = torch.argsort(index_rdcw)
                # index_rdacw = torch.argsort(index_rdacw)

                x_ltcw = x_ltcw[:, index_ltcw]
                x_ltacw = x_ltacw[:, index_ltacw]
                # x_rtcw = x_rtcw[:, index_rtcw]
                # x_rtacw = x_rtacw[:, index_rtacw]
                # x_ldcw = x_ldcw[:, index_ldcw]
                # x_ldacw = x_ldacw[:, index_ldacw]
                # x_rdcw = x_rdcw[:, index_rdcw]
                # x_rdacw = x_rdacw[:, index_rdacw]

                # mean_x_hf = torch.mean(x_hf, dim=1)  # （1，196,192）---》（1， 192）
                # mean_x_hr = torch.mean(x_hr, dim=1)
                # mean_x_vf = torch.mean(x_vf, dim=1)
                # mean_x_vr = torch.mean(x_vr, dim=1)
                # mean_x_37df = torch.mean(x_37df, dim=1)
                # mean_x_37dr = torch.mean(x_37dr, dim=1)
                # mean_x_19df = torch.mean(x_19df, dim=1)
                # mean_x_19dr = torch.mean(x_19dr, dim=1)
                #
                # gate = torch.cat([mean_x_hf, mean_x_hr, mean_x_vf, mean_x_vr, mean_x_37df, mean_x_37dr, mean_x_19df, mean_x_19dr], dim=-1)  # （1， 192*8）
                # gate = self.gate_layers[i](gate)  # （1， 192*8）--》（1， 8）计算量有点高且不合理
                # gate = gate.unsqueeze(-1)  # （1， 8， 1）
                gate = self.weights
                gate = F.softmax(gate, dim=1)
                # x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw + gate[:, 2:3] * x_rtcw + gate[:, 3:4] * x_rtacw  + \
                #     gate[:, 4:5] * x_ldcw + gate[:, 5:6] * x_ldacw + gate[:, 6:7] * x_rdcw + gate[:, 7:8] * x_rdacw
                # x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw + gate[:, 2:3] * x_rdcw + gate[:, 3:4] * x_rdacw
                x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw
            if '49_2+8' == self.path_type: # for input_size=(b, 9, dim_embedding)
                index_vf = torch.tensor([0, 7, 14, 21, 28, 35, 42, 43, 36, 29, 22, 15, 8, 1, 2, 9, 16, 23, 30,
                                         37, 44, 45, 38, 31, 24, 17, 10, 3, 4, 11, 18, 25, 32, 39, 46, 47, 40,
                                         33, 26, 19, 12, 5, 6, 13, 20, 27, 34, 41, 48])  # index of direction: vertical_forward
                index_vr = torch.flip(index_vf, dims=[0])  # index of direction: vertical_reverse
                index_37df = torch.tensor([0, 1, 7, 14, 8, 2, 3, 9, 15, 21, 28, 22, 16, 10, 4, 5, 11, 17, 23,
                                           29, 35, 42, 36, 30, 24, 18, 12, 6, 13, 19, 25, 31, 37, 43, 44, 38,
                                           32, 26, 20, 27, 33, 39, 45, 46, 40, 34, 41, 47, 48])  # index of direction: 37diagonal_forward(signal from 1-9)
                index_37dr = torch.flip(index_37df, dims=[0])  # index of direction: 37diagonal_reverse
                index_19df = torch.tensor([6, 5, 13, 20, 12, 4, 3, 11, 19, 27, 34, 26, 18, 10, 2, 1, 9,17, 25,
                                           33, 41, 48, 40, 32, 24, 16, 8, 0, 7, 15, 23, 31, 39, 47, 46, 38, 30,
                                           22, 14, 21, 29, 37, 45, 44, 36, 28, 35, 43, 42])  # index of direction: 19diagonal_reverse
                index_19dr = torch.flip(index_19df, dims=[0])  # index of direction: 19diagonal_reverse
                index_ltcw = torch.tensor([0, 1, 2, 3, 4, 5, 6, 13, 20, 27, 34, 41, 48, 47, 46, 45, 44, 43, 42, \
                                         35, 28, 21, 14, 7, 8, 9, 10, 11, 12, 19, 26, 33, \
                                         40, 39, 38, 37, 36, 29, 22, 15, 16, 17, 18, 25, \
                                         32, 31, 30, 23, 24]) #index of direction: start from left-top(0) in clockwise
                index_ltacw = torch.tensor([0, 7, 14, 21, 28, 35, 42, 43, 44, 45, 46,47, 48, 41, 34, \
                                         27, 20, 13, 6, 5, 4, 3, 2, 1, 8, 15, 22, 29, 36, 37, 38, 39, \
                                         40, 33, 26, 19, 12, 11, 10, 9, 16, 23, 30, 31, 32, 25, 18, 17, 24]) #index of direction: start from left-top(0) in anticlockwise

                d_hf = x  # direction: horizontal_forward
                d_hr = torch.flip(x, [1])  # direction: horizontal_reverse
                d_vf = x[:, index_vf]
                d_vr = x[:, index_vr]
                d_37df = x[:, index_37df]
                d_37dr = x[:, index_37dr]
                d_19df = x[:, index_19df]
                d_19dr = x[:, index_19dr]
                d_ltcw = x[:, index_ltcw]
                d_ltacw = x[:, index_ltacw]

                x_inputs = [d_hf, d_hr, d_vf, d_vr, d_37df, d_37dr, d_19df, d_19dr, d_ltcw, d_ltacw]
                x_inputs = torch.cat(x_inputs, dim=0)

                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))  # 先合在一起做归一化、mamba mixer
                x_inputs = layer(x_inputs)  # （8B， 9,128）mambermixer 是按照每大组做的吗，还是所有batch的samples之间会有互动

                x_hf, x_hr, x_vf, x_vr, x_37df, x_37dr, x_19df, x_19dr, x_ltcw, x_ltacw = torch.split(x_inputs, B, dim=0)

                # 回去 reverse index
                index_vf = torch.argsort(index_vf)
                index_vr = torch.argsort(index_vr)
                index_37df = torch.argsort(index_37df)
                index_37dr = torch.argsort(index_37dr)
                index_19df = torch.argsort(index_19df)
                index_19dr = torch.argsort(index_19dr)
                index_ltcw = torch.argsort(index_ltcw)
                index_ltacw = torch.argsort(index_ltacw)

                x_hr = torch.flip(x_hr, [1])
                x_vf = x_vf[:, index_vf]
                x_vr = x_vr[:, index_vr]
                x_37df = x_37df[:, index_37df]
                x_37dr = x_37dr[:, index_37dr]
                x_19df = x_19df[:, index_19df]
                x_19dr = x_19dr[:, index_19dr]
                x_ltcw = x_ltcw[:, index_ltcw]
                x_ltacw = x_ltacw[:, index_ltacw]


                # mean_x_hf = torch.mean(x_hf, dim=1)  # （1，196,192）---》（1， 192）
                # mean_x_hr = torch.mean(x_hr, dim=1)
                # mean_x_vf = torch.mean(x_vf, dim=1)
                # mean_x_vr = torch.mean(x_vr, dim=1)
                # mean_x_37df = torch.mean(x_37df, dim=1)
                # mean_x_37dr = torch.mean(x_37dr, dim=1)
                # mean_x_19df = torch.mean(x_19df, dim=1)
                # mean_x_19dr = torch.mean(x_19dr, dim=1)
                #
                # gate = torch.cat([mean_x_hf, mean_x_hr, mean_x_vf, mean_x_vr, mean_x_37df, mean_x_37dr, mean_x_19df, mean_x_19dr], dim=-1)  # （1， 192*8）
                # gate = self.gate_layers[i](gate)  # （1， 192*8）--》（1， 8）计算量有点高且不合理
                # gate = gate.unsqueeze(-1)  # （1， 8， 1）
                gate = self.weights
                gate = F.softmax(gate, dim=1)
                # x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw + gate[:, 2:3] * x_rtcw + gate[:, 3:4] * x_rtacw  + \
                #     gate[:, 4:5] * x_ldcw + gate[:, 5:6] * x_ldacw + gate[:, 6:7] * x_rdcw + gate[:, 7:8] * x_rdacw
                # x = gate[:, 0:1] * x_ltcw + gate[:, 1:2] * x_ltacw + gate[:, 2:3] * x_rdcw + gate[:, 3:4] * x_rdacw
                x = gate[:, 0:1] * x_hf + gate[:, 1:2] * x_hr + gate[:, 2:3] * x_vf + gate[:, 3:4] * x_vr + \
                    gate[:, 4:5] * x_37df + gate[:, 5:6] * x_37dr + gate[:, 6:7] * x_19df + gate[:, 7:8] * x_19dr + \
                    gate[:, 8:9] * x_ltcw + gate[:, 9:10] * x_ltacw
            if '25twoclock' == self.path_type: # for input_size=(b, 9, dim_embedding)
                index_cw = torch.tensor([0, 1, 2, 3, 4, 9, 14, 19, 24, 23, 22, 21,
                                         20, 15, 10, 5, 6, 7, 8, 13, 18, 17, 16, 11, 12]) #index of direction: clockwise
                index_acw = torch.tensor([0, 5, 10, 15, 20, 21, 22, 23, 24, 19, 14, 9,
                                          4, 3, 2, 1, 6,  11, 16, 17, 18, 13, 8, 7, 12]) #index of direction: anticlockwise

                d_cw = x[:, index_cw]  # direction: horizontal_forward
                d_acw = x[:, index_acw]  # direction: horizontal_reverse
                x_inputs = [d_cw, d_acw]
                x_inputs = torch.cat(x_inputs, dim=0)

                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))  # 先合在一起做归一化、mamba mixer
                x_inputs = layer(x_inputs)  # （2， 196,192）mambermixer 是按照每大组做的吗，还是所有batch的samples之间会有互动

                x_cw, x_acw = torch.split(x_inputs, B, dim=0)
                # 回去 reverse index
                index_cw = torch.argsort(index_cw)
                index_acw = torch.argsort(index_acw)

                x_cw = x_cw[:, index_cw]
                x_acw = x_acw[:, index_acw]

                # mean_x_cw = torch.mean(x_cw, dim=1)  # （1，196,192）---》（1， 192）
                # mean_x_acw = torch.mean(x_acw, dim=1)
                #
                # gate = torch.cat([mean_x_cw, mean_x_acw], dim=-1)  # （1， 192*8）
                # gate = self.gate_layers[i](gate)  # （1， 192*8）--》（1， 8）计算量有点高且不合理
                # gate = gate.unsqueeze(-1)  # （1， 8， 1）
                gate = self.weights
                gate = F.softmax(gate, dim=1)
                x = gate[:, 0:1] * x_cw + gate[:, 1:2] * x_acw
            if '9twoclock' == self.path_type: # for input_size=(b, 9, dim_embedding)
                index_cw = torch.tensor([0, 3, 6, 7, 8, 5, 2, 1, 4]) #index of direction: clockwise
                index_acw = torch.tensor([0, 1, 2, 5, 8, 7, 6, 3, 4]) #index of direction: anticlockwise

                d_cw = x[:, index_cw]  # direction: horizontal_forward
                d_acw = x[:, index_acw]  # direction: horizontal_reverse
                x_inputs = [d_cw, d_acw]
                x_inputs = torch.cat(x_inputs, dim=0)

                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))  # 先合在一起做归一化、mamba mixer
                x_inputs = layer(x_inputs)  # （2， 196,192）mambermixer 是按照每大组做的吗，还是所有batch的samples之间会有互动

                x_cw, x_acw = torch.split(x_inputs, B, dim=0)
                # 回去 reverse index
                index_cw = torch.argsort(index_cw)
                index_acw = torch.argsort(index_acw)

                x_cw = x_cw[:, index_cw]
                x_acw = x_acw[:, index_acw]

                # mean_x_cw = torch.mean(x_cw, dim=1)  # （1，196,192）---》（1， 192）
                # mean_x_acw = torch.mean(x_acw, dim=1)
                #
                # gate = torch.cat([mean_x_cw, mean_x_acw], dim=-1)  # （1， 192*8）
                # gate = self.gate_layers[i](gate)  # （1， 192*8）--》（1， 8）计算量有点高且不合理
                # gate = gate.unsqueeze(-1)  # （1， 8， 1）
                gate = self.weights
                gate = F.softmax(gate, dim=1)
                x = gate[:, 0:1] * x_cw + gate[:, 1:2] * x_acw
            if 'forward_reverse_mean' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
                x = (forward_x + torch.flip(reverse_x, [1])) / 2
            if 'forward_reverse_gate' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs)
                forward_x, reverse_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                mean_forward_x = torch.mean(forward_x, dim=1)
                mean_reverse_x = torch.mean(reverse_x, dim=1)
                gate = torch.cat([mean_forward_x, mean_reverse_x], dim=-1)
                gate = self.gate_layers[i](gate)
                gate = gate.unsqueeze(-1)
                x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x
            if 'forward_reverse_shuffle_gate' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]#1, 196(=224/16 * 224/16), 192(=embedding_dim)
                rand_index = torch.randperm(x.size(1))
                x_inputs.append(x[:, rand_index])
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))#先合在一起做归一化、mamba mixer
                x_inputs = layer(x_inputs)#（3， 196,192）mambermixer 是按照每大组做的吗，还是所有batch的samples之间会有互动
                forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])#回去
                # reverse the random index
                rand_index = torch.argsort(rand_index)#回去
                shuffle_x = shuffle_x[:, rand_index]#回去
                # mean_forward_x = torch.mean(forward_x, dim=1)#（1，196,192）---》（1， 192）
                # mean_reverse_x = torch.mean(reverse_x, dim=1)
                # mean_shuffle_x = torch.mean(shuffle_x, dim=1)
                # gate = torch.cat([mean_forward_x, mean_reverse_x, mean_shuffle_x], dim=-1)#（1， 192*3）
                # gate = self.gate_layers[i](gate)#（1， 192*3）--》（1， 3）
                # gate = gate.unsqueeze(-1)#（1， 3， 1）
                gate = self.weights
                gate = F.softmax(gate, dim=1)
                x = gate[:, 0:1] * forward_x + gate[:, 1:2] * reverse_x + gate[:, 2:3] * shuffle_x
            if 'forward_reverse_shuffle_mean' == self.path_type:
                x_inputs = [x, torch.flip(x, [1])]
                rand_index = torch.randperm(x.size(1))
                x_inputs.append(x[:, rand_index])
                x_inputs = torch.cat(x_inputs, dim=0)
                x_inputs = self.pre_norm(x_inputs.to(dtype=self.pre_norm.weight.dtype))
                x_inputs = layer(x_inputs) #MambaMixer
                forward_x, reverse_x, shuffle_x = torch.split(x_inputs, B, dim=0)
                reverse_x = torch.flip(reverse_x, [1])
                # reverse the random index
                rand_index = torch.argsort(rand_index)
                shuffle_x = shuffle_x[:, rand_index]
                x = (forward_x + reverse_x + shuffle_x) / 3

            x = residual + x
            if i == len(self.layers) - 1 and self.final_norm: #####################33
                x = self.ln1(x) #######################

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return outs

    def _format_output(self, x, hw):
        if self.out_type == 'raw':
            return x
        if self.out_type == 'cls_token':
            if self.cls_position == 'head':
                return x[:, 0]
            elif self.cls_position == 'tail':
                return x[:, -1]
            elif self.cls_position == 'head_tail':
                x = torch.mean(x[:, [0, -1]], dim=1)
                return x
            elif self.cls_position == 'middle':
                return x[:, x.size(1) // 2]
        patch_token = x
        if self.cls_token is not None:
            if self.cls_position == 'head':
                patch_token = x[:, 1:]
            elif self.cls_position == 'tail':
                patch_token = x[:, :-1]
            elif self.cls_position == 'head_tail':
                patch_token = x[:, 1:-1]
            elif self.cls_position == 'middle':
                patch_token = torch.cat((x[:, :x.size(1) // 2], x[:, x.size(1) // 2 + 1:]), dim=1)
        if self.out_type == 'featmap':
                B = x.size(0)
                return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
        if self.out_type == 'avg_featmap':
            # weights = self.weights
            # weights = self.gate_layers_for_tokens(patch_token.reshape(patch_token.shape[0], patch_token.shape[1] * self.embed_dims)).reshape((patch_token.shape[0], 49, 1))

            # selected_indices = torch.randperm(49)[:9]  # 随机选择9个索引
            # selected_indices = torch.tensor([0, 6, 11, 17, 24, 28, 38, 47])
            # patch_token = patch_token[:, selected_indices, :]

            # patch_token_weights = patch_token * weights
            # patch_token_weights = patch_token_weights.sum(dim=1)#形状是(64, 32)
            # return self.ln2(patch_token_weights)

            # return self.ln2(patch_token[:, 25])

            return self.ln2(patch_token.mean(dim=1))


class ms_conv_bn_relu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(ms_conv_bn_relu, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.activation = nn.ReLU()
    def forward(self, x):
        out = self.bn(x)
        out = self.conv(out)
        out = self.activation(out)

        return out

class GlobalLocalBlock(nn.Module):
    def __init__(self, img_size, in_channels, out_channels):
        super(GlobalLocalBlock, self).__init__()
        path = ['81_2+8', '49_2+8']
        if img_size == 9:
            path_type = path[0]
            self.global_view = hsiMamba(arch='globalview1', img_size=img_size,
                                        patch_size=1, in_channels=in_channels, out_type='featmap',
                                        path_type=path_type, patch_cfg=dict(stride=1))
            self.global_feature = TokenLearner((img_size-2)*(img_size-2))

        if img_size == 7:
            path_type = path[1]
            self.global_view = hsiMamba(arch='globalview2', img_size=img_size,
                                        patch_size=1, in_channels=in_channels, out_type='featmap',
                                        path_type=path_type, patch_cfg=dict(stride=1))
            self.global_feature = TokenLearner((img_size-2)*(img_size-2))

        self.change_dim = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.ln3 = build_norm_layer(dict(type='LN', eps=1e-6), out_channels)
        self.local_feature = ms_conv_bn_relu(in_channels, out_channels, kernel_size=3)
        self.channel_feature = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.channel_token = TokenLearner((img_size - 2) * (img_size - 2))
        self.ln4 = build_norm_layer(dict(type='LN', eps=1e-6), out_channels)

        self.FusionLayer = GLfusionBlock(out_channels, out_channels, out_channels)
        self.fusion = fusionBlock(out_channels, out_channels, out_channels)

    def forward(self, hsi):
        global_view = self.global_view(hsi)[0]
        B, C, H, W = global_view.shape


        global_feature = self.ln3(self.global_feature(self.change_dim(global_view))).reshape(B, H-2, H-2, -1).permute(0, 3, 1, 2)
        local_feature = self.local_feature(hsi)
        channel_feature = self.ln4(self.channel_token(self.channel_feature(hsi))).reshape(B, H-2, H-2, -1).permute(0, 3, 1, 2)

        feature_map = self.FusionLayer(channel_feature, local_feature)

        feature_map = self.fusion(global_feature, feature_map)

        return feature_map

class GLfusionBlock(nn.Module):
    def __init__(self, ch1, ch2, out_ch):
        super(GLfusionBlock, self).__init__()
        self.cross_attention = NONLocalBlock2D(ch1)
        self.FusionLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=ch1 + ch2,
                out_channels=out_ch,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x1, x2):#x1:global x2:local
        # localf = x2 + x1
        # globalf_enhancement = self.cross_attention(x2, x1, x1)
        # globalf = globalf_enhancement + x2
        #注释部分与下面是等价的操作，做实验的时候用的是注释部分，但写论文的时候发现这样解释不顺，因此换了局部增强与全局增强的代号
        globalf = x2 + x1
        localf_enhancement = self.cross_attention(x2, x1, x1)
        localf = localf_enhancement + x2
        x = torch.concat((localf, globalf), 1)
        x = self.FusionLayer(x)
        return x

class fusionBlock(nn.Module):
    def __init__(self, ch1, ch2, out_ch):
        super(fusionBlock, self).__init__()
        self.ch = ChannelExchange()
        self.FusionLayer = nn.Sequential(
            nn.Conv2d(
                in_channels=ch1 + ch2,
                out_channels=out_ch,
                kernel_size=1,
            ),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x1, x2):
        if x1.shape[1] == x2.shape[1]:
            x1, x2 = self.ch(x1, x2)
        x = torch.concat((x1, x2), dim=1)
        # x = self.ch(x1, x2)
        x = self.FusionLayer(x)
        return x

class Multimodality_Mamba(nn.Module):
    def __init__(self, img_size,  patch_size, stride,  in_channels1, in_channels2, dim_embedding, num_class, path_type):
        super(Multimodality_Mamba, self).__init__()
        self.embedding_dim = dim_embedding
        plane_hsi = [in_channels1, 256, in_channels1]  #[in_channels1, 256, in_channels1]for H2013
        plane_lidar = [in_channels2, 16, 32]
        plane_fusion = [128]

        self.hsi1 = GlobalLocalBlock(img_size=9, in_channels=plane_hsi[0], out_channels=plane_hsi[1])
        self.hsi2 = GlobalLocalBlock(img_size=7, in_channels=plane_hsi[1], out_channels=plane_hsi[2])

        self.lidar1 = ms_conv_bn_relu(plane_lidar[0], plane_lidar[1], kernel_size=3, bias=True)
        self.lidar2 = ms_conv_bn_relu(plane_lidar[1], plane_lidar[2], kernel_size=3, bias=True)

        self.fusion1 = fusionBlock(plane_hsi[1], plane_lidar[1], plane_fusion[0])
        self.fusion2 = fusionBlock(plane_hsi[2], plane_lidar[2], plane_fusion[0])

        self.avg = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Linear(128, num_class)
        # self.classifier = nn.Linear(self.ch_after_fe, num_class)


    def forward(self, hsi, lidar):
        hsi1 = self.hsi1(hsi)
        hsi2 = self.hsi2(hsi1)

        lidar1 = self.lidar1(lidar)
        lidar2 = self.lidar2(lidar1)

        fusion1 = self.fusion1(hsi1, lidar1)
        fusion2 = self.fusion2(hsi2, lidar2)

        feature1 = self.avg(fusion1)
        feature2 = self.avg(fusion2)
        feature = feature1 + feature2
        feature = torch.flatten(feature, 1)
        output = self.classifier(feature)


        return output



if __name__ == '__main__':
    import torch

    img1 = torch.randn(64, 144, 9, 9).cuda()
    img2 = torch.randn(64, 1, 9, 9).cuda()
    epoch = 1
    dim_embedding = 64//2#64//2
    path_type = 'multi_clock_gate'  #'clock_gate'   'eight_directions_gate'  'multi_clock_gate'  'forward_reverse_shuffle_gate'
    for e in range(epoch):
        net = Multimodality_Mamba(img_size=7, patch_size=1, stride=1, in_channels1= img1.shape[1], in_channels2=1, dim_embedding=dim_embedding, num_class=10, path_type=path_type).cuda()
        out = net(img1, img2)
        print("epoch:{}\n".format(e),out)

    macs, params = profile(net, inputs=(img1, img2))
    macs, params = clever_format([macs, params], "%.3f")

    print('macs', macs, '\nparams', params)
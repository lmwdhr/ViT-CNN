# -*- coding: utf-8 -*-
# Torch
import copy

import spectral
import torch.nn as nn
import torch
import torch.optim as optim

# utils
import os
import datetime
import numpy as np
import joblib

from tqdm import tqdm

from model.CascadeMamba import CascadeRSMamba_complete
from model.FICNN_VIT import FICNN_VIT
from model.compare_method.MFT import MFT
from model.compare_method.GLT_Net.GLT_Net import GLT
from model.HybridSN import HybridSN
from model.compare_method.MHST.MHST import MHST
from model.Multimodality_Mamba.Mutimodality_Mamba7 import Multimodality_Mamba
from model.RSMamba import RSMamba_complete
from model.SupConResNet import SupConResNet
from model.compare_method.HCTnet import HCTnet
from model.Selective.fasternet import FasterNet
from model.compare_method.spectralformer import SpectralFormer
from utils import grouper, sliding_window, count_sliding_window, camel_to_snake, applyPCA, adjust_learning_rate
from model.compare_method.FusAtNet import FusAtNet
from model.compare_method.EndNet import EndNet
from model.compare_method.S2EFT import ViT
from model.compare_method.DML_Hong import Early_fusion_CNN, Middle_fusion_CNN, Late_fusion_CNN, Cross_fusion_CNN
from model.S2ENet import S2ENet
from model.FI_CNN import FI_CNN
from model.ResNet18 import ResNet18
from model.S2ENet_ResNet18 import S2ENet_ResNet18
from model.multiScaleCNN import multiScaleCNN
from model.FI_CNN3D import FI_CNN3D
from model.VIT import VIT
from model.proposed import proposed
from model.nncnet import moco_based_NNCNet
from losses import Cross_fusion_CNN_Loss, EndNet_Loss


def get_model(name, **kwargs):
    """
    Instantiate and obtain a model with adequate hyperparameters

    Args:
        name: string of the model name
        kwargs: hyperparameters
    Returns:
        model: PyTorch network
        optimizer: PyTorch optimizer
        criterion: PyTorch loss Function
        kwargs: hyperparameters with sane defaults
    """
    device = kwargs.setdefault("device", torch.device("cpu"))
    n_classes = kwargs["n_classes"]
    (n_bands, n_bands2) = kwargs["n_bands"]
    weights = torch.ones(n_classes)
    weights[torch.LongTensor(kwargs["ignored_labels"])] = 0.0
    weights = weights.to(device)
    weights = kwargs.setdefault("weights", weights)
    dataset = kwargs["dataset"]

    if name == "Early_fusion_CNN":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = Early_fusion_CNN(n_bands, n_bands2, n_classes)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 150)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", False)
    elif name == "Middle_fusion_CNN":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = Middle_fusion_CNN(n_bands, n_bands2, n_classes)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 150)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", False)
    elif name == "Late_fusion_CNN":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = Late_fusion_CNN(n_bands, n_bands2, n_classes)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 150)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", False)
    elif name == "Cross_fusion_CNN":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = Cross_fusion_CNN(n_bands, n_bands2, n_classes)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = Cross_fusion_CNN_Loss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 150)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", False)
    elif name == "FusAtNet":
        kwargs.setdefault("patch_size", 11)
        center_pixel = True
        model = FusAtNet(n_bands, n_bands2, n_classes)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 150)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", False)
    elif name == "EndNet":
        kwargs.setdefault("patch_size", 1)
        center_pixel = True
        model = EndNet(n_bands, n_bands2, n_classes)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = EndNet_Loss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 150)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", False)
    elif name == "S2ENet":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = S2ENet(n_bands, n_bands2, n_classes, kwargs["patch_size"])
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", False)
    elif name == "FI_CNN":
        kwargs.setdefault("patch_size", 23)
        center_pixel = True
        kwargs.setdefault("applyPCA", True) ######
        if kwargs["applyPCA"] == True:  ######
            n_bands = 30
        model = FI_CNN(n_bands, n_bands2, n_classes, kwargs["patch_size"])
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
    elif name == "ResNet18":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        kwargs.setdefault("applyPCA", True)  ######
        if kwargs["applyPCA"] == True:  ######
            n_bands = 30
        model = ResNet18(n_bands, n_bands2, n_classes, kwargs["patch_size"])
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
    elif name == "S2ENet_ResNet18":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = S2ENet_ResNet18(n_bands, n_bands2, n_classes, kwargs["patch_size"])
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", True)
    elif name == "multiScaleCNN":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = multiScaleCNN(n_bands, n_bands2, n_classes, kwargs["patch_size"])
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", False)
    elif name == "FI_CNN3D":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        model = FI_CNN3D(n_bands, n_bands2, n_classes, kwargs["patch_size"])
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
        kwargs.setdefault("applyPCA", False)
    elif name == "VIT":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        embed_dim = 32
        kwargs.setdefault("applyPCA", True)#####
        if kwargs["applyPCA"] == True:######
            n_bands = 30
        model = VIT(n_bands, n_bands2, embed_dim, patch_size=1, num_patches=kwargs["patch_size"] *  kwargs["patch_size"], nheads=4, num_layers=2, num_classes=n_classes, dropout=0.01)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
    elif name == "FICNN_VIT":
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        embed_dim = 32
        kwargs.setdefault("applyPCA", True)#####
        if kwargs["applyPCA"] == True:######
            n_bands = 30
        model = FICNN_VIT(n_bands, n_bands2, embed_dim, patch_size=kwargs["patch_size"], patch_size_vit=1, num_patches=kwargs["patch_size"] *  kwargs["patch_size"], nheads=4, num_layers=2, num_classes=n_classes, dropout=0.01)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
    elif name == "SupConResNet":
        kwargs.setdefault("patch_size", 7)
        kwargs.setdefault("restore", "/home/dell/exp/lmw/code_in_lectures/Multimodal-Remote-Sensing-Toolkit-main/model/pretrain_checkpoint/SupConResNet/ckpt_epoch_1000.pth")
        center_pixel = True
        embed_dim = 32

        checkpoint = torch.load(kwargs["restore"], map_location='cpu')
        state_dict = checkpoint['model']
        new_state_dict = {}
        for k, v in state_dict.items():
            k = k.replace("module.", "")
            new_state_dict[k] = v
        state_dict = new_state_dict

        kwargs.setdefault("applyPCA", True)#####
        if kwargs["applyPCA"] == True:######
            n_bands = 30
        # classifier = LinearClassifier(num_classes=kwargs['n_classes'])
        model = SupConResNet(num_class=kwargs['n_classes'])
        model_dict = model.state_dict()
        state_dict = {key: value for key, value in state_dict.items() if
                           (key in model_dict)}
        # model.encoder.load_state_dict(state_dict["encoder%"])#不能赋值
        # del model.head
        # model.add_module('classifier', classifier)#不能赋值
        # print(model)
        for name, param in model.named_parameters():#不计算梯度
            if "encoder" in name:
                param.requires_grad = False

        lr = kwargs.setdefault("lr", 5)#0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)#128
        kwargs.setdefault("batch_size", 512)#64
    elif name == "HybridSN":
        kwargs.setdefault("patch_size", 25)
        center_pixel = True
        embed_dim = 32
        kwargs.setdefault("applyPCA", True)#####
        if kwargs["applyPCA"] == True:######
            n_bands = 3
        model = HybridSN(n_bands+n_bands2, patch_size=kwargs["patch_size"], class_nums=n_classes)
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
    elif name == "RSMamba_complete":
        kwargs.setdefault("patch_size", 7)
        patch_size = kwargs["patch_size"]
        center_pixel = True
        embed_dim = 192 #经过mamba处理后的维度
        kwargs.setdefault("applyPCA", False)#####
        # if kwargs["applyPCA"] == True:######
        #     n_bands = 30
        path_type = 'forward_reverse_shuffle_gate'#forward_reverse_shuffle_gate
        model = RSMamba_complete(img_size=patch_size, patch_size=1, in_channels1=n_bands, in_channels2=n_bands2, dim_embedding=embed_dim, num_class=n_classes, path_type=path_type, patch_cfg={})
        lr = kwargs.setdefault("lr", 0.001)#0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
    elif name == "CascadeRSMamba_complete":
        kwargs.setdefault("patch_size", 7)
        patch_size = kwargs["patch_size"]
        center_pixel = True
        embed_dim = 192 #经过mamba处理后的维度
        kwargs.setdefault("applyPCA", False)#####
        # if kwargs["applyPCA"] == True:######
        #     n_bands = 30
        path_type = 'forward'
        model = CascadeRSMamba_complete(img_size=patch_size, patch_size=1, in_channels1=n_bands, in_channels2=n_bands2, dim_embedding=embed_dim, num_class=n_classes, path_type=path_type, patch_cfg={})
        lr = kwargs.setdefault("lr", 0.001)#0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 128)
        kwargs.setdefault("batch_size", 64)
    elif name == "Multimodality_Mamba":
        kwargs.setdefault("patch_size", 9)
        patch_size = kwargs["patch_size"]
        center_pixel = True
        embed_dim = 64//2 #经过mamba处理后的维度 192
        kwargs.setdefault("applyPCA", False)#####
        if kwargs["applyPCA"] == True:######
            n_bands = 30
        path_type = 'multi_clock_gate'   #'clock_gate'   'eight_directions_gate'  'forward_reverse_shuffle_gate'   'multi_clock_gate'
        model = Multimodality_Mamba(img_size=patch_size, patch_size=1, stride=1, in_channels1=n_bands, in_channels2=n_bands2, dim_embedding=embed_dim, num_class=n_classes, path_type=path_type)
        # lr = kwargs.setdefault("lr", 0.001)#0.001
        # optimizer = optim.Adam(model.parameters(), lr=lr)######
        lr = kwargs.setdefault("lr", 8e-4) #8e-4for Houston 5e-4for MUUFL
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 200)##128
        kwargs.setdefault("batch_size", 64)
    elif name == "MHST":
        kwargs.setdefault("patch_size", 8)
        patch_size = kwargs["patch_size"]
        center_pixel = True
        kwargs.setdefault("applyPCA", False)#####
        if kwargs["applyPCA"] == True:######
            n_bands = 30
        model = MHST(l1=n_bands, l2=n_bands2, patch_size=patch_size,
                     num_patches=64, num_classes=n_classes,
                     encoder_embed_dim=64,
                     en_depth=5, en_heads=4,
                     mlp_dim=8, dropout=0.1, emb_dropout=0.1,
                     coefficient_hsi=0.6, coefficient_vit=0.7,
                     hsp_vit_depth=8, hsp_vit_num_heads=16,
                     head_tau=5, use_head_select=True,
                     vit_qkv_bias=False,
                     mlp_ratio=4, attnproj_mlp_drop=0.1, attn_drop=0.1)
        lr = kwargs.setdefault("lr", 8e-4)#learning rate for Trento：5e-4 & Houston：8e-4 & Muufl：4e-4 respectively
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 1000)##128
        kwargs.setdefault("batch_size", 64)
    elif name == "GLT_Net":
        kwargs.setdefault("patch_size", 8)
        patch_size = kwargs["patch_size"]
        center_pixel = True
        kwargs.setdefault("applyPCA", False)  #####
        if kwargs["applyPCA"] == True:  ######
            n_bands = 30
        model = GLT(l1=n_bands, l2=n_bands2, patch_size=patch_size, num_patches=64, num_classes=n_classes,
                    encoder_embed_dim=64, decoder_embed_dim=32,
                    en_depth=5, en_heads=4, de_depth=5, de_heads=4, mlp_dim=8, dropout=0.1, emb_dropout=0.1)
        lr = kwargs.setdefault("lr", 5e-4)
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 200)  ##128
        kwargs.setdefault("batch_size", 64)
    elif name == "HCTnet":#用主成分分析到30
        kwargs.setdefault("patch_size", 11)
        patch_size = kwargs["patch_size"]
        center_pixel = True
        kwargs.setdefault("applyPCA", True)  #####
        if kwargs["applyPCA"] == True:  ######
            n_bands = 30
        model = HCTnet(num_classes=n_classes, num_tokens=6, heads=8)#num_tokens: H/T/M/A:6/6/4/4; Attention Heads: H/T/M/A:8/4/8/8
        lr = kwargs.setdefault("lr", 0.0001)#muffl：1e-4，houston2013、trento：1e-3, ausburg:5e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 100)  ##
        kwargs.setdefault("batch_size", 64)
    elif name == "MFT":#用主成分分析到30
        kwargs.setdefault("patch_size", 11)###
        patch_size = kwargs["patch_size"]
        center_pixel = True
        kwargs.setdefault("applyPCA", False)  #####
        if kwargs["applyPCA"] == True:  ######
            n_bands = 30
        model = MFT(patch_size=patch_size, FM=16, NC=n_bands, NCLidar=n_bands2, Classes=n_classes, HSIOnly=False)
        lr = kwargs.setdefault("lr", 0.0005)#muffl：1e-4，houston2013、trento：1e-3, ausburg:5e-4
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 500)  ##
        kwargs.setdefault("batch_size", 64)
    elif name == "SpectralFormer":#用主成分分析到30
        kwargs.setdefault("patch_size", 1)###
        patch_size = kwargs["patch_size"]
        center_pixel = True
        kwargs.setdefault("applyPCA", False)  #####
        if kwargs["applyPCA"] == True:  ######
            n_bands = 30
        model = SpectralFormer(image_size=patch_size,
                               near_band=1,
                               num_patches=n_bands+n_bands2,
                               num_classes=n_classes,
                               dim=64,
                               depth=5,
                               heads=4,
                               mlp_dim=8,
                               dropout=0.1,
                               emb_dropout=0.1,
                               mode='ViT')
        lr = kwargs.setdefault("lr", 0.0005)#muffl：1e-4，houston2013、trento：1e-3, ausburg:5e-4
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 300)  ##
        kwargs.setdefault("batch_size", 64)
    elif name == "S2EFT":#用主成分分析到30
        kwargs.setdefault("patch_size", 7)
        center_pixel = True
        kwargs.setdefault("applyPCA", False)  #####
        if kwargs["applyPCA"] == True:  ######
            n_bands = 30
        model = ViT(
            image_size= kwargs["patch_size"],
            near_band=3,
            num_patches=n_bands,
            num_classes=n_classes,
            dim=64,
            depth=5,
            heads=4,
            mlp_dim=8,
            dropout=0.1,
            emb_dropout=0.1,
            mode='CAF'
        )
        lr = kwargs.setdefault("lr", 0.0005)#muffl：1e-4，houston2013、trento：1e-3, ausburg:5e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 600)  ##
        kwargs.setdefault("batch_size", 64)
    elif name == "PCCNet":
        kwargs.setdefault("patch_size", 9)
        patch_size = kwargs["patch_size"]
        center_pixel = True
        kwargs.setdefault("applyPCA", False)  #####
        if kwargs["applyPCA"] == True:  ######
            n_bands = 30
        model = FasterNet(
            in_chans=n_bands,  #####3
            num_classes=n_classes,  ####1000
            embed_dim=96,
            depths=(1, 2, 8, 2),  #####！！
            mlp_ratio=2.,
            n_div=4,
            patch_size=1,  ####4
            patch_stride=1,  ####4
            patch_size2=1,  # for subsequent layers #########2
            patch_stride2=1,  #########2
            patch_size3=[3, 0, 0],
            patch_stride3=[3, 0, 0],
            patch_norm=True,
            feature_dim=128,  #####1280
            drop_path_rate=0.1,
            layer_scale_init_value=0,
            norm_layer='BN',
            act_layer='RELU',
            fork_feat=True,
            init_cfg=None,
            pretrained=None,
            pconv_fw_type='split_cat')
        lr = kwargs.setdefault("lr", 0.0001)  # muffl：1e-4，houston2013、trento：1e-3, ausburg:5e-4
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 200)  ##
        kwargs.setdefault("batch_size", 64)
    elif name == "groupViT":
        kwargs.setdefault("patch_size", 9)
        patch_size = kwargs["patch_size"]
        center_pixel = True
        kwargs.setdefault("applyPCA", False)#####
        if kwargs["applyPCA"] == True:######
            n_bands = 30
        model = proposed(dataset, patch_size) #'Salinas', 'PaviaU', 'Houston2013'
        # lr = kwargs.setdefault("lr", 0.001)#0.001
        # optimizer = optim.Adam(model.parameters(), lr=lr)######
        lr = kwargs.setdefault("lr", 0.001)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001)
        criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])
        kwargs.setdefault("epoch", 300)##128
        kwargs.setdefault("batch_size", 64)
    elif name == "moco_based_NNCNet":
        kwargs.setdefault("patch_size", 9)
        patch_size = kwargs["patch_size"]
        center_pixel = True
        kwargs.setdefault("applyPCA", False)#####
        if kwargs["applyPCA"] == True:######
            n_bands = 30
        model = moco_based_NNCNet(base_encoder=proposed, dataset='Houston2013', patch_size=patch_size) #'Salinas', 'PaviaU', 'Houston2013'
        kwargs.setdefault("lr", 0.0005)##对学习率，nncnet还弄了个学习率下降
        lr = kwargs["lr"]
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()##criterion = nn.CrossEntropyLoss(weight=kwargs["weights"])原本是带weights的，但是1.因为nncnet的不带 2.而且weights的权重设置的大小事类别的数量16，跟对比损失需要的2048不一样，因此直接不要了
        kwargs.setdefault("epoch", 200)##与nncnet一样
        kwargs.setdefault("batch_size", 64)##nncnet的设置，pertrain是64，train是128(我这里先都一样，都是64)
    else:
        raise KeyError("{} model is unknown.".format(name))

    model = model.to(device)
    # epoch = kwargs.setdefault("epoch", 128)###100
    kwargs.setdefault(
        "scheduler",
        # optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.1, patience=epoch // 4, verbose=True
        # ),
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.9)  # SpectralFormer用的这个
        # torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)#MFT用的这个
        # torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)#111_57_1用的这个
        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch),#原先用的这个
        # torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[90, 150, 180], gamma=0.1),
    )
    # kwargs.setdefault('scheduler', None)
    # kwargs.setdefault("batch_size", 64) ####????
    kwargs.setdefault("supervision", "full")
    kwargs.setdefault("flip_augmentation", False)
    kwargs.setdefault("radiation_augmentation", False)
    kwargs.setdefault("mixture_augmentation", False)
    kwargs["center_pixel"] = center_pixel
    return model, optimizer, criterion, kwargs


#原train
# def train(
#     run,
#     net,
#     optimizer,
#     criterion,
#     data_loader,
#     epoch,
#     scheduler=None,
#     display_iter=100,
#     device=torch.device("cpu"),
#     display=None,
#     val_loader=None,
#     supervision="full",
# ):
#     """
#     Training loop to optimize a network for several epochs and a specified loss
#
#     Args:
#         net: a PyTorch model
#         optimizer: a PyTorch optimizer
#         data_loader: a PyTorch dataset loader
#         epoch: int specifying the number of training epochs
#         criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
#         device (optional): torch device to use (defaults to CPU)
#         display_iter (optional): number of iterations before refreshing the
#         display (False/None to switch off).
#         scheduler (optional): PyTorch scheduler
#         val_loader (optional): validation dataset
#         supervision (optional): 'full' or 'semi'
#     """
#
# # 这个criterion好像就是损失函数， 在模型选择那里定义过
#     if criterion is None:
#         raise Exception("Missing criterion. You must specify a loss function.")
#
#     net.to(device)
#
#     # save_epoch = epoch // 20 if epoch > 20 else 1
#     if epoch==128:
#         save_epoch = 16
#     else:
#         save_epoch = epoch // 20 if epoch > 20 else 1
#
#     losses = np.zeros(1000000)
#     mean_losses = np.zeros(100000000)
#     iter_ = 1
#     loss_win, val_win = None, None
#     val_accuracies = []
#
#     for e in tqdm(range(1, epoch + 1), desc="Training the network (run:{})".format(run)):
#         # Set the network to training mode
#         net.train()
#         avg_loss = 0.0
#
#         # Run the training loop for one epoch
#         for batch_idx, (data, data2, target) in tqdm(
#             enumerate(data_loader), total=len(data_loader)
#         ):
#             # Load the data into the GPU if required
#             data, data2, target = data.to(device), data2.to(device), target.to(device)
#
#             optimizer.zero_grad()
#             if supervision == "full":
#                 output = net(data, data2)
#                 loss = criterion(output, target)
#             elif supervision == "semi":
#                 outs = net(data, data2)
#                 output, rec = outs
#                 loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
#                     1
#                 ](rec, data)
#             else:
#                 raise ValueError(
#                     'supervision mode "{}" is unknown.'.format(supervision)
#                 )
#             loss.backward()
#             optimizer.step()
#
#             avg_loss += loss.item()
#             losses[iter_] = loss.item()
#             mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])#计算最近100次迭代中损失的平均值
#
#             if display_iter and iter_ % display_iter == 0:
#                 string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
#                 string = string.format(
#                     e,
#                     epoch,
#                     batch_idx * len(data),
#                     len(data) * len(data_loader),
#                     100.0 * batch_idx / len(data_loader),
#                     mean_losses[iter_],
#                 )
#                 update = None if loss_win is None else "append"
#                 loss_win = display.line(
#                     X=np.arange(iter_ - display_iter, iter_),
#                     Y=mean_losses[iter_ - display_iter : iter_],
#                     win=loss_win,
#                     update=update,
#                     opts={
#                         "title": "Training loss run:{}".format(run),
#                         "xlabel": "Iterations",
#                         "ylabel": "Loss",
#                     },
#                 )
#                 tqdm.write(string)
#
#                 if len(val_accuracies) > 0:
#                     val_win = display.line(
#                         Y=np.array(val_accuracies),
#                         X=np.arange(len(val_accuracies)),
#                         win=val_win,
#                         opts={
#                             "title": "Validation accuracy run:{}".format(run),
#                             "xlabel": "Epochs",
#                             "ylabel": "Accuracy",
#                         },
#                     )
#             iter_ += 1
#             del (data, target, loss, output)
#
#         # Update the scheduler
#         avg_loss /= len(data_loader)
#         if val_loader is not None:
#             val_acc = val(net, val_loader, device=device, supervision=supervision)
#             val_accuracies.append(val_acc)
#             metric = -val_acc
#         else:
#             metric = avg_loss
#
#         if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
#             scheduler.step(metric)
#         elif scheduler is not None:
#             scheduler.step()
#
#         # Save the weights
#         if e % save_epoch == 0:
#             save_model(
#                 net,
#                 camel_to_snake(str(net.__class__.__name__)),
#                 data_loader.dataset.name,
#                 run=run,
#                 epoch=e,
#                 metric=abs(metric),
#             )


def show_featuremap(note, img, bands, display):
    # [N, C, H, W] -> [C, H, W]
    img = img.cpu().detach().numpy()  # 把tensor变成numpy
    img = img[0]  # 选取batch中的第一张可视化
    # [C, H, W] -> [H, W, C]
    img = np.transpose(img, [1, 2, 0])
    # 对图像的通道进行处理
    rgb = spectral.get_rgb(img, bands)
    # 将 RGB 图像转换为灰度图
    rgb = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    #归一化
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "{}_featuremap (bands {}, {}, {})".format(note, *bands)
    # send to visdom server
    display.images(rgb,
                   opts={'caption': caption, 'height': 300, 'weight':300, 'cmap':'gray'})


def pretrain(
    savename,
    run,
    bands,
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    supervision="full",
    hyperparams=None
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """
    best_loss = 100  #####for best checkpoint
# 这个criterion好像就是损失函数， 在模型选择那里定义过
    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    # save_epoch = epoch // 20 if epoch > 20 else 1
    if epoch==128:
        save_epoch = 16
    else:
        save_epoch = epoch // 20 if epoch > 20 else 1

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network (run:{})".format(run)):
        adjust_learning_rate(optimizer, e-1, hyperparams)###e-1是为了与nncnet的配置保持一致
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        # Run the training loop for one epoch
        for batch_idx, (data1_1, data1_2, data2_1, data2_2, label) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # Load the data into the GPU if required
            data1_1, data1_2, data2_1, data2_2, label = data1_1.to(device), data1_2.to(device), data2_1.to(device), data2_2.to(device), label.to(device)

            # optimizer.zero_grad()##原先是放在这的
            if supervision == "full":
                # output, origin, layers_outpout, masked_output = net(data, data2)  #打印各层特征图
                output, target, k = net(data1_1, data1_2, data2_1, data2_2)
                loss = criterion(output, target)
            elif supervision == "semi":###不管，没用着
                outs = net(data1_1, data1_2, data2_1, data2_2)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data1_1, data1_2, data2_1, data2_2)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )

            optimizer.zero_grad()##和nncnet保持一致
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])#计算最近100次迭代中损失的平均值

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data1_1),
                    len(data1_1) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
                    win=loss_win,
                    update=update,
                    opts={
                        "title": "Training loss run:{}".format(run),
                        "xlabel": "Iterations",
                        "ylabel": "Loss",
                    },
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(
                        Y=np.array(val_accuracies),
                        X=np.arange(len(val_accuracies)),
                        win=val_win,
                        opts={
                            "title": "Validation accuracy run:{}".format(run),
                            "xlabel": "Epochs",
                            "ylabel": "Accuracy",
                        },
                    )
            iter_ += 1
            del (data1_1, data1_2, data2_1, data2_2, target, loss, output)

        ###打印最后一代的各层特征图，自增
        # if e == epoch:  ###每一次训练输出一次, 即最后一代输出
        #     # origin (B,C,H,W)     layers_outpout, masked_output list:num_layers-1
        #     show_featuremap("origin", origin, bands, display)
        #     # 获取输出列表这是一个列表，里面每个代表了每层的输出
        #     for i, (layers_fm, masked_fm) in enumerate(zip(layers_outpout, masked_output)):
        #         if i == 0 or i == 2:
        #             show_featuremap("layers{}".format(i), layers_fm, bands, display)
        #             show_featuremap("masked{}".format(i), masked_fm, bands, display)

        # Update the scheduler
        avg_loss /= len(data_loader)
        metric = avg_loss

        #save optimal weights
        if abs(metric) <= best_loss: ###改成小于等于bestloss
            best_loss = abs(metric)
            best_model_wts = copy.deepcopy(net.state_dict())
            save_model(
                    savename,
                    net,
                    camel_to_snake(str(net.__class__.__name__)),
                    data_loader.dataset.name,
                    train_state = 'pre_train',
                    type = 'best_epoch',##'final_epoch', 'best_epoch'
                    run=run,
                    epoch=e,
                    metric=abs(metric),
                )

        # save final weights
        if e == 128 or e == 200 or e == 300:
            save_model(
                savename,
                net,
                camel_to_snake(str(net.__class__.__name__)),
                data_loader.dataset.name,
                train_state='pre_train',
                type='final_epoch',  ##'final_epoch', 'best_epoch'
                run=run,
                epoch=e,
                metric=abs(metric),
            )


#train (return best model param dict版)
def train(
    savename,
    run,
    bands,
    net,
    optimizer,
    criterion,
    data_loader,
    epoch,
    scheduler=None,
    display_iter=100,
    device=torch.device("cpu"),
    display=None,
    val_loader=None,
    supervision="full"
):
    """
    Training loop to optimize a network for several epochs and a specified loss

    Args:
        net: a PyTorch model
        optimizer: a PyTorch optimizer
        data_loader: a PyTorch dataset loader
        epoch: int specifying the number of training epochs
        criterion: a PyTorch-compatible loss function, e.g. nn.CrossEntropyLoss
        device (optional): torch device to use (defaults to CPU)
        display_iter (optional): number of iterations before refreshing the
        display (False/None to switch off).
        scheduler (optional): PyTorch scheduler
        val_loader (optional): validation dataset
        supervision (optional): 'full' or 'semi'
    """
    best_val_acc = 0.0  #####for best checkpoint
# 这个criterion好像就是损失函数， 在模型选择那里定义过
    if criterion is None:
        raise Exception("Missing criterion. You must specify a loss function.")

    net.to(device)

    # save_epoch = epoch // 20 if epoch > 20 else 1
    if epoch==128:
        save_epoch = 16
    else:
        save_epoch = epoch // 20 if epoch > 20 else 1
    final_epoch = epoch###自增

    losses = np.zeros(1000000)
    mean_losses = np.zeros(100000000)
    iter_ = 1
    loss_win, val_win = None, None
    val_accuracies = []

    for e in tqdm(range(1, epoch + 1), desc="Training the network (run:{})".format(run)):
        # Set the network to training mode
        net.train()
        avg_loss = 0.0

        # Run the training loop for one epoch
        for batch_idx, (data, data2, target) in tqdm(
            enumerate(data_loader), total=len(data_loader)
        ):
            # Load the data into the GPU if required
            data, data2, target = data.to(device), data2.to(device), target.to(device)

            optimizer.zero_grad()
            if supervision == "full":
                # output, origin, layers_outpout, masked_output = net(data, data2)  #打印各层特征图
                output = net(data, data2)
                loss = criterion(output, target)
            elif supervision == "semi":
                outs = net(data, data2)
                output, rec = outs
                loss = criterion[0](output, target) + net.aux_loss_weight * criterion[
                    1
                ](rec, data)
            else:
                raise ValueError(
                    'supervision mode "{}" is unknown.'.format(supervision)
                )
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            losses[iter_] = loss.item()
            mean_losses[iter_] = np.mean(losses[max(0, iter_ - 100) : iter_ + 1])#计算最近100次迭代中损失的平均值

            if display_iter and iter_ % display_iter == 0:
                string = "Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}"
                string = string.format(
                    e,
                    epoch,
                    batch_idx * len(data),
                    len(data) * len(data_loader),
                    100.0 * batch_idx / len(data_loader),
                    mean_losses[iter_],
                )
                update = None if loss_win is None else "append"
                loss_win = display.line(
                    X=np.arange(iter_ - display_iter, iter_),
                    Y=mean_losses[iter_ - display_iter : iter_],
                    win=loss_win,
                    update=update,
                    opts={
                        "title": "Training loss run:{}".format(run),
                        "xlabel": "Iterations",
                        "ylabel": "Loss",
                    },
                )
                tqdm.write(string)

                if len(val_accuracies) > 0:
                    val_win = display.line(
                        Y=np.array(val_accuracies),
                        X=np.arange(len(val_accuracies)),
                        win=val_win,
                        opts={
                            "title": "Validation accuracy run:{}".format(run),
                            "xlabel": "Epochs",
                            "ylabel": "Accuracy",
                        },
                    )
            iter_ += 1
            del (data, target, loss, output)

        ###打印最后一代的各层特征图，自增
        # if e == epoch:  ###每一次训练输出一次, 即最后一代输出
        #     # origin (B,C,H,W)     layers_outpout, masked_output list:num_layers-1
        #     show_featuremap("origin", origin, bands, display)
        #     # 获取输出列表这是一个列表，里面每个代表了每层的输出
        #     for i, (layers_fm, masked_fm) in enumerate(zip(layers_outpout, masked_output)):
        #         if i == 0 or i == 2:
        #             show_featuremap("layers{}".format(i), layers_fm, bands, display)
        #             show_featuremap("masked{}".format(i), masked_fm, bands, display)

        # Update the scheduler
        avg_loss /= len(data_loader)
        if val_loader is not None:
            val_acc = val(net, val_loader, device=device, supervision=supervision)
            val_accuracies.append(val_acc)
            metric = -val_acc
        else:
            metric = avg_loss

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(metric)
        elif scheduler is not None:
            scheduler.step()

        # Save the weights
        # if e % save_epoch == 0:
        #     save_model(
        #         savename,
        #         net,
        #         camel_to_snake(str(net.__class__.__name__)),
        #         data_loader.dataset.name,
        #         run=run,
        #         epoch=e,
        #         metric=abs(metric),
        #     )

        #save optimal weights
        if abs(metric) >= best_val_acc:
            best_val_acc = abs(metric)
            best_model_wts = copy.deepcopy(net.state_dict())
            if abs(metric) > best_val_acc or e % save_epoch == 0:##添加条件以减少checkponint文件（这样有时候会筛掉非规定代的最好结果的checkpoint，但是test时用的仍是最好模型的模型参数，因此这方面无须担心
                save_model(
                        savename,
                        net,
                        camel_to_snake(str(net.__class__.__name__)),
                        data_loader.dataset.name,
                        train_state = 'train',
                        type = 'best_epoch',##'final_epoch', 'best_epoch'
                        run=run,
                        epoch=e,
                        metric=abs(metric),
                    )

        # save final weights
        if e == final_epoch:
            save_model(
                savename,
                net,
                camel_to_snake(str(net.__class__.__name__)),
                data_loader.dataset.name,
                train_state='train',
                type='final_epoch',  ##'final_epoch', 'best_epoch'
                run=run,
                epoch=e,
                metric=abs(metric),
            )

    return best_model_wts

def save_model(savename, model, model_name, dataset_name, train_state, type, **kwargs):
    model_dir = "./checkpoints/" + model_name + "/" + dataset_name + "/" + train_state + "/" + type + "/"
    """
    Using strftime in case it triggers exceptions on windows 10 system
    """
    time_str = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    if isinstance(model, torch.nn.Module):
        filename = time_str + savename + "_run{run}_epoch{epoch}_{metric:.2f}".format(
            **kwargs
        )
        tqdm.write("Saving neural network weights in {}".format(filename))
        torch.save(model.state_dict(), model_dir + filename + ".pth")
    else:
        filename = time_str
        tqdm.write("Saving model params in {}".format(filename))
        joblib.dump(model, model_dir + filename + ".pkl")


def test(run, net, img1, img2, hyperparams):
    """
    Test a model on a specific image
    """
    net.eval()
    patch_size = hyperparams["patch_size"]
    center_pixel = hyperparams["center_pixel"]
    batch_size, device = hyperparams["batch_size"], hyperparams["device"]
    n_classes = hyperparams["n_classes"]
    if hyperparams["applyPCA"] == True:
        img1 = applyPCA(img1, 3)

    kwargs = {
        "step": hyperparams["test_stride"],
        "window_size": (patch_size, patch_size),
    }
    #构建概率数组，对HSI图像中的每一个像素，都先预留出对每一类概率预测结果的空空来
    probs = np.zeros(img1.shape[:2] + (n_classes,))

    iterations = count_sliding_window(img1, img2, **kwargs) // batch_size
    for batch in tqdm(
        grouper(batch_size, sliding_window(img1, img2, **kwargs)),
        total=(iterations),
        desc="Inference on the image (run:{})".format(run),
    ):
        with torch.no_grad():
            if patch_size == 1:
                '''因为patchsize也就是窗口大小=1，所以每一个窗口中其实就只有这一个数据'''
                data = [b[0][0, 0] for b in batch]
                data = np.copy(data)
                data = torch.from_numpy(data)

                data2 = [b[1][0, 0] for b in batch]
                data2 = np.copy(data2)
                data2 = torch.from_numpy(data2)
            else:
                data = [b[0] for b in batch]
                data = np.copy(data)
                data = data.transpose(0, 3, 1, 2)
                data = torch.from_numpy(data)
                # data = data.unsqueeze(1)

                data2 = [b[1] for b in batch]
                data2 = np.copy(data2)
                data2 = data2.transpose(0, 3, 1, 2)
                data2 = torch.from_numpy(data2)
                # data2 = data2.unsqueeze(1)

            indices = [b[2:] for b in batch]#列表中每一个元素都是(x, y, w, h)(左上角坐标们，宽高)
            data = data.to(device)
            data2 = data2.to(device)
            output= net(data, data2)
            if isinstance(output, tuple):  # For multiple outputs
                output = output[0]
            output = output.to("cpu")

            if patch_size == 1 or center_pixel:
                output = output.numpy()
            else:
                output = np.transpose(output.numpy(), (0, 2, 3, 1))
            for (x, y, w, h), out in zip(indices, output):
                if center_pixel:
                    probs[x + w // 2, y + h // 2] += out
                else:
                    probs[x : x + w, y : y + h] += out
    return probs


def val(net, data_loader, device="cpu", supervision="full"):
    # TODO : fix me using metrics()
    accuracy, total = 0.0, 0.0
    ignored_labels = data_loader.dataset.ignored_labels
    for batch_idx, (data, data2, target) in enumerate(data_loader):
        with torch.no_grad():
            # Load the data into the GPU if required
            data, data2, target = data.to(device), data2.to(device), target.to(device)
            if supervision == "full":
                output= net(data, data2)
            elif supervision == "semi":
                outs = net(data)
                output, rec = outs

            if isinstance(output, tuple):   # For multiple outputs
                output = output[0]
            _, output = torch.max(output, dim=1)
            for out, pred in zip(output.view(-1), target.view(-1)):
                if out.item() in ignored_labels:
                    continue
                else:
                    accuracy += out.item() == pred.item()
                    total += 1
    return accuracy / total

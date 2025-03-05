# -*- coding: utf-8 -*-
"""
DEEP LEARNING FOR MULTIMODAL REMOTE SENSING.
(https://github.com/likyoo/Multimodal-Remote-Sensing-Toolkit)

This script allows the user to run several deep models
against various multimodal datasets. It is developed on
the top of DeepHyperX (https://github.com/nshaud/DeepHyperX)

"""
# Python 2/3 compatiblity
from __future__ import print_function
from __future__ import division

# Torch
import copy

import scipy
import torch
import torch.utils.data as data

# Numpy, scipy, scikit-image, spectral
import numpy as np

# Visualization
import seaborn as sns
import visdom

from utils import (
    metrics,
    convert_to_color_,
    convert_from_color_,
    display_dataset,
    display_lidar_data,
    display_predictions,
    explore_spectrums,
    plot_spectrums,
    sample_gt,
    show_results,
    compute_imf_weights,
    get_device,
    restore_from_padding,
    seed_torch,
)
from datasets import get_dataset, MultiModalX, open_file, DATASETS_CONFIG, MultiModalX_all
from model_utils import get_model, train, test, pretrain
import argparse

import sys

# 将print输出重定向到文件
filename = './results/trytry.txt'
sys.stdout = open(filename, 'w')##Ablation2_3direction

# 分割路径以获取文件名, 自增
file_name = filename.split('/')[-1]
# 分割文件名以获取所需部分，自增
required_part = file_name.split('.')[0]

# DATASETS_CONFIG是一个字典， 其键值对中的值v 仍为字典
# 最后生成的列表要么是v["name"]， 要么是 k
dataset_names = [
    v["name"] if "name" in v.keys() else k for k, v in DATASETS_CONFIG.items()
]


# Argument parser for CLI interaction(命令行交互)
# 创建parser对象
parser = argparse.ArgumentParser(
    description="Run deep learning experiments on" " various hyperspectral datasets"
)
# choice： 需提供一个列表， 在用户在命令行输入 --dataset时， 对用户的输入进行检查， 不在列表中就报错
parser.add_argument(
    "--dataset", type=str, default='MUUFL', choices=dataset_names,
    help="Dataset to use."
          "Houston2013, Trento, MUUFL, Augsburg， IP, Salinas, PaviaU, Houston2018"
)
parser.add_argument(
    "--applyPCA", type=bool,
    help="optional, if absent will be set by the model"
         "得手动改datasets.py  287行"
         "model_utils.py def test(run, net, img1, img2, hyperparams):里的"
         "model_utils.py的每个模型里的applyPCA参数"
)
parser.add_argument(
    "--model",
    type=str,
    default="Multimodality_Mamba",
    help="Model to train. Available:\n"
    "EndNet, "
    "Early_fusion_CNN, "
    "Middle_fusion_CNN, "
    "Late_fusion_CNN, "
    "Cross_fusion_CNN, "
    "FusAtNet, "
    "S2ENet, "
    "FI_CNN, "
    "ResNet18, "
    "S2ENet_ResNet18, "
    "multiScaleCNN, "
    "FI_CNN3D, "
    "VIT, "
    "FICNN_VIT, "
    "SupConResNet, "
    "HybridSN, "
    "RSMamba_complete,"
    "CascadeRSMamba_complete,"
    "Multimodality_Mamba, "
    "MHST, "
    "GLT_Net, 暂时用不了，这个的数据集不好弄"
    "HCTnet, "
    "MFT, "
    "SpectralFormer,"
    "PCCNet, "
    "groupViT, "
    "moco_based_NNCNet, "
    "Others, ",
)
parser.add_argument(
    "--folder",
    type=str,
    help="Folder where to store the "
    "datasets (defaults to the current working directory).",
    default="./Datasets/",
)
parser.add_argument(
    "--cuda",
    type=int,
    default=0,
    help="Specify CUDA device (defaults to -1, which learns on CPU)",
)
parser.add_argument("--runs", type=int, default=10, help="Number of runs (default: 1)")
# 权重初始化文件
parser.add_argument(
    "--restore",
    type=str,
    default=None,
    help="Weights to use for initialization, e.g. a checkpoint",
)
parser.add_argument(
    "--seed",
    type=int,
    default=1,###0
    help="Set random seed",
)

# Dataset options
# 创建了一个参数组， 其中的参数 可以在命令行中单独指定， 只是他们会显示在 Dataset 这个分组中
group_dataset = parser.add_argument_group("Dataset")
group_dataset.add_argument(
    "--train_val_split",
    type=float,
    default=1,
    help="Percentage of samples to use for training and validation, "
         "'1' means all training data are used to train",
)
group_dataset.add_argument(
    "--training_sample",
    type=float,
    default=20,
    help="Percentage of samples to use for training (default: 10%%) and testing"
         "if sampling_mode =='random_fixednumber', this parameter represents number of trainingset in each class ",
)
# 样本采样方式
group_dataset.add_argument(
    "--sampling_mode",
    type=str,
    help='fixed'
          'fixed'     
         'disjoint'
         'random_fixednumber',
    default= 'random_fixednumber',
)
group_dataset.add_argument(
    "--train_set",
    type=str,
    default=None,
    help="Path to the train ground truth (optional, this "
    "supersedes the --sampling_mode option)",
)
group_dataset.add_argument(
    "--test_set",
    type=str,
    default=None,
    help="Path to the test set (optional, by default "
    "the test_set is the entire ground truth minus the training)",
)

# Training options
group_train = parser.add_argument_group("Training")
group_train.add_argument(
    "--epoch",
    type=int,
    help="Training epochs (optional, if" " absent will be set by the model)",
)
# 空间邻域
group_train.add_argument(
    "--patch_size",
    type=int,
    help="Size of the spatial neighbourhood (optional, if "
    "absent will be set by the model)",
)
group_train.add_argument(
    "--lr", type=float, help="Learning rate, set by the model if not specified."
)

# 逆中位频率类别平衡
group_train.add_argument(
    "--class_balancing",
    action="store_true",
    help="Inverse median frequency class balancing (default = False)",
)
group_train.add_argument(
    "--batch_size",
    type=int,
    help="Batch size (optional, if absent will be set by the model",
)
group_train.add_argument(
    "--test_stride",
    type=int,
    default=1,
    help="Sliding window step stride during inference (default = 1)",
)

# Data augmentation parameters
group_da = parser.add_argument_group("Data augmentation")
group_da.add_argument(
    "--flip_augmentation", action="store_true", help="Random flips (if patch_size > 1)"
)
# 辐射噪声，光照
group_da.add_argument(
    "--radiation_augmentation",
    action="store_true",
    help="Random radiation noise (illumination)",
)
# 光谱（Spectra）之间的随机混合：
# 在高光谱图像处理中，光谱代表了图像中不同波段的特征。光谱之间的随机混合是指将不同波段的光谱数据进行随机组合或混合，以创造新的光谱样本。
# 这种混合可以通过对不同波段的数据进行加权和组合来实现，以产生具有多样化特征的新样本。
group_da.add_argument(
    "--mixture_augmentation", action="store_true", help="Random mixes between spectra"
)
# 探索性因素，不管
parser.add_argument(
    "--with_exploration", action="store_true", help="See data exploration visualization"
)
# 本来应该能下，这里下不了
parser.add_argument(
    "--download",
    type=str,
    default=None,
    nargs="+",
    choices=dataset_names,
    help="Download the specified datasets and quits.",
)


args = parser.parse_args()

CUDA_DEVICE = get_device(args.cuda)

# % of training samples
SAMPLE_PERCENTAGE = args.training_sample
SAMPLE_TRAIN_VALID = args.train_val_split
# Data augmentation ?
FLIP_AUGMENTATION = args.flip_augmentation
RADIATION_AUGMENTATION = args.radiation_augmentation
MIXTURE_AUGMENTATION = args.mixture_augmentation
# Dataset name
DATASET = args.dataset
# Model name
MODEL = args.model
# Number of runs (for cross-validation)
N_RUNS = args.runs
# Spatial context size (number of neighbours in each spatial direction)
PATCH_SIZE = args.patch_size
# Add some visualization of the spectra ?
DATAVIZ = args.with_exploration
# Target folder to store/download/load the datasets
FOLDER = args.folder
# Number of epochs to run
EPOCH = args.epoch
# Sampling mode, e.g random sampling
SAMPLING_MODE = args.sampling_mode
# Pre-computed weights to restore
CHECKPOINT = args.restore
# Learning rate for the SGD
LEARNING_RATE = args.lr
# Automated class balancing
CLASS_BALANCING = args.class_balancing
# Training ground truth file
TRAIN_GT = args.train_set
# Testing ground truth file
TEST_GT = args.test_set
TEST_STRIDE = args.test_stride

# set random seed
seed = np.arange(N_RUNS, dtype='int').tolist()
# seed_torch(seed=args.seed) #原

# 既检查了参数是否被提供（不为 None），又确保参数的长度大于 0，即确保参数非空。
if args.download is not None and len(args.download) > 0:
    for dataset in args.download:
        get_dataset(dataset, target_folder=FOLDER)
    quit()

viz = visdom.Visdom(env=DATASET + " " + MODEL)
if not viz.check_connection:
    print("Visdom is not connected. Did you run 'python -m visdom.server' ?")

# 将命令行获取的参数， 以字典形式传递给hyperparams
hyperparams = vars(args)
# Load the dataset
img1, img2, gt, LABEL_VALUES, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(DATASET, FOLDER)

# 这些操作通常用于为数据的可视化或标注准备颜色信息。 used
# if palette is None:
#     # Generate color palette
#     palette = {0: (0, 0, 0)}
#     for k, color in enumerate(sns.color_palette("hls", len(LABEL_VALUES) - 1)):
#         palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
# invert_palette = {v: k for k, v in palette.items()}

if palette is None:
    # Generate color palette
    palette = {0: (0, 0, 0)}
    for k, color in enumerate(sns.color_palette("pastel", 10) + sns.color_palette("bright", len(LABEL_VALUES) - 1 - 10)):
        palette[k + 1] = tuple(np.asarray(255 * np.array(color), dtype="uint8"))
invert_palette = {v: k for k, v in palette.items()}


def convert_to_color(x):
    return convert_to_color_(x, palette=palette)

def convert_from_color(x):
    return convert_from_color_(x, palette=invert_palette)


# Show the image and the ground truth
display_dataset(img1, gt, RGB_BANDS, LABEL_VALUES, palette, viz)
display_lidar_data(img2, viz)
color_gt = convert_to_color(gt)

#显示gt图
# plt.figure(dpi=300)
# plt.imshow(color_gt)
# plt.axis("off")
# plt.show()


# 数据探索，先不管
if DATAVIZ:
    # Data exploration : compute and show the mean spectrums
    mean_spectrums = explore_spectrums(
        img1, gt, LABEL_VALUES, viz, ignored_labels=IGNORED_LABELS
    )
    plot_spectrums(mean_spectrums, viz, title="Mean spectrum/class")

# Number of classes
N_CLASSES = len(LABEL_VALUES)
# Number of bands (last dimension of the image tensor)
N_BANDS = (img1.shape[-1], img2.shape[-1])

# Instantiate the experiment based on predefined networks
hyperparams.update(
    {
        "n_classes": N_CLASSES,
        "n_bands": N_BANDS,
        "ignored_labels": IGNORED_LABELS,
        "device": CUDA_DEVICE,
    }
)
hyperparams = dict((k, v) for k, v in hyperparams.items() if v is not None)

results = []
# run the experiment several times
# 首先看有没有指定训练集和测试集， 如果分了， 就按照分的来； 如果没分， 就按照sample那个参数指定的百分比， 随机分出训练集和测试集
for run in range(N_RUNS):
    seed_torch(seed=seed[2])
    if TRAIN_GT is not None and TEST_GT is not None:
        train_gt = open_file(TRAIN_GT)['TRLabel']#['TRLabel']for Houston2013
        test_gt = open_file(TEST_GT)['TSLabel']#['TSLabel']for Houston2013
    elif TRAIN_GT is not None:
        train_gt = open_file(TRAIN_GT)
        test_gt = np.copy(gt)
        w, h = test_gt.shape
        # 令测试集中在训练集中大于零的部分 等于零
        test_gt[(train_gt > 0)[:w, :h]] = 0
    elif TEST_GT is not None:
        test_gt = open_file(TEST_GT)
    else:
        # Sample random training spectra
        train_gt, test_gt = sample_gt(gt, SAMPLE_PERCENTAGE, mode=SAMPLING_MODE, seed=seed[run])
        # scipy.io.savemat('20_TRLabel.mat', {'20_TRLabel': train_gt})
        # scipy.io.savemat('20_TSLabel.mat', {'20_TSLabel': test_gt})
    print(
        "{} samples selected (over {})".format(
            np.count_nonzero(train_gt), np.count_nonzero(gt)
        )
    )
    print(
        "Running an experiment with the {} model".format(MODEL),
        "run {}/{}".format(run + 1, N_RUNS),
    )

    # 在visdom里显示训练集， 测试集的真实标签
    #训练集，测试集的划分固定时可这么干；不固定时，当然还是每次都显示的好
    if(TRAIN_GT is None and TEST_GT is None):
        display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth", run=run)
        display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth", run=run)
    else:
        if (run == 0):
            display_predictions(convert_to_color(train_gt), viz, caption="Train ground truth")
            display_predictions(convert_to_color(test_gt), viz, caption="Test ground truth")
    # delete  所有的真实标签， 被删了这句
    # display_predictions(convert_to_color(open_file('Datasets/Houston2013/gt.mat')['gt']), viz, caption="ground truth")

    # 类别平衡这里， 没看
    if CLASS_BALANCING:
        weights = compute_imf_weights(train_gt, N_CLASSES, IGNORED_LABELS)
        hyperparams["weights"] = torch.from_numpy(weights)
    # Neural network  神经网络！！！！
    model, optimizer, loss, hyperparams = get_model(MODEL, **hyperparams)

    # Split train set in train/val
    # 把训练集分成训练集和验证集
    if SAMPLE_TRAIN_VALID != 1:
        train_gt, val_gt = sample_gt(train_gt, SAMPLE_TRAIN_VALID, mode="random")
    else:
        # Use all training data to train the model  不懂
        _, val_gt = sample_gt(train_gt, 0.95, mode="random")

    # Generate the dataset for other model
    train_dataset = MultiModalX(img1, img2, train_gt, **hyperparams)
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=hyperparams["batch_size"],
        # pin_memory=hyperparams['device'],
        shuffle=True,
        # drop_last=True###之前的时候都是不打开的，直到开始有pretrain的想法
    )
    val_dataset = MultiModalX(img1, img2, val_gt, **hyperparams)
    val_loader = data.DataLoader(
        val_dataset,
        # pin_memory=hyperparams['device'],
        batch_size=hyperparams["batch_size"],
        # drop_last=True###之前的时候都是不打开的，直到开始有pretrain的想法
    )


    print("超参数：\n",hyperparams)
    print("Network :")
    # with torch.no_grad():
    #     for input, input2,  _ in train_loader:
    #         break
    #     summary(model.to(hyperparams["device"]), [input.size()[1:], input2.size()[1:]])
        # We would like to use device=hyperparams['device'] altough we have
        # to wait for torchsummary to be fixed first.

    ##for load encoder except cls. head in pretrain
    # if CHECKPOINT is not None:
    #     model_dict = torch.load(CHECKPOINT)
    #     new_state_dict = {}
    #     for k, v in model_dict.items():
    #         if 'encoder_q' in k and 'encoder_q.head' not in k:
    #             prefix = "encoder_q."
    #             if k.startswith(prefix):
    #                 k = k[len(prefix):]
    #             new_state_dict[k] = v
    #     model.load_state_dict(new_state_dict, strict=False)

    ##for load all
    if CHECKPOINT is not None:
      model.load_state_dict(torch.load(CHECKPOINT))


    try:
        #train
        best_model_wts = train(
            required_part,
            run,
            RGB_BANDS,
            model,
            optimizer,
            loss,
            train_loader,
            hyperparams["epoch"],
            scheduler=hyperparams["scheduler"],
            device=hyperparams["device"],
            supervision=hyperparams["supervision"],  #####for best checkpoint
            val_loader=val_loader,
            display=viz
        )
    except KeyboardInterrupt:
        # Allow the user to stop the training
        pass

    model.load_state_dict(best_model_wts) #####for best checkpoint
    # 测试一下模型， 每个样本属于不同类别的概率（代码没看， 感觉正常来说是这样）
    #每个像素都测了
    probabilities= test(run, model, img1, img2, hyperparams)

    # 生成测试集的F1 score by class, confusion matrix
    try:
        prediction = np.argmax(probabilities, axis=-1)
        run_results = metrics(
            prediction,
            test_gt,
            ignored_labels=hyperparams["ignored_labels"],
            n_classes=N_CLASSES,
        )
    except:#不懂
        probabilities = restore_from_padding(probabilities, patch_size=[hyperparams["patch_size"], hyperparams["patch_size"]])
        prediction = np.argmax(probabilities, axis=-1)
        run_results = metrics(
            prediction,
            test_gt,
            ignored_labels=hyperparams["ignored_labels"],
            n_classes=N_CLASSES,
        )

    # 显示所有像素的预测结果
    color_prediction = convert_to_color(prediction)
    display_predictions(
        color_prediction,
        viz,
        caption="Prediction_All run:{}".format(run),
    )

    #仅显示有标签数据的预测结果
    mask = np.zeros(gt.shape, dtype="bool")
    for l in IGNORED_LABELS:
        mask[gt == l] = True
    prediction[mask] = 0#背景在预测结果中全置为0

    color_prediction = convert_to_color(prediction)
    display_predictions(
        color_prediction,
        viz,
        # gt=convert_to_color(test_gt),
        # caption="Prediction vs. test ground truth",
        caption="Prediction run:{}".format(run),
    )

    #flt_test_gt和flt_prediction变量分别存储了测试集的真实标签和模型的预测结果，并且它们都是一维数组。这种形状重塑操作通常用于在评估模型性能时对真实标签和预测结果进行比较。
    flt_test_gt = test_gt.reshape(-1)
    flt_prediction = prediction.reshape(-1)

    results.append(run_results)
    show_results(run, run_results, viz, label_values=LABEL_VALUES)

if N_RUNS > 1:
    show_results(run, results, viz, label_values=LABEL_VALUES, agregated=True)


# 恢复标准输出
sys.stdout = sys.__stdout__

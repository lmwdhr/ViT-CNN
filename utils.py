# -*- coding: utf-8 -*-
import heapq
import math
import random
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.model_selection
import seaborn as sns
import itertools
import spectral
import visdom
import matplotlib.pyplot as plt
from scipy import io, misc
import os
import re
import torch
from sklearn.decomposition import PCA
import torch.nn.functional as F

#根据scheduler降低学习率，from nncnet
def adjust_learning_rate(optimizer, epoch, kwargs):
    """Decay the learning rate based on schedule"""
    lr = kwargs["lr"]
    if kwargs["cos"]:  # cosine lr schedule
        lr *= 0.5 * (1. + math.cos(math.pi * epoch / kwargs["epoch"]))
    else:  # stepwise lr schedule
        for milestone in kwargs["scheduler"]:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

'''寻找形状为（b, h*w）的张量中，批量中每张图中的前k个最大数
tensor形状为（b, k）, 返回形状为[[b1位置索引],[把b2位置索引]....], datatype:numpy??'''
def find_top_k_indices(tensor, k):
    b, hw = tensor.shape
    top_k_indices = torch.zeros((b, k), dtype=torch.long).to('cuda:0')  # 确保索引为整数类型

    # 对每组数进行降序排序，获取索引
    for i in range(b):
        group = tensor[i]
        # argsort返回的是降序排序后的索引
        sorted_indices = torch.argsort(group, descending=True)  # 指定降序排序
        # 选择前k个索引
        top_k_indices[i] = sorted_indices[:k]

    return top_k_indices


def extract_elements_by_indices(tensor, indices, k):
    b, c, w, h = tensor.shape
    tensor = tensor.reshape((b, c, -1))
    extracted_elements = torch.zeros((b, c, k), dtype=torch.float32).to('cuda:0')  # 初始化一个形状为(b, k)的数组来存储提取的元素

    for i in range(b):
        group_indices = indices[i]
        group = tensor[i]
        # 使用高级索引提取对应元素
        extracted_elements[i] = group[:, group_indices]

    return extracted_elements

def extract_windows_by_indices(patch1, patch2, indices, k):###根据patch1的索引提取patch2的窗口
    b, c, w1, h2 = patch1.shape#小
    _, _, w2, h2 = patch2.shape#大
    pad = (w2-w1) // 2###4
    indices_pos = torch.arange(w2*h2, dtype=torch.float32).to('cuda:0').reshape((w2, h2))[pad: w1+pad, pad: w1+pad].reshape((-1))
    extracted_indices = torch.zeros((b, k), dtype=torch.int).to('cuda:0')  # 初始化一个形状为(b, k)的数组来存储提取的元素
    topk_windows = torch.zeros((b, k, c, 9, 9), dtype=torch.float32).to('cuda:0')

    for i in range(b):
        group_indices = indices[i]
        # 使用高级索引提取对应元素
        extracted_indices[i] = indices_pos[group_indices]
        for j in range(k):
            # 计算行索引i
            index_i = extracted_indices[i, j] // w2
            # 计算列索引j
            index_j = extracted_indices[i, j] % w2
            new_patch = patch2[i, :, index_i - 4:index_i + 5, index_j - 4:index_j + 5]
            topk_windows  = topk_windows.clone()
            topk_windows[i, j] = new_patch

    return topk_windows ###形状（b, k, c, h, w）

def applyPCA(X, numComponents):
    """
    apply PCA to the image to reduce dimensionality
  """
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX

def get_device(ordinal):
    # Use GPU ?
    if ordinal < 0:
        print("Computation on CPU")
        device = torch.device('cpu')
    elif torch.cuda.is_available():
        print("Computation on CUDA GPU device {}".format(ordinal))
        device = torch.device('cuda:{}'.format(ordinal))
    else:
        print("/!\\ CUDA was requested but is not available! Computation will go on CPU. /!\\")
        device = torch.device('cpu')
    return device


def open_file(dataset):
    _, ext = os.path.splitext(dataset)
    ext = ext.lower()
    if ext == '.mat':
        # Load Matlab array
        return io.loadmat(dataset)
    elif ext == '.tif' or ext == '.tiff':
        # Load TIFF file
        return misc.imread(dataset)
    elif ext == '.hdr':
        img = spectral.open_image(dataset)
        return img.load()
    else:
        raise ValueError("Unknown file format: {}".format(ext))

def convert_to_color_(arr_2d, palette=None):
    """Convert an array of labels to RGB color-encoded image.

    Args:
        arr_2d: int 2D array of labels
        palette: dict of colors used (label number -> RGB tuple)

    Returns:
        arr_3d: int 2D images of color-encoded labels in RGB format

    """
    arr_3d = np.zeros((arr_2d.shape[0], arr_2d.shape[1], 3), dtype=np.uint8)
    if palette is None:
        raise Exception("Unknown color palette")

    for c, i in palette.items():
        m = arr_2d == c
        arr_3d[m] = i

    return arr_3d


def convert_from_color_(arr_3d, palette=None):
    """Convert an RGB-encoded image to grayscale labels.

    Args:
        arr_3d: int 2D image of color-coded labels on 3 channels
        palette: dict of colors used (RGB tuple -> label number)

    Returns:
        arr_2d: int 2D array of labels

    """
    if palette is None:
        raise Exception("Unknown color palette")

    arr_2d = np.zeros((arr_3d.shape[0], arr_3d.shape[1]), dtype=np.uint8)

    for c, i in palette.items():
        m = np.all(arr_3d == np.array(c).reshape(1, 1, 3), axis=2)
        arr_2d[m] = i

    return arr_2d


def display_predictions(pred, vis, gt=None, caption="", run=0):
    if gt is None:
        vis.images([np.transpose(pred, (2, 0, 1))],
                    opts={'caption': caption + "run:{}".format(run) })
    else:
        vis.images([np.transpose(pred, (2, 0, 1)),
                    np.transpose(gt, (2, 0, 1))],
                    nrow=2,
                    opts={'caption': caption + "run:{}".format(run)})

def display_dataset(img, gt, bands, labels, palette, vis):
    """Display the specified dataset.

    Args:
        img: 3D hyperspectral image
        gt: 2D array labels
        bands: tuple of RGB bands to select
        labels: list of label class names
        palette: dict of colors
        display (optional): type of display, if any

    """
    print("Image 1 has dimensions {}x{} and {} channels".format(*img.shape))
    rgb = spectral.get_rgb(img, bands)
    rgb /= np.max(rgb)
    rgb = np.asarray(255 * rgb, dtype='uint8')

    # Display the RGB composite image
    caption = "RGB (bands {}, {}, {})".format(*bands)
    # send to visdom server
    vis.images([np.transpose(rgb, (2, 0, 1))],
                opts={'caption': caption})

def display_lidar_data(img, vis):
    """Display the LiDAR data.
        Args:
            img: 2D LiDAR image
        """
    print("Image 2 has dimensions {}x{} and {} channels".format(*img.shape))
    gray = img / np.max(img)
    gray = np.asarray(255 * gray, dtype='uint8')

    # Display the lidar composite image
    caption = "LiDAR"
    # send to visdom server
    vis.images([np.transpose(gray, (2, 0, 1))],
               opts={'caption': caption})


def explore_spectrums(img, complete_gt, class_names, vis,
                      ignored_labels=None):
    """Plot sampled spectrums with mean + std for each class.

    Args:
        img: 3D hyperspectral image
        complete_gt: 2D array of labels
        class_names: list of class names
        ignored_labels (optional): list of labels to ignore
        vis : Visdom display
    Returns:
        mean_spectrums: dict of mean spectrum by class

    """
    mean_spectrums = {}
    for c in np.unique(complete_gt):
        if c in ignored_labels:
            continue
        mask = complete_gt == c
        class_spectrums = img[mask].reshape(-1, img.shape[-1])
        step = max(1, class_spectrums.shape[0] // 100)
        fig = plt.figure()
        plt.title(class_names[c])
        # Sample and plot spectrums from the selected class
        for spectrum in class_spectrums[::step, :]:
            plt.plot(spectrum, alpha=0.25)
        mean_spectrum = np.mean(class_spectrums, axis=0)
        std_spectrum = np.std(class_spectrums, axis=0)
        lower_spectrum = np.maximum(0, mean_spectrum - std_spectrum)
        higher_spectrum = mean_spectrum + std_spectrum

        # Plot the mean spectrum with thickness based on std
        plt.fill_between(range(len(mean_spectrum)), lower_spectrum,
                         higher_spectrum, color="#3F5D7D")
        plt.plot(mean_spectrum, alpha=1, color="#FFFFFF", lw=2)
        vis.matplot(plt)
        mean_spectrums[class_names[c]] = mean_spectrum
    return mean_spectrums


def plot_spectrums(spectrums, vis, title=""):
    """Plot the specified dictionary of spectrums.

    Args:
        spectrums: dictionary (name -> spectrum) of spectrums to plot
        vis: Visdom display
    """
    win = None
    for k, v in spectrums.items():
        n_bands = len(v)
        update = None if win is None else 'append'
        win = vis.line(X=np.arange(n_bands), Y=v, name=k, win=win, update=update,
                       opts={'title': title})


def build_dataset(mat, gt, ignored_labels=None):
    """Create a list of training samples based on an image and a mask.

    Args:
        mat: 3D hyperspectral matrix to extract the spectrums from
        gt: 2D ground truth
        ignored_labels (optional): list of classes to ignore, e.g. 0 to remove
        unlabeled pixels
        return_indices (optional): bool set to True to return the indices of
        the chosen samples

    """
    samples = []
    labels = []
    # Check that image and ground truth have the same 2D dimensions
    assert mat.shape[:2] == gt.shape[:2]

    for label in np.unique(gt):
        if label in ignored_labels:
            continue
        else:
            indices = np.nonzero(gt == label)
            samples += list(mat[indices])
            labels += len(indices[0]) * [label]
    return np.asarray(samples), np.asarray(labels)


def get_random_pos(img, window_shape):
    """ Return the corners of a random window in the input image

    Args:
        img: 2D (or more) image, e.g. RGB or grayscale image
        window_shape: (width, height) tuple of the window

    Returns:
        xmin, xmax, ymin, ymax: tuple of the corners of the window

    """
    w, h = window_shape
    W, H = img.shape[:2]
    x1 = random.randint(0, W - w - 1)
    x2 = x1 + w
    y1 = random.randint(0, H - h - 1)
    y2 = y1 + h
    return x1, x2, y1, y2


def padding_image(image, patch_size=None, mode="symmetric", constant_values=None):
    """Padding an input image.
    Modified at 2020.11.16. If you find any issues, please email at mengxue_zhang@hhu.edu.cn with details.
    填充的是：复制了离他最近的那个值
    Args:
        image: 2D+ image with a shape of [h, w, ...],
        The array to pad
        patch_size: optional, a list include two integers, default is [1, 1] for pure spectra algorithm,两个参数分别指两个方向横竖上的padding大小。如果你想让图片宽从1变3，并且是以左右边各增加1的方式，那么参数设置应为patch_size=[2, 2], mode="symmetric"
        The patch size of the algorithm
        mode: optional, str or function, default is "symmetric",
        Including 'constant', 'reflect', 'symmetric', more details see np.pad()
        constant_values: optional, sequence or scalar, default is 0,
        Used in 'constant'.  The values to set the padded values for each axis
    Returns:
        padded_image with a shape of [h + patch_size[0] // 2 * 2, w + patch_size[1] // 2 * 2, ...]

    """
    if patch_size is None:
        patch_size = [1, 1]
    h = patch_size[0] // 2
    w = patch_size[1] // 2
    pad_width = [[h, h], [w, w]]
    [pad_width.append([0, 0]) for i in image.shape[2:]]
    padded_image = np.pad(image, pad_width, mode=mode)
    return padded_image

def restore_from_padding(image, patch_size=None):
    if patch_size is None:
        patch_size = [1, 1]
    h = patch_size[0] // 2
    w = patch_size[1] // 2
    W, H = image.shape[:2]
    restore_img = image[w:W-w, h:H-h, :]

    return restore_img


def sliding_window(image1, image2, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image1.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    """
    Compensate one for the stop value of range(...). because this function does not include the stop value.
    Two examples are listed as follows.
    When step = 1, supposing w = h = 3, W = H = 7, and step = 1.
    Then offset_w = 0, offset_h = 0.
    In this case, the x should have been ranged from 0 to 4 (4-6 is the last window),
    i.e., x is in range(0, 5) while W (7) - w (3) + offset_w (0) + 1 = 5. Plus one !
    Range(0, 5, 1) equals [0, 1, 2, 3, 4].

    When step = 2, supposing w = h = 3, W = H = 8, and step = 2.
    Then offset_w = 1, offset_h = 1.
    In this case, x is in [0, 2, 4] while W (8) - w (3) + offset_w (1) + 1 = 6. Plus one !
    Range(0, 6, 2) equals [0, 2, 4]/

    Same reason to H, h, offset_h, and y.
    """
    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image1[x:x + w, y:y + h], image2[x:x + w, y:y + h], x, y, w, h#默认返回的是这行
            else:
                yield x, y, w, h


def count_sliding_window(top, top2, step=10, window_size=(20, 20)):
    """ Count the number of windows in an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    sw = sliding_window(top, top2, step, window_size, with_data=False)
    return sum(1 for _ in sw)
#####自写
def padding_image_tensor(image, patch_size=None, mode='replicate', constant_values=None):
    """Padding an input image.
    Modified at 2020.11.16. If you find any issues, please email at mengxue_zhang@hhu.edu.cn with details.
    填充的是：复制了离他最近的那个值
    Args:
        image: 2D+ image with a shape of [h, w, ...],
        The tensor to pad
        patch_size: optional, a list include two integers, default is [1, 1] for pure spectra algorithm,
            two parameters respectively refer to the padding size in both horizontal and vertical directions.
            If you want to change the width of the image from 1 to 3, and do it by adding 1 to both left and right,
            then the parameter should be set to patch_size=[2, 2], mode="symmetric".
        mode: optional, str or function, default is "symmetric",
            Including 'constant', 'reflect', 'symmetric', more details see torch.nn.functional.pad()
        constant_values: optional, sequence or scalar, default is 0,
            Used in 'constant'. The values to set the padded values for each axis
    Returns:
        padded_image with a shape of [h + patch_size[0] // 2 * 2, w + patch_size[1] // 2 * 2, ...]
    """
    if patch_size is None:
        patch_size = [1, 1]
    h = patch_size[0] // 2
    w = patch_size[1] // 2
    image = image.unsqueeze(0)
    pad_width = (h, h, w, w)  # PyTorch pad function uses a different format for pad_width
    # Add [0, 0] for each additional dimension
    pad_width += ((0, 0),) * (len(image.shape) - 3)

    if constant_values is not None:
        constant_values = torch.tensor(constant_values).to(
            image.device)  # Ensure constant_values is on the same device as image

    padded_image = F.pad(image, pad_width, mode=mode)
    return padded_image.squeeze(0)
#####自写
def sliding_window_singleimage(image, step=10, window_size=(20, 20), with_data=True):
    """Sliding window generator over an input image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral [h, w, ....]
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image
    w, h = window_size
    W, H = image.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step
    """
    Compensate one for the stop value of range(...). because this function does not include the stop value.
    Two examples are listed as follows.
    When step = 1, supposing w = h = 3, W = H = 7, and step = 1.
    Then offset_w = 0, offset_h = 0.
    In this case, the x should have been ranged from 0 to 4 (4-6 is the last window),
    i.e., x is in range(0, 5) while W (7) - w (3) + offset_w (0) + 1 = 5. Plus one !
    Range(0, 5, 1) equals [0, 1, 2, 3, 4].

    When step = 2, supposing w = h = 3, W = H = 8, and step = 2.
    Then offset_w = 1, offset_h = 1.
    In this case, x is in [0, 2, 4] while W (8) - w (3) + offset_w (1) + 1 = 6. Plus one !
    Range(0, 6, 2) equals [0, 2, 4]/

    Same reason to H, h, offset_h, and y.
    """
    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            if with_data:
                yield image[x:x + w, y:y + h], x, y, w, h#默认返回的是这行
            else:
                yield x, y, w, h
#####自写
def generate_windows(image, step=10, window_size=(20, 20), padding=1):
    """ generate windows from an image.

    Args:
        image: 2D+ image to slide the window on, e.g. RGB or hyperspectral, ...
        step: int stride of the sliding window
        window_size: int tuple, width and height of the window
    Returns:
        int number of windows
    """
    image_padding = padding_image_tensor(image, patch_size=(padding * 2, padding * 2))
    windows = []

    # print("image_padding：", image_padding)

    sw = sliding_window_singleimage(image_padding, step, window_size, with_data=True)
    for window, _, _, _, _ in sw:
        windows.append(window)
    return windows
#####自写
def adding_windows_singleimage(image, mask, step=10, window_size=(20, 20), padding=1):
    """using windows from window generator to form a new image.

    Args:
        image: A zero image uesd as base whiteboard. It shape is same as the padding image. 2D+ image to slide the window on, e.g. RGB or hyperspectral [h, w, ....]
        step: int stride of the sliding window used in window generator
        window_size: int tuple, width and height of the window used in window generator
        with_data (optional): bool set to True to return both the data and the
        corner indices
    Yields:
        ([data], x, y, w, h) where x and y are the top-left corner of the
        window, (w,h) the window size

    """
    # slide a window across the image_padding
    image_padding = padding_image_tensor(image, patch_size=(padding * 2, padding * 2))
    w, h = window_size
    W, H = image_padding.shape[:2]
    offset_w = (W - w) % step
    offset_h = (H - h) % step

    i=0
    """
    Compensate one for the stop value of range(...). because this function does not include the stop value.
    Two examples are listed as follows.
    When step = 1, supposing w = h = 3, W = H = 7, and step = 1.
    Then offset_w = 0, offset_h = 0.
    In this case, the x should have been ranged from 0 to 4 (4-6 is the last window),
    i.e., x is in range(0, 5) while W (7) - w (3) + offset_w (0) + 1 = 5. Plus one !
    Range(0, 5, 1) equals [0, 1, 2, 3, 4].

    When step = 2, supposing w = h = 3, W = H = 8, and step = 2.
    Then offset_w = 1, offset_h = 1.
    In this case, x is in [0, 2, 4] while W (8) - w (3) + offset_w (1) + 1 = 6. Plus one !
    Range(0, 6, 2) equals [0, 2, 4]/

    Same reason to H, h, offset_h, and y.
    """
    for x in range(0, W - w + offset_w + 1, step):
        if x + w > W:
            x = W - w
        for y in range(0, H - h + offset_h + 1, step):
            if y + h > H:
                y = H - h
            image_padding[x:x + w, y:y + h] = image_padding[x:x + w, y:y + h] + mask[i]#默认返回的是这行
            i = i+1
    return image_padding



def grouper(n, iterable):
    """ Browse an iterable by grouping n elements by n elements.

    Args:
        n: int, size of the groups
        iterable: the iterable to Browse
    Yields:
        chunk of n elements from the iterable

    """
    it = iter(iterable)
    while True:
        chunk = tuple(itertools.islice(it, n))
        if not chunk:
            return
        yield chunk


def metrics(prediction, target, ignored_labels=[], n_classes=None):
    """Compute and print metrics (accuracy, confusion matrix and F1 scores).

    Args:
        prediction: list of predicted labels
        target: list of target labels
        ignored_labels (optional): list of labels to ignore, e.g. 0 for undef
        n_classes (optional): number of classes, max(target) by default
    Returns:
        accuracy, F1 score by class, confusion matrix
    """
    ignored_mask = np.zeros(target.shape[:2], dtype=bool)
    for l in ignored_labels:
        ignored_mask[target == l] = True
    ignored_mask = ~ignored_mask
    target = target[ignored_mask]
    prediction = prediction[ignored_mask]

    results = {}

    n_classes = np.max(target) + 1 if n_classes is None else n_classes

    '''混淆矩阵每一行代表该类别的预测结果情况'''
    cm = confusion_matrix(
        target,
        prediction,
        labels=range(n_classes))

    results["Confusion matrix"] = cm

    # Compute global accuracy
    total = np.sum(cm)
    accuracy = sum([cm[x][x] for x in range(len(cm))])
    accuracy *= 100 / float(total)

    results["Accuracy"] = accuracy

    # Compute F1 score
    F1scores = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            F1 = 2. * cm[i, i] / (np.sum(cm[i, :]) + np.sum(cm[:, i]))
        except ZeroDivisionError:
            F1 = 0.
        F1scores[i] = F1

    results["F1 scores"] = F1scores

    # Compute precision for every class
    Precisions = np.zeros(len(cm))
    for i in range(len(cm)):
        try:
            Precision = 1. * cm[i, i] / np.sum(cm[i, :])
        except ZeroDivisionError:
            Precision = 0.
        Precisions[i] = Precision

    results["Precisions"] = Precisions

    # Compute Average Accuracy (AA)
    AAs = []
    for i in range(len(cm)):
        try:
            recall = cm[i][i] / np.sum(cm[i, :])
            if np.isnan(recall):
                continue
        except ZeroDivisionError:
            recall = 0.
        AAs.append(recall)
    results['AA'] = np.mean(AAs)

    # Compute kappa coefficient
    pa = np.trace(cm) / float(total)
    pe = np.sum(np.sum(cm, axis=0) * np.sum(cm, axis=1)) / \
        float(total * total)
    kappa = (pa - pe) / (1 - pe)
    results["Kappa"] = kappa

    return results


# 在visdom上展示 F1 scores 和 混淆矩阵
def show_results(run, results, vis, label_values=None, agregated=False):
    text = ""

    if agregated:
        accuracies = [r["Accuracy"] for r in results]
        AAs = [r["AA"] for r in results]
        kappas = [r["Kappa"] for r in results]
        F1_scores = [r["F1 scores"] for r in results]
        Precisions = [r["Precisions"] for r in results]

        F1_scores_mean = np.mean(F1_scores, axis=0)
        F1_scores_std = np.std(F1_scores, axis=0)
        Precisions_mean = np.mean(Precisions, axis=0)
        Precisions_std = np.std(Precisions, axis=0)
        cm = np.mean([r["Confusion matrix"] for r in results], axis=0)
        text += "Agregated results :\n"
    else:
        cm = results["Confusion matrix"]
        accuracy = results["Accuracy"]
        F1scores = results["F1 scores"]
        Precision = results['Precisions']
        AA = results['AA']
        kappa = results["Kappa"]

    vis.heatmap(cm, opts={'title': "Confusion matrix run：{}".format(run),
                          'colorscale': "inferno",
                          'marginbottom': 150,
                          'marginleft': 150,
                          'width': 500,
                          'height': 500,
                          'rownames': label_values, 'columnnames': label_values})
    text += "Confusion matrix (run:{}):\n".format(run)
    text += str(cm)
    text += "---\n"

    if agregated:
        text += ("Accuracy: {:.04f} +- {:.04f}\n".format(np.mean(accuracies),
                                                         np.std(accuracies)))
        # for label, score, std in zip(label_values, accuracies_mean,
        #                              accuracies_std):
        #     text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        text += "Accuracy : {:.04f}%\n".format(accuracy)
        # for label, score in zip(label_values, accuracy):
        #     text += "\t{}: {:.03f}\n".format(label, score)
    text += "---\n"

    text += "F1 scores :\n"
    if agregated:
        for label, score, std in zip(label_values, F1_scores_mean,
                                     F1_scores_std):
            text += "\t{}: {:.04f} +- {:.04f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, F1scores):
            text += "\t{}: {:.04f}\n".format(label, score)
    text += "---\n"

    text += "Precisions :\n"
    if agregated:
        for label, score, std in zip(label_values, Precisions_mean,
                                     Precisions_std):
            text += "\t{}: {:.04f} +- {:.04f}\n".format(label, score, std)
    else:
        for label, score in zip(label_values, Precision):
            text += "\t{}: {:.04f}\n".format(label, score)
    text += "---\n"

    if agregated:
        text += ("AA: {:.04f} +- {:.04f}\n".format(np.mean(AAs),
                                                         np.std(AAs)))
        # for label, score, std in zip(label_values, accuracies_mean,
        #                              accuracies_std):
        #     text += "\t{}: {:.03f} +- {:.03f}\n".format(label, score, std)
    else:
        text += "AA : {:.04f}\n".format(AA)

    if agregated:
        text += ("Kappa: {:.04f} +- {:.04f}\n".format(np.mean(kappas),
                                                      np.std(kappas)))
    else:
        text += "Kappa: {:.04f}\n".format(kappa)



    vis.text(text.replace('\n', '<br/>'))
    print(text)

def samplingFixedNum(sample_num, groundTruth, seed):              #divide dataset into train and test datasets
    labels_loc = {}
    train_ = {}
    test_ = {}
    np.random.seed(seed)
    m = max(groundTruth)
    for i in range(m):
        indices = [j for j, x in enumerate(groundTruth.ravel().tolist()) if x == i + 1]
        np.random.shuffle(indices)
        labels_loc[i] = indices
        train_[i] = indices[:sample_num]
        test_[i] = indices[sample_num:]                     #difference derivation
    train_fix_indices = []
    test_fix_indices = []
    for i in range(m):
        train_fix_indices += train_[i]
        test_fix_indices += test_[i]
    np.random.shuffle(train_fix_indices)
    np.random.shuffle(test_fix_indices)
    return train_fix_indices, test_fix_indices

def sample_gt(gt, train_size, mode='random', seed=0):
    """Extract a fixed percentage of samples from an array of labels.

    Args:
        gt: a 2D array of int labels
        percentage: [0, 1] float
    Returns:
        train_gt, test_gt: 2D arrays of int labels

    """
    indices = np.nonzero(gt)
    X = list(zip(*indices)) # x,y features
    y = gt[indices].ravel() # classes
    train_gt = np.zeros_like(gt)
    test_gt = np.zeros_like(gt)
    if train_size > 1:
       train_size = int(train_size)
    if mode == 'random':
       train_indices, test_indices = sklearn.model_selection.train_test_split(X, train_size=train_size, stratify=y)
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       # train_gt[train_indices] = gt[train_indices]
       # test_gt[test_indices] = gt[test_indices]
       train_gt[train_indices[0], train_indices[1]] = gt[train_indices[0], train_indices[1]]
       test_gt[test_indices[0], test_indices[1]] = gt[test_indices[0], test_indices[1]]
    elif mode == 'fixed':
       print("Sampling {} with train size = {}".format(mode, train_size))
       train_indices, test_indices = [], []
       for c in np.unique(gt):
           if c == 0:
              continue
           indices = np.nonzero(gt == c)
           X = list(zip(*indices)) # x,y features

           train, test = sklearn.model_selection.train_test_split(X, train_size=train_size)
           train_indices += train
           test_indices += test
       train_indices = [list(t) for t in zip(*train_indices)]
       test_indices = [list(t) for t in zip(*test_indices)]
       train_gt[train_indices] = gt[train_indices]
       test_gt[test_indices] = gt[test_indices]
    elif mode == 'disjoint':
        train_gt = np.copy(gt)
        test_gt = np.copy(gt)
        for c in np.unique(gt):
            mask = gt == c
            for x in range(gt.shape[0]):
                first_half_count = np.count_nonzero(mask[:x, :])
                second_half_count = np.count_nonzero(mask[x:, :])
                try:
                    ratio = first_half_count / (first_half_count + second_half_count)
                    if ratio > 0.9 * train_size:
                        break
                except ZeroDivisionError:
                    continue
            mask[:x, :] = 0
            train_gt[mask] = 0

        test_gt[train_gt > 0] = 0
    elif mode == 'random_fixednumber':
        print("Sampling {} with fixed train size in each class = {}".format(mode, train_size))
        gt_reshape = gt.reshape(np.prod(gt.shape[:2]), ).astype(np.int)
        train_indices, test_indices = samplingFixedNum(train_size, gt_reshape, seed)
        train_gt = np.zeros(np.prod(gt.shape[:2]), )
        train_gt[train_indices] = gt_reshape[train_indices]
        test_gt = np.zeros(np.prod(gt.shape[:2]), )
        test_gt[test_indices] = gt_reshape[test_indices]
        train_gt = train_gt.reshape(int(np.prod(gt.shape[:1])), int(np.prod(gt.shape[1:])))
        test_gt = test_gt.reshape(int(np.prod(gt.shape[:1])), int(np.prod(gt.shape[1:])))
    else:
        raise ValueError("{} sampling is not implemented yet.".format(mode))
    return train_gt, test_gt


def compute_imf_weights(ground_truth, n_classes=None, ignored_classes=[]):
    """ Compute inverse median frequency weights for class balancing.

    For each class i, it computes its frequency f_i, i.e the ratio between
    the number of pixels from class i and the total number of pixels.

    Then, it computes the median m of all frequencies. For each class the
    associated weight is m/f_i.

    Args:
        ground_truth: the annotations array
        n_classes: number of classes (optional, defaults to max(ground_truth))
        ignored_classes: id of classes to ignore (optional)
    Returns:
        numpy array with the IMF coefficients
    """
    n_classes = np.max(ground_truth) if n_classes is None else n_classes
    weights = np.zeros(n_classes)
    frequencies = np.zeros(n_classes)

    for c in range(0, n_classes):
        if c in ignored_classes:
            continue
        frequencies[c] = np.count_nonzero(ground_truth == c)

    # Normalize the pixel counts to obtain frequencies
    frequencies /= np.sum(frequencies)
    # Obtain the median on non-zero frequencies
    idx = np.nonzero(frequencies)
    median = np.median(frequencies[idx])
    weights[idx] = median / frequencies[idx]
    weights[frequencies == 0] = 0.
    return weights

def camel_to_snake(name):
    s = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s).lower()

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

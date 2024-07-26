import matplotlib.pyplot as plt
import numpy as np
import mindspore
import os
import imageio
import cv2
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 20000000000

# ===========================================================================================================
def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = decode_segmap(label_masks, dataset)
    rgb_masks = mindspore.Tensor(np.array(rgb_masks))
    return rgb_masks


def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal' or dataset == 'coco':
        n_classes = 21
        label_colours = get_pascal_labels()

    # 添加你要分割的类别数n_classes
    elif dataset == 'SegVOC5':
        n_classes = 5
        label_colours = get_SegVOC5_labels()

    elif dataset == 'SegVOC7':
        n_classes = 7
        label_colours = get_SegVOC7_labels()

    elif dataset == 'SegVOC9':
        n_classes = 9
        label_colours = get_SegVOC9_labels()

    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb



def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """

    return np.asarray([[0, 0, 0],
                       [128, 0, 0],
                       [0, 128, 0],
                       [128, 128, 0],
                       [0, 0, 128],
                       [128, 0, 128],
                       [0, 128, 128],
                       [128, 128, 128],
                       [64, 0, 0],
                       [192, 0, 0],
                       [64, 128, 0],
                       [192, 128, 0],
                       [64, 0, 128],
                       [192, 0, 128],
                       [64, 128, 128],
                       [192, 128, 128],
                       [0, 64, 0],
                       [128, 64, 0],
                       [0, 192, 0],
                       [128, 192, 0],
                       [0, 64, 128]])


def get_SegVOC5_labels():  # 5 class
    return np.asarray([
        [0, 0, 255],      # label 0 的色彩      # blue ---water
        [0, 255, 0],      # label 1 的色彩      # green --- trees
        [255, 0, 0],      # label 2 的色彩      # red --- buildings
        [255, 255, 0],    # label 3 的色彩      # yellow --- framland
        [0, 0, 0],        # label 4 的色彩      # black ---Unknown
         ])

def get_SegVOC7_labels():  # 7 class
    return np.asarray([
        [0, 0, 255],      # label 0 的色彩      # blue ---water
        [0, 255, 0],      # label 1 的色彩      # green --- trees
        [255, 255, 0],    # label 2 的色彩      # yellow --- high_buildings
        [255, 0, 0],      # label 3 的色彩      # red --- low_buildings
        [210, 180, 140],  # label 4 的色彩      # brown --- road/bridge
        [255, 0, 255],    # label 5 的色彩      # rosein ---Bareland/grassland
        [255, 255, 255],  # label 6 的色彩      # white ---Unknown
         ])

def get_SegVOC9_labels():  # 9 class
    return np.asarray([
        [0, 0, 255],      # label 0 的色彩      # blue ---water
        [0, 255, 0],      # label 1 的色彩      # green --- trees
        [255, 255, 0],    # label 2 的色彩      # yellow --- high_buildings
        [255, 0, 255],    # label 3 的色彩      # rosein --- low_buildings
        [130, 80, 20],    # label 4 的色彩      # dark brown --- road
        [210, 180, 140],  # label 5 的色彩      # brown --- bare_land/small_grassland
        [0, 255, 255],    # label 6 的色彩      # lightcyan --- large_grassland
        [255, 0, 0],      # label 7 的色彩       # red --- airport runway
        [0, 0, 0],        # label 8 的色彩      # white ---Unknown
         ])

# ===========================================================================================================
# 评价指标

def cal_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(np.int32) + b[k], minlength=n ** 2).reshape(n, n)

def OverallAccuracy(confusionMatrix):
    #  返回所有类的整体像素精度OA
    # acc = (TP + TN) / (TP + TN + FP + TN)
    OA = np.diag(confusionMatrix).sum() / confusionMatrix.sum()
    return OA

def Kappa(confusion_matrix):
    pe_rows = np.sum(confusion_matrix, axis=0)
    pe_cols = np.sum(confusion_matrix, axis=1)
    sum_total = sum(pe_cols)
    pe = np.dot(pe_rows, pe_cols) / float(sum_total ** 2)
    po = np.trace(confusion_matrix) / float(sum_total)
    return (po - pe) / (1 - pe)

def Mean_Intersection_over_Union(confusion_matrix):
    IoU = np.diag(confusion_matrix) / (
                np.sum(confusion_matrix, axis=1) +
                np.sum(confusion_matrix, axis=0) -
                np.diag(confusion_matrix))
    IoU = IoU[:-1]

    MIoU = np.nanmean(IoU)
    return IoU, MIoU

def Precision(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    precision = precision[:-1]
    return precision

def Recall(confusionMatrix):
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    recall = recall[:-1]
    return recall

def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    f1score = 2 * precision * recall / (precision + recall)
    f1score = f1score[:-1]
    return f1score

def Frequency_Weighted_Intersection_over_Union(confusionMatrix):
    freq = np.sum(confusionMatrix, axis=1) / np.sum(confusionMatrix)
    iu = np.diag(confusionMatrix) / (
            np.sum(confusionMatrix, axis=1) +
            np.sum(confusionMatrix, axis=0) -
            np.diag(confusionMatrix))
    FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
    return FWIoU

def Record_result_evaluation(args, hist, target_names, precision, recall, f1ccore, OA, kappa, MIoU, FWIoU):
    with open(args.out_path + '/mindspore_progress_record.txt', 'a') as f:
        f.write('# ========================================================================================== \n')
        f.write(str(args) + '\n')
        f.write('Confusion matrix:\n')
        f.write(str(hist) + '\n')
        f.write('target_names:    \n' + str(target_names) + '\n')
        f.write('precision:       \n' + str(precision) + '\n')
        f.write('recall:          \n' + str(recall)    + '\n' )
        f.write('f1ccore:         \n' + str(f1ccore)   + '\n')
        f.write("OA:           " + str(OA) + '\n')
        f.write("kappa:        " + str(kappa) + '\n')
        f.write("MIoU:         " + str(MIoU) + '\n')
        f.write("FWIoU:        " + str(FWIoU) + '\n')

# ============================================================================================================================
def read_img_information(index):
    # 读取SAR图和GT原图
    if index == 1:
        gt_shape = (6187, 4278, 3)
        img_name = '_1_Traunstein'
        target_names = ['water', 'trees', 'buildings', 'framland', 'unknown']
        aim = 4

    elif index == 2:
        gt_shape = (16000, 18332, 3)
        img_name = '_2_Napoli'
        target_names = ['water', 'trees', 'buildings', 'framland', 'unknown']
        aim = 4

    elif index == 3:
        gt_shape = (16716, 18308, 3)
        img_name = '_3_PoDelta'
        target_names = ['water', 'trees', 'buildings', 'framland', 'unknown']
        aim = 4
    elif index == 4:
        gt_shape = (17296, 26606, 3)
        img_name = '_4_Istanbul'
        target_names = ['water', 'tree', 'low_buildings', 'road/bridges', 'unknown']
        aim = 4

    elif index == 5:
        gt_shape = (7224, 7691, 3)
        img_name = '_5_Rosenheim'
        target_names = ['water', 'trees', 'low_buildings', 'framland', 'unknown']
        aim = 4

    elif index == 6:
        gt_shape = (13836, 25454, 3)
        img_name = '_6_Washington'
        target_names = ['water', 'trees', 'hign_buildings', 'low_buildings', 'road/bridge', 'bare land grassland','unknown']
        aim = 6

    elif index == 7:
        gt_shape = (9281, 16309, 3)
        img_name = '_7_HongKongAirport'
        target_names = ['water', 'tree', 'high_buildings', 'low_buildings', 'road', 'bare_land/small_grassland', 'large_grassland', 'airport runway', 'unknow/parking lot']
        aim = 8

    return gt_shape, img_name, target_names, aim



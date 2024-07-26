# import matplotlib.pyplot as plt
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
    elif dataset == 'SegVOC5_1' or dataset == 'SegVOC5_2' or dataset == 'SegVOC5_3':
        n_classes = 5
        label_colours = get_SegVOC5_123_labels()

    elif dataset == 'SegVOC5_4':
        n_classes = 5
        label_colours = get_SegVOC5_4_labels()
    elif dataset == 'SegVOC5_5':
        n_classes = 5
        label_colours = get_SegVOC5_5_labels()

    elif dataset == 'SegVOC7_6':
        n_classes = 7
        label_colours = get_SegVOC7_6_labels()

    elif dataset == 'SegVOC9_7':
        n_classes = 9
        label_colours = get_SegVOC9_7_labels()
    elif dataset == 'SegVOC9_7':
        n_classes = 9
        label_colours = get_SegVOC9_7_labels()
    elif dataset == 'sar8':
        n_classes = 5
        label_colours = get_SegVOC5_123_labels()
    elif dataset == 'sar9':
        n_classes = 5
        label_colours = get_SegVOC5_123_labels()
    elif dataset == 'gf3sar1' or dataset == 'gf3sar2' or dataset == 'gf3sar3':
        n_classes = 6
        label_colours = get_gf3sar_labels()

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
    # if plot:
    #     plt.imshow(rgb)
    #     plt.show()
    # else:
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

def get_SegVOC5_123_labels():  # 5 class
    return np.asarray([
        [0, 0, 255],      # label 0 的色彩      # blue ---water
        [0, 255, 0],      # label 1 的色彩      # green --- trees
        [255, 0, 0],      # label 2 的色彩      # red --- buildings
        [255, 255, 0],    # label 3 的色彩      # yellow --- framland
        [0, 0, 0],        # label 4 的色彩      # black ---Unknown
         ])

def get_SegVOC5_4_labels():  # 5 class
    return np.asarray([
        [0, 0, 255],      # label 0 的色彩      # blue ---water
        [0, 255, 0],      # label 1 的色彩      # green --- trees
        [255, 0, 0],      # label 2 的色彩      # red --- buildings
        [210,180,140],    # label 3 的色彩      # yellow --- framland
        [0, 0, 0],        # label 4 的色彩      # black ---Unknown
         ])

def get_SegVOC5_5_labels():  # 5 class
    return np.asarray([
        [0, 0, 255],      # label 0 的色彩      # blue ---water
        [0, 255, 0],      # label 1 的色彩      # green --- trees
        [255, 0, 0],      # label 2 的色彩      # red --- buildings
        [150, 6, 205],    # label 3 的色彩      # yellow --- framland
        [0, 0, 0],        # label 4 的色彩      # black ---Unknown
         ])

def get_SegVOC7_6_labels():  # 7 class
    return np.asarray([
        [0, 0, 255],      # label 0 的色彩      # blue ---water
        [0, 255, 0],      # label 1 的色彩      # green --- trees
        [255, 255, 0],    # label 2 的色彩      # yellow --- high_buildings
        [255, 0, 0],      # label 3 的色彩      # red --- low_buildings
        [210, 180, 140],  # label 4 的色彩      # brown --- road/bridge
        [255, 0, 255],    # label 5 的色彩      # rosein ---Bareland/grassland
        [255, 255, 255],  # label 6 的色彩      # white ---Unknown
         ])

def get_SegVOC9_7_labels():  # 9 class
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
def get_gf3sar_labels():
    return np.asarray([
        [0, 0, 0],  # label 0 的色彩      # black ---无效类别
        [255, 0, 0],  # label 1 的色彩      # red ---建筑
        [0, 0, 255],  # label 2 的色彩      # blue ---水体
        [255, 255, 0],  # label 3 的色彩      # yellow --- 耕地
        [0, 255, 0],  # label 4 的色彩      # green --- 绿化
        [210, 180, 140],  # label 5 的色彩      # brown --- 路
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
    print('iou',IoU)
    IoU = IoU[1:]
    print('iou',IoU)
    MIoU = np.nanmean(IoU)
    return IoU, MIoU

def Precision(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    # precision = precision[:-1]
    precision = precision[1:]
    return precision

def Recall(confusionMatrix):
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    recall = recall[1:]
    return recall

def F1Score(confusionMatrix):
    precision = np.diag(confusionMatrix) / confusionMatrix.sum(axis=0)
    recall = np.diag(confusionMatrix) / confusionMatrix.sum(axis=1)
    f1score = 2 * precision * recall / (precision + recall)
    f1score = f1score[1:]
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
    with open(args.out_path + '/Record_test_parameters_and_pred_result.txt', 'a') as f:
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


def Record_train_parameters_set(args):
    with open(args.train_dir + '/Record_train_parameters_set.txt', 'a') as f:
        f.write('# ========================================================================================== \n')
        f.write('The train parameters seting is show:\n')
        f.write('# Ascend or CPU \n')
        f.write('# dataset \n')

        f.write('train_dir:          ' + str(args.train_dir) + '\n')
        f.write('train_data_file:    ' +str(args.train_data_file) + '\n')
        f.write('val_data_file:      ' +str(args.val_data_file)  + '\n' )
        f.write('eval_per_epoch:     ' + str(args.eval_per_epoch)   + '\n')
        # f.write('save_per_epoch:     ' + str(args.save_per_epoch) + '\n\n')

        f.write('batch_size:         ' + str(args.batch_size) + '\n')
        f.write('crop_size:          ' + str(args.crop_size) + '\n')
        f.write('image_mean:         ' + str(args.image_mean) + '\n')
        f.write('image_std:          ' + str(args.image_std)    + '\n' )
        f.write('min_scale:          ' + str(args.min_scale)   + '\n')
        f.write('max_scale:          ' + str(args.max_scale) + '\n')
        f.write('ignore_label:       ' + str(args.ignore_label) + '\n')
        f.write('num_classes:        ' + str(args.num_classes) + '\n\n')

        f.write('# optimizer \n')
        f.write('train_epochs:       ' + str(args.train_epochs)    + '\n' )
        f.write('lr_type:            ' + str(args.lr_type)   + '\n')
        f.write('base_lr:            ' + str(args.base_lr) + '\n')
        f.write('lr_decay_step:      ' + str(args.lr_decay_step) + '\n')
        f.write('lr_decay_rate:      ' + str(args.lr_decay_rate) + '\n')
        f.write('loss_scale:         ' + str(args.loss_scale)    + '\n' )
        f.write('weight_decay:       ' + str(args.weight_decay)   + '\n')
        # f.write('use_balanced_weights:       ' + str(args.use_balanced_weights) + '\n\n')

        f.write('# model \n')
        f.write('model:              ' + str(args.model) + '\n')
        f.write('freeze_bn:          ' + str(args.freeze_bn)    + '\n' )
        f.write('pretrainedmodel:    ' +str(args.data_url)+'/'+ str(args.pretrainedmodel_filename)  + '\n')

        f.write('device_target:      ' + str(args.device_target) + '\n')
        f.write('is_distributed:     ' + str(args.is_distributed) + '\n')
        f.write('rank:               ' + str(args.rank) + '\n')
        f.write('group_size:         ' + str(args.group_size)    + '\n' )
        f.write('save_steps:         ' + str(args.save_steps)   + '\n')
        f.write('keep_checkpoint_max:' + str(args.keep_checkpoint_max) + '\n')
        # f.write('amp_level:          ' + str(args.amp_level) + '\n\n')

        f.write('# ModelArts \n')
        f.write('modelArts_mode:     ' + str(args.modelArts_mode) + '\n')
        f.write('train_url:          ' + str(args.train_url) + '\n')
        f.write('data_url:           ' + str(args.data_url)  + '\n')
        f.write('train_data_filename:' + str(args.train_data_filename) + '\n')
        f.write('val_data_filename:   ' + str(args.val_data_filename) + '\n')
        f.write('pretrainedmodel_filename:' + str(args.pretrainedmodel_filename) + '\n\n')



def Record_test_parameters_set(args):
    with open(args.out_path + '/Record_test_parameters_and_pred_result.txt', 'a') as f:
        f.write('# ========================================================================================== \n')
        f.write('MindSpore DeepLabV3+ eval parameters seting is show :\n')
        f.write('# test data \n')
        f.write('data_root:          ' + str(args.data_root) + '\n')
        f.write('data_lst:           ' + str(args.data_lst) + '\n')
        f.write('batch_size:         ' + str(args.batch_size) + '\n')
        f.write('crop_size:          ' + str(args.crop_size) + '\n')
        f.write('image_mean:         ' + str(args.image_mean) + '\n')
        f.write('image_std:          ' + str(args.image_std)    + '\n\n' )

        f.write('scales:             ' + str(args.scales)   + '\n')
        f.write('flip:               ' + str(args.flip) + '\n')
        f.write('ignore_label:       ' + str(args.ignore_label) + '\n')
        f.write('num_classes:        ' + str(args.num_classes) + '\n\n')

        f.write('index:              ' + str(args.index)    + '\n' )
        f.write('out_path:           ' + str(args.train_url)   + '\n')
        f.write('dataset:            ' + str(args.dataset) + '\n')
        f.write('slidesize:          ' + str(args.slidesize) + '\n\n')

        f.write('# model \n')
        f.write('model:              ' + str(args.model) + '\n')
        f.write('freeze_bn:          ' + str(args.freeze_bn)    + '\n' )
        f.write('ckpt_path:          ' + str(args.ckpt_path)   + '\n\n')


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

    elif index == 8:
        single_size = (714, 714, 3)
        gt_shape = (3600, 3600, 3)
        img_name = '_8_RosenheimTwo'
        target_names = ['water', 'forest', 'buildings', 'framland', 'unknown']
        aim = 4
        img_num = 49

    elif index == 9:
        single_size = (733, 733, 3)
        gt_shape = (8000, 8000, 3)
        img_name = '_9_JiuJiang'
        target_names = ['water', 'forest', 'buildings', 'framland', 'unknown']
        aim = 4
        img_num = 225


    elif index == 'gf3sar1':
        single_size = (512, 512, 3)
        gt_shape = (9216, 10240, 3)
        img_name = '_gf3sar1_shandong'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree', 'load']
        aim = 0
        img_num =  360#928  # train:

    elif index == 'gf3sar2':
        single_size = (512, 512, 3)
        gt_shape = (7680, 9728, 3)
        img_name = '_gf3sar2_korea'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree', 'load']
        aim = 0
        img_num =  285#744  # train:

    elif index == 'gf3sar3':
        single_size = (748, 776, 3)
        gt_shape = (3840, 2304, 3)
        img_name = '_gf3sar3_xian'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree', 'load']
        aim = 0
        img_num = 28#84  # train:

    elif index == 'CD_demo_1_main':
        single_size = (700, 700, 3)
        gt_shape = (5000, 5000, 3)
        img_name = 'CD_demo_1'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree']
        aim = 0
        img_num = 100

    elif index == 'CD_demo_1_other':
        single_size = (700, 700, 3)
        gt_shape = (5000, 5000, 3)
        img_name = 'CD_demo_1_other'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree']
        aim = 0
        img_num = 100

    elif index == 'CD_demo_2_main':
        single_size = (700, 700, 3)
        gt_shape = (5000, 12000, 3)
        img_name = 'CD_demo_2_main'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree']
        aim = 0
        img_num = 240

    elif index == 'CD_demo_2_other':
        single_size = (700, 700, 3)
        gt_shape = (5000, 12000, 3)
        img_name = 'CD_demo_2_other'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree']
        aim = 0
        img_num = 240

    elif index == 'CD_demo_3_main':
        single_size = (586, 700, 3)
        gt_shape = (19300, 15000, 3)
        img_name = 'CD_demo_3_main'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree']
        aim = 0
        img_num = 1500

    elif index == 'CD_demo_3_other':
        single_size = (586, 700, 3)
        gt_shape = (19300, 15000, 3)
        img_name = 'CD_demo_3_other'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree']
        aim = 0
        img_num = 1500

    elif index == 'IR_demo_1_main':
        single_size = (712, 711, 3)
        gt_shape = (7692, 7666, 3)
        img_name = 'IR_demo_1_main'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree']
        aim = 0
        img_num = 225

    elif index == 'IR_demo_1_other':
        single_size = (712, 711, 3)
        gt_shape = (7692, 7666, 3)
        img_name = 'IR_demo_1_other'
        target_names = ['others', 'buildings', 'water', 'framland', 'tree']
        aim = 0
        img_num = 225

    return single_size, gt_shape, img_name, target_names, aim, img_num



# ============================================================================================================================
def read_img(index):
    # 读取SAR图和GT原图
    if index == 1:
        iput_img_original = cv2.imread('./Data/InitialImage/1-Traunstein/TraunsteinSAR8.png',flags=0)
        gt_original = cv2.imread('./Data/InitialImage/1-Traunstein/TraunsteinGt.png',flags=1)
        gt_original = cv2.cvtColor(gt_original, cv2.COLOR_BGR2RGB)
        img_name = '_1_Traunstein'
        target_names = ['water', 'trees', 'buildings', 'framland', 'unknown']
        aim = 4

    elif index == 2:
        iput_img_original = cv2.imread('./Data/InitialImage/2-CSK - Napoli/NapoliSAR8.png',flags=0)
        gt_original = cv2.imread('./Data/InitialImage/2-CSK - Napoli/NapoliGT.png',flags=1)
        gt_original = cv2.cvtColor(gt_original, cv2.COLOR_BGR2RGB)
        img_name = '_2_Napoli'
        target_names = ['water', 'trees', 'buildings', 'framland', 'unknown']
        aim = 4

    elif index == 3:
        iput_img_original = cv2.imread('./Data/InitialImage/3-CSK - PoDelta/PoDeltaSAR8.png',flags=0)
        gt_original = cv2.imread('./Data/InitialImage/3-CSK - PoDelta/PoDeltaGT.png',flags=1)
        gt_original = cv2.cvtColor(gt_original, cv2.COLOR_BGR2RGB)
        img_name = '_3_PoDelta'
        target_names = ['water', 'trees', 'buildings', 'framland', 'unknown']
        aim = 4
    elif index == 4:
        iput_img_original = Image.open(r'./Data/InitialImage/4-Istanbul/IstanbulSAR8_nor_95.png').convert('L')
        iput_img_original = np.array(iput_img_original)
        gt_original = Image.open('./Data/InitialImage/4-Istanbul/IstanbulGT.png')
        gt_original = np.array(gt_original)
        img_name = '_4_Istanbul'
        target_names = ['water', 'tree', 'low_buildings', 'road/bridges', 'unknown']
        aim = 4

    elif index == 5:
        iput_img_original = cv2.imread('./Data/InitialImage/5-Rosenheim/RosenheimSAR8_nor_95.png', flags=0)
        gt_original = cv2.imread('./Data/InitialImage/5-Rosenheim/RosenheimGT.png', flags=1)
        gt_original = cv2.cvtColor(gt_original, cv2.COLOR_BGR2RGB)
        img_name = '_5_Rosenheim'
        target_names = ['water', 'trees', 'low_buildings', 'framland', 'unknown']
        aim = 4

    elif index == 6:
        iput_img_original = cv2.imread('./Data/InitialImage/6-Washington/WashingtonSAR8_nor_95.png', flags=0)
        gt_original = cv2.imread('./Data/InitialImage/6-Washington/WashingtonGT.png', flags=1)
        gt_original = cv2.cvtColor(gt_original, cv2.COLOR_BGR2RGB)
        img_name = '_6_Washington'
        target_names = ['water', 'trees', 'hign_buildings', 'low_buildings', 'road/bridge', 'bare land grassland','unknown']
        aim = 6

    elif index == 7:
        iput_img_original = cv2.imread('./Data/InitialImage/7-HongkongAirport/HongKongAirportSAR8_nor_95.png', flags=0)
        gt_original = cv2.imread('./Data/InitialImage/7-HongkongAirport/HongKongAirportGT.png', flags=1)
        gt_original = cv2.cvtColor(gt_original, cv2.COLOR_BGR2RGB)
        img_name = '_7_HongKongAirport'
        target_names = ['water', 'tree', 'high_buildings', 'low_buildings', 'road', 'bare_land/small_grassland', 'large_grassland', 'airport runway', 'unknow/parking lot']
        aim = 8


    return iput_img_original, gt_original, img_name, target_names, aim

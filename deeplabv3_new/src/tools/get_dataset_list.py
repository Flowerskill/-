import argparse
import os
import numpy as np
from PIL import Image
import scipy.io

parser = argparse.ArgumentParser('dataset list generator')
parser.add_argument("--data_dir", type=str, default='', help='where dataset stored.')

args, _ = parser.parse_known_args()
data_dir = args.data_dir
print("Data dir is:", data_dir)

# # VOC2012
# VOC_IMG_DIR = os.path.join(data_dir, 'VOCdevkit/VOC2012/JPEGImages')
# VOC_ANNO_DIR = os.path.join(data_dir, 'VOCdevkit/VOC2012/SegmentationClass')
# VOC_ANNO_GRAY_DIR = os.path.join(data_dir, 'VOCdevkit/VOC2012/SegmentationClassGray')          # new create
# VOC_TRAIN_TXT = os.path.join(data_dir, 'VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt')
# VOC_VAL_TXT = os.path.join(data_dir, 'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt')


# SARVOC7
VOC_IMG_DIR = os.path.join(data_dir, 'JPEGImages')
VOC_ANNO_DIR = os.path.join(data_dir, 'SegmentationClass')
VOC_ANNO_GRAY_DIR = os.path.join(data_dir, 'SegmentationClassGray')          # new create
VOC_TRAIN_TXT = os.path.join(data_dir, 'ImageSets/Segmentation/train8.txt')
VOC_VAL_TXT = os.path.join(data_dir, 'ImageSets/Segmentation/val8.txt')
VOC_TEST_TXT = os.path.join(data_dir, 'ImageSets/Segmentation/test8.txt')


VOC_TRAIN_LST_TXT = os.path.join(data_dir, 'MindRecord/train_sar8_lst.txt')
VOC_VAL_LST_TXT = os.path.join(data_dir, 'MindRecord/val_sar8_lst.txt')
VOC_TEST_LST_TXT = os.path.join(data_dir, 'MindRecord/test_sar8_lst.txt')

path_txt = os.path.join(data_dir, "MindRecord")
if not os.path.exists(path_txt):
    os.makedirs(path_txt)


def __get_data_list(data_list_file):
    with open(data_list_file, mode='r') as f:
        return f.readlines()


def conv_voc_colorpng_to_graypng():
    if not os.path.exists(VOC_ANNO_GRAY_DIR):
        os.makedirs(VOC_ANNO_GRAY_DIR)

    for ann in os.listdir(VOC_ANNO_DIR):
        ann_im = Image.open(os.path.join(VOC_ANNO_DIR, ann))
        ann_im = Image.fromarray(np.array(ann_im))
        ann_im.save(os.path.join(VOC_ANNO_GRAY_DIR, ann))


# 训练集
def create_voc_train_lst_txt():
    voc_train_data_lst = __get_data_list(VOC_TRAIN_TXT)
    with open(VOC_TRAIN_LST_TXT, mode='w') as f:
        for id_ in voc_train_data_lst:
            id_ = id_.strip()
            img_ = os.path.join(VOC_IMG_DIR, id_ + '.jpg')
            anno_ = os.path.join(VOC_ANNO_GRAY_DIR, id_ + '.png')
            f.write(img_ + ' ' + anno_ + '\n')

# 验证集
def create_voc_val_lst_txt():
    voc_val_data_lst = __get_data_list(VOC_VAL_TXT)
    with open(VOC_VAL_LST_TXT, mode='w') as f:
        for id_ in voc_val_data_lst:
            id_ = id_.strip()
            img_ = os.path.join(VOC_IMG_DIR, id_ + '.jpg')
            anno_ = os.path.join(VOC_ANNO_GRAY_DIR, id_ + '.png')
            f.write(img_ + ' ' + anno_ + '\n')

# 测试集
def create_voc_test_lst_txt():
    voc_test_data_lst = __get_data_list(VOC_TEST_TXT)
    with open(VOC_TEST_LST_TXT, mode='w') as f:
        for id_ in voc_test_data_lst:
            id_ = id_.strip()
            img_ = os.path.join(VOC_IMG_DIR, id_ + '.jpg').replace('../../', './')
            anno_ = os.path.join(VOC_ANNO_GRAY_DIR, id_ + '.png').replace('../../', './')
            f.write(img_ + ' ' + anno_ + '\n')



if __name__ == '__main__':
    print('converting VOCdevkit color png to gray png ...')
    conv_voc_colorpng_to_graypng()
    print('converting done.')

    create_voc_train_lst_txt()
    print('generating VOCdevkit train list success.')

    create_voc_val_lst_txt()
    print('generating VOCdevkit val list success.')

    create_voc_test_lst_txt()
    print('generating VOCdevkit test list success.')


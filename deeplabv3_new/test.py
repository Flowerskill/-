import os
import argparse
from cv2 import UMAT_MAGIC_VAL
import numpy as np
import cv2
import ast
import os.path as osp

from mindspore import Tensor
import mindspore.common.dtype as mstype
import mindspore.nn as nn
from mindspore import context, Tensor, ops
from mindspore.communication.management import init, get_rank, get_group_size, get_local_rank, create_group, \
    get_group_rank_from_world_rank
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src.deeplab_v3plus import DeepLabV3Plus
from src import dataset as data_generator
# from src.tool import OverallAccuracy, Precision, Recall, F1Score, Frequency_Weighted_Intersection_over_Union
from src.tool import cal_hist, OverallAccuracy, Kappa, Mean_Intersection_over_Union, Precision, Recall, F1Score, \
    Frequency_Weighted_Intersection_over_Union, read_img_information, Record_result_evaluation
from src.tool import Record_test_parameters_set
from src.dataset import pre_process
import time
import imageio
from PIL import Image
import moxing as mox
from mindspore.profiler.profiling import Profiler


# from sklearn import metrics


# progress post handle
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)

def parse_args():
    parser = argparse.ArgumentParser('MindSpore DeepLabV3+ eval')
    parser.add_argument('--out_path', type=str, default='', help='mask image to save')
    # test data
    parser.add_argument('--data_root', type=str, default='.', help='root path of val data')
    parser.add_argument('--data_lst', type=str, default='.', help='list of val data')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size,default=16')
    parser.add_argument('--crop_size', type=int, default=512, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[54.303, 76.681, 67.047], help='image mean')
    parser.add_argument('--image_std', type=list, default=[32.368,  32.384, 38.725],
                        help='image std')  # xianyang[32.368,  32.384, 38.725]tongchuan_4[28.153, 32.254, 33.600]xian[30.222, 29.708, 36.227]yanta[33.714, 32.438, 37.919],yanan2[21.595, 29.402, 31.737]
    parser.add_argument('--min_scale', type=float, default=0.5, help='minimum scale of data argumentation')
    parser.add_argument('--max_scale', type=float, default=2.0, help='maximum scale of data argumentation')
    parser.add_argument('--scales', type=float, default=[1.0], action='append', help='scales of evaluation ')
    parser.add_argument('--flip', action='store_true', help='perform left-right flip')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=10, help='number of classes')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--index', type=str, default='baoji_3', help='which image to test')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'CPU', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')  # ModelArts
    parser.add_argument('--train_url', type=str, default='',
                        help='where result saved')
    parser.add_argument('--dataset', type=str, default='baoji_3', help='dataset name (default: pascal)')
    # post progress related set
    parser.add_argument('--slidesize', type=int, default=0, help=' sample bound')
    # model
    parser.add_argument("--data_dir_target", type=str, default='21zhang_test.mindrecord',
                        help="Path to the directory containing the source dataset.")
    parser.add_argument('--model', type=str, default='DeepLabV3plus_s16', help='select model')
    parser.add_argument('--freeze_bn', action='store_true', default=True, help='freeze bn')
    # 'obs://deeplabv3plus/lsj/building/xian_9/out/3/'
    parser.add_argument('--restore_from_file', type=str, default='obs://deeplabv3plus/luoxingyu/21zhang_all_train/train_output_lr_0.1_epoch_300/',
                        help='model to evaluate')
    parser.add_argument('--restore_from_name', type=str, default='DeepLabV3plus_s16-300_35.ckpt',
                        help='model to evaluate')
    parser.add_argument('--modelArts_mode', type=ast.literal_eval, default=True,
                        help='train on modelarts or not, default is False')
    #'obs://sar/sar_data/sar-test/'
    parser.add_argument('--data_url', type=str, default='obs://deeplabv3plus/luoxingyu/21zhang_all_train/data/',
                        help='the directory path of saved file')
    parser.add_argument('--is_distributed', type=bool, default='', help='distributed training ')
    args, _ = parser.parse_known_args()
    return args


palette = [
        0, 0, 255,  # "water"
        225, 255, 0,  #  "farmland"
        0, 100, 0,  #  "vegetation_forest"
        0, 255, 0,  # "vegetation_meadow"
        192, 192, 192,  # "road"
        128, 0, 0,  # "village"
        255, 0, 0,  # "city"
        255, 255, 255,  #  "other"
        244, 164, 96,   # ground
         0,  0,  0,   # "background"
    ]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask


class BuildEvalNetwork(nn.Cell):
    def __init__(self, network):
        super(BuildEvalNetwork, self).__init__()
        self.network = network
        self.softmax = nn.Softmax(axis=1)

    def construct(self, input_data):
        output = self.network(input_data)
        output = self.softmax(output)
        return output


def eval_batch(args, eval_net, img_lst, single_size, flip=True):
    result_lst = []
    batch_size = img_lst.shape[0]
    batch_img = np.zeros((batch_size, 3, single_size[0], single_size[1]), dtype=np.float32)
    # for l in range(args.batch_size):
    #     img_ = pre_process(args, img_lst[l])
    #     batch_img[l] = img_
    # batch_img = np.ascontiguousarray(img_lst) # 将一个内存不连续存储的数组转换为内存连续存储的数组，使得运行速度更快。
    net_out = eval_net(img_lst)
    net_out = net_out.asnumpy()

    if flip:
        batch_img = batch_img[:, :, :, ::-1]
        net_out_flip = eval_net(Tensor(batch_img, mstype.float32))
        net_out += net_out_flip.asnumpy()[:, :, :, ::-1]

    for bs in range(batch_size):
        probs_ = net_out[bs].transpose((1, 2, 0))  # (722, 723, 5)
        result_lst.append(probs_)
        # if batch_size == 2:
        #     print('wwww')

    return result_lst


def eval_batch_scales(args, eval_net, img_lst, single_size, flip=True):
    probs_lst = eval_batch(args, eval_net, img_lst, single_size, flip=flip)
    result_msk = []
    for i in probs_lst:
        result_msk.append(i.argmax(axis=2))
    return result_msk


def net_eval():
    load_s_time = time.time()
    args = parse_args()

    if args.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
    else:
        context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True, save_graphs=False,
                            device_target="Ascend", device_id=int(os.getenv('DEVICE_ID')))

    if args.modelArts_mode:
        mox.file.set_auth(ak='3YUHYFFBN3WOUMN3PMUD', sk='w4eWQdkT22GPthPcTJulTrpSKxtkeJKeAXJGSzpe',
                          server='obs.cn-northwest-229.yantachaosuanzhongxin.com')
        local_record = os.path.join(args.train_url, 'results/')
        #local_record = '/cache/path/results/'
        args.out_path = local_record
        #local_data_url = '/cache/data'
        #local_train_url = '/cache/ckpt'
        # download dataset from obs to cache
        #mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
        from pathlib import Path
        local_plog_path = os.path.join(Path.home(), 'ascend/log/')
        # mox.file.copy_parallel(src_url=args.ckpt_path, dst_url=local_train_url)
        device_num = int(os.getenv('RANK_SIZE'))
        if device_num > 1:
            init()
            # group = str('0-16')
            # rank_ids = [0,16]
            # create_group(group, rank_ids)
            # group_rank_id = get_group_rank_from_world_rank(16,group)
            args.rank = get_rank()
            args.group_size = get_group_size()
            print("rank_id is {}, device_num is {}".format(args.rank, device_num))
            context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL,
                                              device_num=args.group_size, global_rank=args.rank)
            # local_data_url = os.path.join(local_data_url, str(device_id))
        # download dataset from obs to cache
        # args.ckpt_path = os.path.join(local_data_url, args.ckpt_path)
        #mox.file.copy_parallel(src_url=args.restore_from_file, dst_url=local_train_url)

        args.restore_from = os.path.join(args.train_url, args.restore_from_name)
        args.data_dir_target = os.path.join(args.data_url, args.data_dir_target)
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
    Record_test_parameters_set(args)
    # data list
    # with open(args.data_lst) as f:
    #     img_lst = f.readlines()

    # network
    if args.model == 'DeepLabV3plus_s16':
        network = DeepLabV3Plus('eval', args.num_classes, 16, args.freeze_bn)
    elif args.model == 'DeepLabV3plus_s8':
        network = DeepLabV3Plus('eval', args.num_classes, 8, args.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(args.model))

    eval_net = BuildEvalNetwork(network)

    # load model
    param_dict = load_checkpoint(args.restore_from)
    load_param_into_net(eval_net, param_dict)
    eval_net.set_train(False)
    # print(eval_net) # network structure

    target_dataset = data_generator.SegDataset(image_mean=args.image_mean,
                                               image_std=args.image_std,
                                               data_file=args.data_dir_target,
                                               batch_size=args.batch_size,
                                               crop_size=args.crop_size,
                                               max_scale=args.max_scale,
                                               min_scale=args.min_scale,
                                               ignore_label=args.ignore_label,
                                               num_classes=args.num_classes,
                                               num_readers=2,
                                               num_parallel_calls=4,
                                               shard_id=args.rank,
                                               shard_num=args.group_size)

    target_dataset = target_dataset.get_dataset(repeat=1)
    # evaluate
    hist = np.zeros((args.num_classes, args.num_classes))
    # batch_name_lst = []
    bi = 0
    # rownum = 12737 // 512
    # colnum = 14340 // 512
    image_num = 1158

    single_size = [512, 512, 3]
    # result_img = []
    # r = 0
    # c = 0
    total_time = 0
    total_hist_time = 0
    load_u_time = time.time()
    # eval
    split_num = int(image_num / args.group_size)
    data_rank_num = (image_num % args.group_size)
    print("rank_id is {},len(target_dataset) is {},group_size is {}".format(args.rank, split_num, args.group_size))
    for index, batch in enumerate(target_dataset):
        s_time = time.time()
        image, label = batch

        n = len(image)
        bi = n
        # if r == rownum - 1 and c == colnum - 1:
        #     break
        label_img_lst = []

        for i in range(n):
            # msk_ = label[i]
            # label_img_lst.append(msk_)
            try:
                label_img_lst.append(label[i])
                bi += 1
            except RuntimeError:
                print("RuntimeError")
                # mox.file.copy_parallel(src_url="/cache/path/", dst_url=args.train_url)
                # mox.file.copy_parallel(src_url=local_plog_path, dst_url=args.train_url)
                # mox.file.copy_parallel(src_url='/tmp/log/', dst_url=args.train_url)

        if bi > 0:
            # label_img_lst = label.squeeze(0)
            first_s = time.time()
            batch_res = eval_batch_scales(args, eval_net, image, single_size, flip=args.flip)
            first_e = time.time()
            print(first_e - first_s)
            for mi in range(n):
                # Remove boundary, value is class label, shape is hw
                label_img = label_img_lst[mi][args.slidesize: label_img_lst[mi].shape[0] - args.slidesize,
                            args.slidesize:label_img_lst[mi].shape[1] - args.slidesize]
                label_img = label_img.asnumpy()
                batch_res_image = batch_res[mi][args.slidesize: batch_res[mi].shape[0] - args.slidesize,
                                  args.slidesize:batch_res[mi].shape[1] - args.slidesize]
                if args.group_size <= 8:
                    img_index = index * args.batch_size + args.rank * split_num + mi
                else:
                    img_index = index * args.batch_size + args.rank * (split_num + 1) + mi
                # result_img[img_index, :, :] = batch_res_image
                # batch_res_image = Tensor(batch_res_image).asnumpy()
                result_all = colorize_mask(batch_res_image)
                # save the complete pred result image
                result_all.save('%s/%s_color.png' % (args.out_path, str(img_index)))

                label_img = label_img.flatten()
                batch_res_image = batch_res_image.flatten()

                hist += cal_hist(label_img, batch_res_image, args.num_classes)

            print('processed {} images'.format(img_index))
            bi = 0


    target_names = ['water', 'farmland', 'vegetation_forest', 'vegetation_meadow', 'road', 'village', 'city', 'other', 'land', 'background']
    OA = OverallAccuracy(hist)
    precision = Precision(hist)
    recall = Recall(hist)
    IoU, MIoU = Mean_Intersection_over_Union(hist)
    f1ccore = F1Score(hist)
    kappa = Kappa(hist)
    FWIoU = Frequency_Weighted_Intersection_over_Union(hist)
    Record_result_evaluation(args, hist, target_names, precision, recall, f1ccore, OA, kappa, MIoU, FWIoU)

    # 以下记录不包含黑色背景
    hist_withoutblack = hist[0:9, 0:9]
    target_names_withoutblack = ['water', 'farmland', 'vegetation_forest', 'vegetation_meadow', 'road', 'village', 'city', 'other', 'land']
    OA_withoutblack = OverallAccuracy(hist_withoutblack)
    precision_withoutblack = Precision(hist_withoutblack)
    recall_withoutblack = Recall(hist_withoutblack)
    IoU_withoutblack, MIoU_withoutblack = Mean_Intersection_over_Union(hist_withoutblack)
    f1score_withoutblack = F1Score(hist_withoutblack)
    kappa_withoutblack = Kappa(hist_withoutblack)
    FWIoU_withoutblack = Frequency_Weighted_Intersection_over_Union(hist_withoutblack)
    Record_result_evaluation(args, hist_withoutblack, target_names_withoutblack, precision_withoutblack, recall_withoutblack, f1score_withoutblack, OA_withoutblack, kappa_withoutblack, MIoU_withoutblack, FWIoU_withoutblack)

    all_end = time.time()
    # with open(args.out_path + '/Record_test_information.txt', 'a') as f:
    #     f.write('# ========================================================================================== \n')
    #     f.write('load_model_time:       ' + str(load_u_time - load_s_time) + '\n')
    #     f.write('total_img_test_time:   ' + str(total_time) + '\n')
    #     f.write('every_img_test_time:   ' + str(total_time / (image_num / int(args.group_size))) + '\n')
    #     # f.write('every_hist_cal_time:   ' + str(hist_u_time - hist_s_time) + '\n')
    #     f.write('total_hist_time:       ' + str(total_hist_time) + '\n')
    #     f.write('save_all_img_time:     ' + str(save_time) + '\n')
    #     f.write('cal_evaluation_time:   ' + str(cal_u_time - cal_s_time) + '\n')
    #     f.write('total_test_time:       ' + str(all_end - load_u_time) + '\n')

    # mox.file.copy_parallel(src_url=args.out_path, dst_url=args.train_url)
    # if args.rank == 0:
    #     mox.file.copy_parallel(src_url="/cache/path/", dst_url=args.train_url)
    #     mox.file.copy_parallel(src_url=local_plog_path, dst_url=args.train_url)
    #     mox.file.copy_parallel(src_url='/tmp/log/', dst_url=args.train_url)


if __name__ == '__main__':
    args = parse_args()
    profiler = Profiler(output_path=args.train_url)
    net_eval()
    profiler.analyse()
    # mox.file.copy_parallel(src_url="/cache/path/", dst_url=args.train_url)




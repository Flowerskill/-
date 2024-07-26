import os
import argparse
import ast
from mindspore import context
from mindspore.train.model import Model
from mindspore.context import ParallelMode
import mindspore.nn as nn
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from mindspore.communication.management import init, get_rank, get_group_size
from mindspore.train.callback import LossMonitor, TimeMonitor, SummaryCollector
from mindspore.train.loss_scale_manager import FixedLossScaleManager
from mindspore.common import set_seed
from src import dataset_train as data_generator
from src import loss, learning_rates, callback
from src.deeplab_v3plus import DeepLabV3Plus
from src.metrics import Dice, MIoU
from src.tool import Record_train_parameters_set
import time
set_seed(1)


# ubuntu18.04  GPU CUDA 10.1 MindSpore 1.3.0 python 3.7.5
# 基于VOC公开数据集的本地训练
# 针对SAR图像，禁用数据增强

# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class BuildTrainNetwork(nn.Cell):
    def __init__(self, network, criterion):
        super(BuildTrainNetwork, self).__init__()
        self.network = network
        self.criterion = criterion

    def construct(self, input_data, label):
        output = self.network(input_data)
        net_loss = self.criterion(output, label)
        return net_loss


def parse_args():
    parser = argparse.ArgumentParser('MindSpore DeepLabV3+ training')
    # Ascend or CPU
    parser.add_argument('--train_dir', type=str, default='', help='where training log and CKPTs saved(ckpt_save_dir)')

    # dataset
    parser.add_argument('--train_data_file', type=str, default='', help='path and Name of one MindRecord file')
    parser.add_argument('--val_data_file', type=str, default='', help='path and Name of one MindRecord file')
    parser.add_argument('--eval_per_epoch', type=int, default=50, help='The accuracy of the model is verified at a fixed position of epoch')

    parser.add_argument('--batch_size', type=int, default=2, help='batch size, default=32 ')
    parser.add_argument('--crop_size', type=int, default=512, help='crop size')
    parser.add_argument('--image_mean', type=list, default=[103.53, 116.28, 123.675],  help='image mean') # BGR
    parser.add_argument('--image_std', type=list, default=[57.375, 57.120, 58.395], help='image std')     # BGR
    parser.add_argument('--min_scale', type=float, default=0.5, help='minimum scale of data argumentation')
    parser.add_argument('--max_scale', type=float, default=2.0, help='maximum scale of data argumentation')
    parser.add_argument('--ignore_label', type=int, default=255, help='ignore label')
    parser.add_argument('--num_classes', type=int, default=2, help='number of classes')

    # optimizer
    parser.add_argument('--train_epochs', type=int, default=50, help='epoch, default=300')
    parser.add_argument('--lr_type', type=str, default='poly', choices=['poly', 'step', 'cos'], help='type of learning rate, default = cos')
    parser.add_argument('--base_lr', type=float, default=0.001, help='base learning rate cos:0.015 poly:0.007')
    parser.add_argument('--lr_decay_step', type=int, default=40000, help='learning rate decay step')
    parser.add_argument('--lr_decay_rate', type=float, default=0.15, help='learning rate decay rate')
    parser.add_argument('--loss_scale', type=float, default=3072.0, help='loss scale')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='loss scale  default = 0.0001 or 5e-4')

    # model
    parser.add_argument('--model', type=str, default='DeepLabV3plus_s16', help='select model')
    parser.add_argument('--freeze_bn', action='store_true', help='freeze bn')
    parser.add_argument('--ckpt_pre_trained', type=str, default='', help='PreTrained model')

    # train
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'CPU', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--is_distributed', action='store_true', help='distributed training')
    parser.add_argument('--rank', type=int, default=0, help='local rank of distributed')
    parser.add_argument('--group_size', type=int, default=1, help='world size of distributed')
    parser.add_argument('--save_steps', type=int, default=130, help='steps interval for saving')
    parser.add_argument('--keep_checkpoint_max', type=int, default=500, help='max checkpoint for saving default=200')

    # ModelArts
    parser.add_argument('--modelArts_mode', type=ast.literal_eval, default=True,
                        help='train on modelarts or not, default is False')
    parser.add_argument('--train_url', type=str, default='obs://sar-deeplab-test/sar-train5/', help='where training log and CKPTs saved')
    parser.add_argument('--data_url', type=str, default='obs://deeplabv3plus/sar_data_test/', help='the directory path of saved file')
    parser.add_argument('--train_data_filename', type=str, default='shate_330.mindrecord', help='Name of the MindRecord file')
    parser.add_argument('--val_data_filename', type=str, default='voc_val_sar1.mindrecord',help='Name of the MindRecord file')
    parser.add_argument('--pretrainedmodel_filename', type=str, default='new.ckpt', help='Name of the pretraining model file')

    args, _ = parser.parse_known_args()
    return args


def train():
    args = parse_args()
    time3 = time.time()
    if args.device_target == "CPU":
        context.set_context(mode=context.GRAPH_MODE, save_graphs=False, device_target="CPU")
    elif args.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", save_graphs=False)
    else:
        context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True, save_graphs=False,
                            device_target="Ascend", device_id=int(os.getenv('DEVICE_ID')))

    # init multicards training
    if args.modelArts_mode:
        import moxing as mox
        mox.file.set_auth(ak='3YUHYFFBN3WOUMN3PMUD', sk='w4eWQdkT22GPthPcTJulTrpSKxtkeJKeAXJGSzpe',
                          server='obs.cn-northwest-229.yantachaosuanzhongxin.com')
        local_data_url = '/cache/data'
        local_train_url = '/cache/ckpt'
        from pathlib import Path
        local_plog_path = os.path.join(Path.home(), 'ascend/log/')
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        if device_num > 1:
            init()
            args.rank = get_rank()
            args.group_size = get_group_size()
            parallel_mode = ParallelMode.DATA_PARALLEL
            context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True,
                                              device_num=args.group_size)
            local_data_url = os.path.join(local_data_url, str(device_id))
        # download dataset from obs to cache
        mox.file.copy_parallel(src_url=args.data_url, dst_url=local_data_url)
        train_data_file = local_data_url + '/' + args.train_data_filename
        val_data_file = local_data_url + '/' + args.val_data_filename
        ckpt_file = local_data_url + '/' + args.pretrainedmodel_filename
        train_dir = local_train_url
    else:
        if args.is_distributed:
            init()
            args.rank = get_rank()
            args.group_size = get_group_size()
            parallel_mode = ParallelMode.DATA_PARALLEL
            context.set_auto_parallel_context(parallel_mode=parallel_mode, gradients_mean=True,
                                              device_num=args.group_size)
        train_data_file = args.train_data_file
        ckpt_file = args.ckpt_pre_trained
        val_data_file = args.val_data_file
        train_dir = args.train_dir

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    Record_train_parameters_set(args)
    print(args)
    # train dataset
    train_dataset = data_generator.SegDataset(image_mean=args.image_mean,
                                        image_std=args.image_std,
                                        data_file= train_data_file,
                                        batch_size=args.batch_size,
                                        crop_size=args.crop_size,
                                        max_scale=args.max_scale,
                                        min_scale=args.min_scale,
                                        ignore_label=args.ignore_label,
                                        num_classes=args.num_classes,
                                        num_readers=64,
                                        num_parallel_calls=64,
                                        shard_id=args.rank,
                                        shard_num=args.group_size)
    train_dataset = train_dataset.get_dataset(repeat=1)  # data: dtype=float32  label: dtype=uint8
    for item in train_dataset.create_dict_iterator(output_numpy=True):
        print('train:', item['data'].dtype)


    # eval dataset
    val_dataset = data_generator.SegDataset(image_mean=args.image_mean,
                                        image_std=args.image_std,
                                        data_file= val_data_file,
                                        batch_size=1,
                                        crop_size=args.crop_size,
                                        max_scale=args.max_scale,
                                        min_scale=args.min_scale,
                                        ignore_label=args.ignore_label,
                                        num_classes=args.num_classes,
                                        num_readers=2,
                                        num_parallel_calls=4,
                                        shard_id=args.rank,
                                        shard_num=args.group_size)
    val_dataset = val_dataset.get_dataset(repeat=1)
    for item in val_dataset.create_dict_iterator(output_numpy=True):
        print('val:', item['data'].dtype)




    # network
    if args.model == 'DeepLabV3plus_s16':
        network = DeepLabV3Plus('train', args.num_classes, 16, args.freeze_bn)
    elif args.model == 'DeepLabV3plus_s8':
        network = DeepLabV3Plus('train', args.num_classes, 8, args.freeze_bn)
    else:
        raise NotImplementedError('model [{:s}] not recognized'.format(args.model))

    # loss
    loss_ = loss.SoftmaxCrossEntropyLoss(args.num_classes, args.ignore_label)
    loss_.add_flags_recursive(fp32=True)
    train_net = BuildTrainNetwork(network, loss_)

    # load pretrained model
    if args.ckpt_pre_trained or args.pretrainedmodel_filename:
        param_dict = load_checkpoint(ckpt_file)
        load_param_into_net(train_net, param_dict)

    # optimizer
    iters_per_epoch = train_dataset.get_dataset_size()
    print("iters_per_epoch = ", iters_per_epoch)

    total_train_steps = iters_per_epoch * args.train_epochs
    if args.lr_type == 'cos':
        lr_iter = learning_rates.cosine_lr(args.base_lr, total_train_steps, total_train_steps)
    elif args.lr_type == 'poly':
        lr_iter = learning_rates.poly_lr(args.base_lr, total_train_steps, total_train_steps, end_lr=0.0, power=0.9)
    elif args.lr_type == 'exp':
        lr_iter = learning_rates.exponential_lr(args.base_lr, args.lr_decay_step, args.lr_decay_rate,
                                                total_train_steps, staircase=True)
    else:
        raise ValueError('unknown learning rate type')

    # 模式一 Drop the parameter update if there is an overflow
    # manager_loss_scale = FixedLossScaleManager()
    # opt = nn.SGD(params=train_net.trainable_params(), learning_rate=lr_iter, momentum=0.9,
    #                   weight_decay=args.weight_decay)
    #
    # # 模式二 Execute parameter update even if overflow occurs
    manager_loss_scale = FixedLossScaleManager(args.loss_scale, drop_overflow_update=False)
    opt = nn.Momentum(params=train_net.trainable_params(), learning_rate=lr_iter, momentum=0.9,
                      weight_decay=args.weight_decay, loss_scale=args.loss_scale)

    amp_level = "O3" if args.device_target == "Ascend" else "O0"

    # model = Model(train_net, optimizer=opt, amp_level=amp_level, loss_scale_manager=manager_loss_scale)
    model = Model(network, loss_fn=loss_, optimizer=opt, amp_level=amp_level, loss_scale_manager=manager_loss_scale, metrics={"Dice":  MIoU(num_class=args.num_classes)})


    # callback for saving ckpts
    time_cb = TimeMonitor(data_size=iters_per_epoch)
    loss_cb = LossMonitor()
    # summary_cb = SummaryCollector(summary_dir='/cache/train_output/summary_dir',collect_freq=1)
    cbs = [time_cb, loss_cb]

    if args.rank == 0:
        config_ck = CheckpointConfig(save_checkpoint_steps=args.eval_per_epoch * iters_per_epoch, keep_checkpoint_max=args.keep_checkpoint_max)
        ckpoint_cb = ModelCheckpoint(prefix=args.model, directory=train_dir, config=config_ck)
        cbs.append(ckpoint_cb)
    # epoch_per_eval = {"epoch": [], "acc": []}
    # eval_cb = callback.EvalCallBack(model, val_dataset, args.eval_per_epoch, epoch_per_eval)
    # cbs.append(eval_cb)
    #
    # # epoch_per_eval = {"epoch": [], "acc": []}
    # # cbs.append(callback.SaveCallback(model, val_dataset, args, epoch_per_eval))
    # time4 = time.time()
    # print(time4-time3)
    model.train(args.train_epochs, train_dataset, callbacks=cbs)

    # show Model accuracy variation chart
    # callback.eval_show(epoch_per_eval, train_dir, args)
    # with open(os.path.join(train_dir, 'log_dice.txt'), 'w') as f:
    #     f.write(str(epoch_per_eval) )

    time5 = time.time()
    if args.modelArts_mode:
        # copy train result from cache to obs
        if args.rank == 0:
            mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
            # mox.file.copy_parallel(src_url='/cache/train_output/summary_dir', dst_url=args.train_url)
            mox.file.copy_parallel(src_url=local_plog_path, dst_url=args.train_url)
    time6 = time.time()
    # print(time6-time5)

if __name__ == '__main__':
    # 记录文件，时间和评估指标
    import moxing as mox
    # mox.file.shift('os', 'mox')
    args = parse_args()
    # if not os.path.exists(args.train_dir):
    #     os.makedirs(args.train_dir)

    # time1 = time.time()

    train()

    # time2 = time.time()
    # # print(time2-time1)
    # with open(args.train_dir + '/Record_train_parameters_set.txt', 'a') as f:
    #     f.write('\n\n')
    #     f.write('# ========================================================================================== \n')
    #     f.write('Train time:                         ' + str((time2 - time1) / 60) + ' mins\n')
    # mox.file.copy_parallel(src_url=args.train_dir, dst_url=args.train_url)

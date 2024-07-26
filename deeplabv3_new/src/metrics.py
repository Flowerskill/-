import numpy as np
from mindspore.nn.metrics.metric import Metric, rearrange_inputs
from src.tool import Mean_Intersection_over_Union, cal_hist, OverallAccuracy, Precision, Recall
from mindspore.nn import Softmax

class MIoU(Metric):

    def __init__(self, smooth=1e-5, num_class=5):
        """调用super进行初始化"""
        super(MIoU, self).__init__()
        self.smooth = smooth  # 防止除数为零，miou常用于三维医学分割指标
        self.num_class = num_class  # 防止除数为零，miou常用于三维医学分割指标
        # 调用clear清空变量
        self.clear()

    def clear(self):
        """清除内部计算结果，变量初始化"""
        self._miou_coeff_sum = 0
        self._samples_num = 0
        self.IoU = 0
        self.MIoU = 0
        self.OA = 0
        # self.pre = np.zeros(self.num_class)
        # self.rec = np.zeros(self.num_class)

    @rearrange_inputs
    def update(self, *inputs):
        """更新内部计算结果"""

        # 校验输入的数量，y_pred为预测值，y为实际值
        if len(inputs) != 2:
            raise ValueError('Miou need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))
        # 将输入的数据格式变为numpy array
        y_pred = self._convert_data(inputs[0]).argmax(axis=1)
        y = self._convert_data(inputs[1])

        # 参数计算
        self._samples_num += y.shape[0]

        # 校验输入的shape是否一致
        if y_pred.shape != y.shape:
            raise RuntimeError('y_pred and y should have same the dimension, but the shape of y_pred is{}, '
                               'the shape of y is {}.'.format(y_pred.shape, y.shape))

        self._miou_coeff_sum += cal_hist(y_pred, y, self.num_class)


    def eval(self):
        """进行MIoU计算"""
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')

        # MIoU that background is not calculated
        self.IoU, self.MIoU = Mean_Intersection_over_Union(self._miou_coeff_sum)
        self.OA = OverallAccuracy(self._miou_coeff_sum)
        precision = Precision(self._miou_coeff_sum)
        recall = Recall(self._miou_coeff_sum)
        return [self.MIoU, self.OA, precision, recall]


class Dice(Metric):

    def __init__(self, smooth=1e-5, num_class=5):
        """调用super进行初始化"""
        super(Dice, self).__init__()
        self.smooth = smooth  # 防止除数为零，dice常用于三维医学分割指标
        self.num_class = num_class
        # 调用clear清空变量
        self.clear()
        self.clear1()
        self.clear2()

    def clear(self):
        """清除内部计算结果，变量初始化"""
        """这里是进行完一个epoch后再进行清零"""
        self._samples_num = 0
        self.epoch_total_dice = 0
        self.epoch_avg_dice = 0

    def clear1(self):
        """自由定义清零位置"""
        self.oneimage_avg_dice = 0
        self._dice_coeff_sum = 0

    def clear2(self):
        """自由定义清零位置"""
        self.onebatch_total_dice = 0

    @rearrange_inputs
    def update(self, *inputs):
        """更新内部计算结果"""

        # 校验输入的数量，y_pred为预测值，y为实际值
        if len(inputs) != 2:
            raise ValueError('Dice need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        # since the shape of inputs[0] is (N, C, H, W), softmax operations are done in dimension C, so set axis to 1
        softmax = Softmax(axis=1)
        # apply softmax operation to inputs[0]
        # inputs[0] is unnormalized scores
        # inputs[0]是模型的输出，做softmax后化为(0, 1)之间的概率
        y_pred1 = softmax(inputs[0])
        class_set = []
        for i in range(self.num_class):
            class_set.append([i])
        # 将输入的数据格式变为numpy array
        y_pred = self._convert_data(y_pred1)
        y1 = self._convert_data(inputs[1])
        # 对y进行one-hot编码，并将其shape化为与y_pred相等的[N, C, H, W]
        semantic_map = []
        for index in class_set:
            equality = np.equal(y1, index)
            # class_map = np.all(equality, axis = 0)
            semantic_map.append(equality)
        semantic_map = np.stack(semantic_map, axis=0).astype(np.float32)
        y = semantic_map.reshape(semantic_map.shape[1], semantic_map.shape[0], semantic_map.shape[2],
                                 semantic_map.shape[3])
        # print (y)
        # the shape of y: (N, C, H, W), every channel is a binary class map
        # 参数计算
        # _samples_num表示一个epoch中样本数的累加，如果batchsize为2，则累加一次增加2，当完成一个epoch的预测时，_samples_num将是验证集中所有的样本数
        self._samples_num += y.shape[0]
        # print (self._samples_num)
        # 校验输入的shape是否一致
        if y_pred.shape != y.shape:
            raise RuntimeError('y_pred and y should have same the dimension, but the shape of y_pred is{}, '
                               'the shape of y is {}.'.format(y_pred.shape, y.shape))
        self.clear1()  # 计算不同batch之前，self.clear1()中的参数要清零
        # 根据公式实现Dice的过程计算
        for bs in range(y.shape[0]):
            bias = 0
            for cls in range(y.shape[1]):
                # 先求交集，利用dot对应点相乘再相加
                intersection = np.dot(y_pred[bs][cls].flatten(), y[bs][cls].flatten())
                # 求并集，先将输入shape都拉到一维，然后分别进行点乘，再将两个输入进行相加
                unionset = np.dot(y_pred[bs][cls].flatten(), y_pred[bs][cls].flatten()) + np.dot(y[bs][cls].flatten(),
                                                                                                 y[bs][cls].flatten())
                # 利用公式进行计算，加smooth是为了防止分母为0，避免当pred和true都为0时，分子被0除的问题，同时减少过拟合
                single_dice_coeff = 2 * float(intersection) / float(unionset + self.smooth)
                if single_dice_coeff != 0:
                    bias += 1
                # 对每一批次的系数进行累加
                self._dice_coeff_sum += single_dice_coeff  # the sum of Dice of classes
                # print("self._dice_coeff_sum:", self._dice_coeff_sum)
            self.oneimage_avg_dice = self._dice_coeff_sum / float(bias + self.smooth)  # average Dice of one image
            # print ("self.oneimage_avg_dice:", self.oneimage_avg_dice)
            self.onebatch_total_dice += self.oneimage_avg_dice  # the sum of Dice of samples in one batch
            # print ("self.onebatch_total_dice:", self.onebatch_total_dice)
            self.clear1()  # 计算同一batch不同图片的Dice值时，self.clear1()中的参数也要清零
        self.onebatch_avg_dice = self.onebatch_total_dice / y.shape[0]
        # print ("self.onebatch_avg_dice:", self.onebatch_avg_dice)
        self.epoch_total_dice += self.onebatch_avg_dice
        self.clear2()  # 计算完一个batch的Dice值后，self.clear2()中的参数要清零
        self.epoch_avg_dice = self.epoch_total_dice / float(self._samples_num / y.shape[0])  # 最终输出的Dice值

    def eval(self):
        """进行Dice计算"""
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self.epoch_avg_dice



class Dice_(Metric):

    def __init__(self, smooth=1e-5, num_class=5):
        """调用super进行初始化"""
        super(Dice_, self).__init__()
        self.smooth = smooth # 防止除数为零，dice常用于三维医学分割指标
        self.num_class = num_class
        # 调用clear清空变量
        self.clear()
        self.clear1()
        self.clear2()

    def clear(self):
        """清除内部计算结果，变量初始化"""
        """这里是进行完一个epoch后再进行清零"""
        self._samples_num = 0
        self.epoch_total_dice = 0
        self.epoch_avg_dice = 0

    def clear1(self):
        """自由定义清零位置"""
        self.oneimage_avg_dice = 0
        self._dice_coeff_sum = 0

    def clear2(self):
        """自由定义清零位置"""
        self.onebatch_total_dice = 0

    @rearrange_inputs
    def update(self, *inputs):
        """更新内部计算结果"""

        # 校验输入的数量，y_pred为预测值，y为实际值
        if len(inputs) != 2:
            raise ValueError('Dice need 2 inputs (y_pred, y), but got {}'.format(len(inputs)))

        # since the shape of inputs[0] is (N, C, H, W), softmax operations are done in dimension C, so set axis to 1
        #softmax = Softmax(axis=1)
        # apply softmax operation to inputs[0]
        # inputs[0] is unnormalized scores
        #y_pred1 = softmax(inputs[0])
        class_set = []
        for i in range(self.num_class):
            class_set.append([i])
        # 将输入的数据格式变为numpy array
        # 对y_pred1进行one-hot编码
        y_pred1 = self._convert_data(inputs[0]).argmax(axis = 1) #y_pred1是与y1格式相同的预测map
        semantic_map1 = []
        for index1 in class_set:
            equality1 = np.equal(y_pred1, index1)
            # class_map = np.all(equality, axis = 0)
            semantic_map1.append(equality1)
        semantic_map1 = np.stack(semantic_map1, axis=0).astype(np.float32)
        y_pred = semantic_map1.reshape(semantic_map1.shape[1], semantic_map1.shape[0], semantic_map1.shape[2], semantic_map1.shape[3])

        y1 = self._convert_data(inputs[1])
        semantic_map = []
        for index in class_set:
            equality = np.equal(y1, index)
            #class_map = np.all(equality, axis = 0)
            semantic_map.append(equality)
        semantic_map = np.stack(semantic_map, axis = 0).astype(np.float32)
        y = semantic_map.reshape(semantic_map.shape[1], semantic_map.shape[0], semantic_map.shape[2], semantic_map.shape[3])
        # print (y)
        # the shape of y: (N, C, H, W), every channel is a binary class map
        # 参数计算
        # _samples_num表示一个epoch中样本数的累加，如果batchsize为2，则累加一次增加2，当完成一个epoch的预测时，_samples_num将是验证集中所有的样本数
        self._samples_num += y.shape[0]
        #print (self._samples_num)
        # 校验输入的shape是否一致
        if y_pred.shape != y.shape:
            raise RuntimeError('y_pred and y should have same the dimension, but the shape of y_pred is{}, '
                               'the shape of y is {}.'.format(y_pred.shape, y.shape))
        self.clear1() # 计算不同batch之前，self.clear1()中的参数要清零
        # 根据公式实现Dice的过程计算
        for bs in range(y.shape[0]):
            bias = 0
            for cls in range(y.shape[1]):
                # 先求交集，利用dot对应点相乘再相加
                intersection = np.dot(y_pred[bs][cls].flatten(), y[bs][cls].flatten())
                # 求并集，先将输入shape都拉到一维，然后分别进行点乘，再将两个输入进行相加
                unionset = np.dot(y_pred[bs][cls].flatten(), y_pred[bs][cls].flatten()) + np.dot(y[bs][cls].flatten(), y[bs][cls].flatten())
                # 利用公式进行计算，加smooth是为了防止分母为0，避免当pred和true都为0时，分子被0除的问题，同时减少过拟合
                single_dice_coeff = 2 * float(intersection) / float(unionset + self.smooth)
                if single_dice_coeff != 0:
                    bias += 1
                # 对每一批次的系数进行累加
                self._dice_coeff_sum += single_dice_coeff # the sum of Dice of classes
                #print("self._dice_coeff_sum:", self._dice_coeff_sum)
            self.oneimage_avg_dice = self._dice_coeff_sum / float(bias + self.smooth) # average Dice of one image
            #print ("self.oneimage_avg_dice:", self.oneimage_avg_dice)
            self.onebatch_total_dice += self.oneimage_avg_dice # the sum of Dice of samples in one batch
            #print ("self.onebatch_total_dice:", self.onebatch_total_dice)
            self.clear1() # 计算同一batch不同图片的Dice值时，self.clear1()中的参数也要清零
        self.onebatch_avg_dice = self.onebatch_total_dice / y.shape[0]
        #print ("self.onebatch_avg_dice:", self.onebatch_avg_dice)
        self.epoch_total_dice += self.onebatch_avg_dice
        self.clear2() # 计算完一个batch的Dice值后，self.clear2()中的参数要清零
        self.epoch_avg_dice = self.epoch_total_dice / float(self._samples_num / y.shape[0]) # 最终输出的Dice值

    def eval(self):
        """进行Dice计算"""
        if self._samples_num == 0:
            raise RuntimeError('Total samples num must not be 0.')
        return self.epoch_avg_dice



class ErrorRateAt95Recall(Metric):

    def __init__(self):
        super(ErrorRateAt95Recall, self).__init__()
        self.clear()

    def clear(self):
        """Clears the internal evaluation result."""
        self.distances = []
        self.labels = []
        self.num_tests = 0

    @rearrange_inputs
    def update(self, *inputs):
        distance = self._convert_data(inputs[0])
        ll = self._convert_data(inputs[1])
        self.distances.append(distance)
        self.labels.append(ll)
        self.num_tests += ll.shape[0]

    def eval(self):
        if len(self.labels) == 0:
            raise RuntimeError('labels must not be 0.')
        distances = np.vstack(self.distances).reshape(self.num_tests)
        labels = np.vstack(self.labels).reshape(self.num_tests)
        recall_point = 0.95
        labels = labels[np.argsort(distances)]
        # Sliding threshold: get first index where recall >= recall_point.
        # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
        # 'recall_point' of the total number of elements with label==1.
        # (np.argmax returns the first occurrence of a '1' in a bool array).
        threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

        FP = np.sum(labels[:threshold_index] == 0)  # Below threshold (i.e., labelled positive), but should be negative
        TN = np.sum(labels[threshold_index:] == 0)  # Above threshold (i.e., labelled negative), and should be negative
        return float(FP) / float(FP + TN)

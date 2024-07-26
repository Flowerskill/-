import numpy as np
import argparse

from mindspore.common.tensor import Tensor
from mindspore._checkparam import Validator
import matplotlib.pyplot as plt
from mindspore.train.callback import Callback
from mindspore.communication.management import init, get_rank, get_group_size
current_epoch = []
current_loss = []
def parse_args():
    parser = argparse.ArgumentParser('MindSpore DeepLabV3+ training')
    parser.add_argument('--train_epochs', type=int, default=300, help='epoch, default=300')

    args, _ = parser.parse_known_args()
    return args

class LossFigure(Callback):

    def __init__(self, per_print_times=1):
        super(LossFigure, self).__init__()
        Validator.check_non_negative_int(per_print_times)
        self._per_print_times = per_print_times
        self._last_print_time = 0

    def step_end(self, run_context):
        args = parse_args()
        init()
        current_rank = get_rank()
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs

        if isinstance(loss, (tuple, list)):
            if isinstance(loss[0], Tensor) and isinstance(loss[0].asnumpy(), np.ndarray):
                loss = loss[0]

        if isinstance(loss, Tensor) and isinstance(loss.asnumpy(), np.ndarray):
            loss = float(np.mean(loss.asnumpy()))

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError("epoch: {} step: {}. Invalid loss, terminating training.".format(
                cb_params.cur_epoch_num, cur_step_in_epoch))

        #In disaster recovery scenario, the cb_params.cur_step_num may be rollback to previous step
        # and be less than self._last_print_time, so self._last_print_time need to be updated.

        if self._per_print_times != 0 and (cb_params.cur_step_num <= self._last_print_time):
            while cb_params.cur_step_num <= self._last_print_time:
                self._last_print_time -=\
                    max(self._per_print_times, cb_params.batch_num if cb_params.dataset_sink_mode else 1)

        if self._per_print_times != 0 and (cb_params.cur_step_num - self._last_print_time) >= self._per_print_times:
            self._last_print_time = cb_params.cur_step_num
            if current_rank == 0:
                current_epoch.append(cb_params.cur_epoch_num)
                current_loss.append(loss)

                if cb_params.cur_epoch_num == args.train_epochs:
                    import moxing as mox
                    local_train_url = '/cache/ckpt'
                    with open(local_train_url + '/loss.txt', 'a') as f:
                        f.write('epoch list is: ')
                        f.write('\n')
                        f.write(str(current_epoch))
                        f.write('\n')
                        f.write("loss list is: ")
                        f.write('\n')
                        f.write(str(current_loss))
                    figure_dir = local_train_url + '/loss.png'
                    plt.xlabel("epoch")
                    plt.ylabel("loss")
                    plt.title("Loss Curve")
                    print("current_epoch is ", current_epoch)
                    print("current_loss is ", current_loss)
                    plt.plot(current_epoch, current_loss, color='red', marker='.')
                    plt.savefig(figure_dir)















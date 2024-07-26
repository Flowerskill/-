from mindspore.train.callback import Callback
import matplotlib.pyplot as plt
import os
from mindspore import save_checkpoint
class EvalCallBack(Callback):
    """Precision verification using callback function."""
    # define the operator required
    def __init__(self, models, eval_dataset, tr_dataset, eval_per_epochs, eval_out):
        super(EvalCallBack, self).__init__()
        self.models = models
        self.train_dataset = tr_dataset
        self.eval_dataset = eval_dataset
        self.eval_per_epochs = eval_per_epochs
        self.eval_out = eval_out

    # define operator function in epoch end
    def epoch_end(self, run_context):
        cb_param = run_context.original_args()
        cur_epoch = cb_param.cur_epoch_num
        if cur_epoch % self.eval_per_epochs == 0:
            acc = self.models.eval(self.eval_dataset)
            self.eval_out["epoch"].append(cur_epoch)
            self.eval_out["Miou"].append(acc["out"][0])
            self.eval_out["OA"].append(acc["out"][1])
            self.eval_out["Precision"].append(acc["out"][2])
            self.eval_out["Recall"].append(acc["out"][3])
            train_acc = self.models.eval(self.train_dataset)
            self.eval_out["tr_OA"].append(train_acc["out"][1])
            print(train_acc['out'][1])
            print(acc['out'][1])
            print(acc['out'][0])
            print(acc['out'][2])
            print(acc['out'][3])
            # print(acc)

def eval_show(epoch_per_eval, path, args):
    plt.xlabel("epoch number")
    plt.ylabel("Model accuracy")
    plt.title("Model accuracy variation chart")
    plt.plot(epoch_per_eval["epoch"], epoch_per_eval["acc"], "red")
    plt.savefig(os.path.join(path, "train_epochs_" + str(args.train_epochs) + "_batch_size_" + str(args.batch_size) +
                              "_lr_type_" + str(args.lr_type) + "_base_lr_" + str(args.base_lr) +
                              "_lr_decay_rate_" + str(args.lr_decay_rate) + "_weight_decay_" + str(args.weight_decay) + "_Dice.png"))
    plt.show()



class SaveCallback(Callback):
    def __init__(self, eval_model, ds_eval, args, epochs_per_eval):
        super(SaveCallback, self).__init__()
        self.model = eval_model
        self.ds_eval = ds_eval
        self.acc = 0
        self.train_dir = args.train_dir
        self.epochs_per_eval = epochs_per_eval

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        result = self.model.eval(self.ds_eval)
        self.epochs_per_eval["epoch"].append(cur_epoch)
        self.epochs_per_eval["acc"].append(result["Dice"])
        print(result)

        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        if result['Dice'] > self.acc:
            self.acc = result['Dice']
            file_name = os.path.join(self.train_dir, str(cur_epoch) + "__" + str(self.acc) + ".ckpt")
            save_checkpoint(save_obj=cb_params.train_network, ckpt_file_name=file_name)
            print("Save the maximum accuracy checkpoint,the Dice is", self.acc)








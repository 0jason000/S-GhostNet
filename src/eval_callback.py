
import os
import stat
import time
from mindspore import save_checkpoint
from mindspore import log as logger
from mindspore.train.callback import Callback


class EvalCallBack(Callback):
    def __init__(self, eval_function, eval_param_dict, interval=1, eval_start_epoch=1, save_best_ckpt=True,
                 ckpt_directory="./", besk_ckpt_name="best.ckpt", metrics_name="acc"):
        super(EvalCallBack, self).__init__()
        self.eval_param_dict = eval_param_dict
        self.eval_function = eval_function
        self.eval_start_epoch = eval_start_epoch
        if interval < 1:
            raise ValueError("interval should >= 1.")
        self.interval = interval
        self.save_best_ckpt = save_best_ckpt
        self.best_res = 0
        self.best_epoch = 0
        if not os.path.isdir(ckpt_directory):
            os.makedirs(ckpt_directory)
        self.bast_ckpt_path = os.path.join(ckpt_directory, besk_ckpt_name)
        self.metrics_name = metrics_name

    def remove_ckpoint_file(self, file_name):
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        if cur_epoch >= self.eval_start_epoch and (cur_epoch - self.eval_start_epoch) % self.interval == 0:
            eval_start = time.time()
            res = self.eval_function(self.eval_param_dict)
            eval_cost = time.time() - eval_start
            print("epoch: {}, {}: {}, eval_cost:{:.2f}".format(cur_epoch, self.metrics_name, res, eval_cost))
            if res >= self.best_res:
                self.best_res = res
                self.best_epoch = cur_epoch
                print("update best result: {}".format(res))
                if self.save_best_ckpt:
                    if os.path.exists(self.bast_ckpt_path):
                        self.remove_ckpoint_file(self.bast_ckpt_path)
                    save_checkpoint(cb_params.train_network, self.bast_ckpt_path)
                    print("update best checkpoint at: {}".format(self.bast_ckpt_path))

    def end(self, run_context):
        v = run_context
        assert v == run_context
        print("End training, the best {0} is: {1}, the best {0} epoch is {2}".format(self.metrics_name,
                                                                                     self.best_res,
                                                                                     self.best_epoch))

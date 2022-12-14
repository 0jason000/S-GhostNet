
"""model utils"""
import math
import argparse

import numpy as np


def str2bool(value):
    """Convert string arguments to bool type"""
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


def get_lr_tinynet_c(base_lr, total_epochs, steps_per_epoch, decay_epochs=1, decay_rate=0.9,
                     warmup_epochs=0., warmup_lr_init=0., global_epoch=0):
    """Get scheduled learning rate"""
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    global_steps = steps_per_epoch * global_epoch
    self_warmup_delta = ((base_lr - warmup_lr_init) / \
                         warmup_epochs) if warmup_epochs > 0 else 0
    self_decay_rate = decay_rate if decay_rate < 1 else 1/decay_rate
    for i in range(total_steps):
        epochs = math.floor(i/steps_per_epoch)
        cond = 1 if (epochs < warmup_epochs) else 0
        warmup_lr = warmup_lr_init + epochs * self_warmup_delta
        decay_nums = math.floor(epochs / decay_epochs)
        decay_rate = math.pow(self_decay_rate, decay_nums)
        decay_lr = base_lr * decay_rate
        lr = cond * warmup_lr + (1 - cond) * decay_lr
        lr_each_step.append(lr)
    lr_each_step = lr_each_step[global_steps:]
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step


def get_lr(lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(int): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_end + \
                (lr_max - lr_end) * \
                (1. + math.cos(math.pi * (i - warmup_steps) /
                               (total_steps - warmup_steps))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)

    return lr_each_step


def add_weight_decay(net, weight_decay=1e-5, skip_list=None):
    """Apply weight decay to only conv and dense layers (len(shape) > =2)
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
        weight_decay (float): weight decay tobe used.
        skip_list (tuple): list of parameter names without weight decay
    Returns:
        A list of group of parameters, separated by different weight decay.
    """
    decay = []
    no_decay = []
    if not skip_list:
        skip_list = ()
    for param in net.trainable_params():
        if len(param.shape) == 1 or \
           param.name.endswith(".bias") or \
           param.name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def count_params(net):
    """Count number of parameters in the network
    Args:
        net (mindspore.nn.Cell): Mindspore network instance
    Returns:
        total_params (int): Total number of trainable params
    """
    total_params = 0
    for param in net.trainable_params():
        total_params += np.prod(param.shape)
    return total_params

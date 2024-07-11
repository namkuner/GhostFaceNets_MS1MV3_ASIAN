import logging
import os
import time
from typing import List

import torch
from eval import verification
from utils.utils_logging import AverageMeter
from torch.utils.tensorboard import SummaryWriter
from torch import distributed
from eval.vilfw import evaluate
from eval.vilfw import VILWFDataset
import numpy as np
from configs.ghostfacenets import config
class CallBackVerification():

    def __init__(self, summary_writer=None, wandb_logger=None,bath_size =64,val_loader =None):
        self.rank: int = 0
        self.batch_size =bath_size
        self.summary_writer = summary_writer
        self.wandb_logger = wandb_logger
        self.val_loader =val_loader
        self.data_dir = config.data_dir
    def ver_test(self, backbone: torch.nn.Module, global_step: int):


        same_acc, diff_acc, overall_acc, auc, threshs = evaluate(self.val_loader, backbone, self.batch_size)
        logging.info(f'[{global_step}] Accuracy/Val_same_accuracy mean: {same_acc:.4f}')
        logging.info(f'[{global_step}] Accuracy/Val_diff_accuracy mean: {diff_acc:.4f}')
        logging.info(f'[{global_step}] Accuracy/Val_accuracy mean: {overall_acc:.4f}')
        logging.info(f'[{global_step}] AUC: {auc:.4f}')
        logging.info(f'[{global_step}] Estimated threshold: {np.mean(threshs):.4f}')
        self.summary_writer: SummaryWriter
        self.summary_writer.add_scalar(tag='%s with globe step %d'.format(self.data_dir,global_step), scalar_value=overall_acc, global_step=global_step )
        if self.wandb_logger:
            import wandb
            self.wandb_logger.log({
                f'VILFW overall_acc - namkuner': overall_acc,
                f'VILFW same_acc - namkuner': same_acc,
                f'VILFW diff_acc - namkuner' : diff_acc,
                f'VILFW AUC - namkuner ': auc,
                f'VILFW thresh- namkuner': np.mean(threshs)

            })





    def __call__(self, num_update, backbone: torch.nn.Module):
        if self.rank == 0 and num_update > 0:
            backbone.eval()
            self.ver_test(backbone, num_update)
            backbone.train()


class CallBackLogging(object):
    def __init__(self, frequent, total_step, batch_size, start_step=0, writer=None):
        self.frequent: int = frequent
        self.rank: int = -0
        self.world_size: int = 1
        self.time_start = time.time()
        self.total_step: int = total_step
        self.start_step: int = start_step
        self.batch_size: int = batch_size
        self.writer = writer

        self.init = False
        self.tic = 0

    def __call__(self,
                 global_step: int,
                 loss: AverageMeter,
                 epoch: int,
                 fp16: bool,
                 learning_rate: float,
                 grad_scaler: torch.cuda.amp.GradScaler):
        if self.rank == 0 and global_step > 0 and global_step % self.frequent == 0:
            if self.init:
                try:
                    speed: float = self.frequent * self.batch_size / (time.time() - self.tic)
                    speed_total = speed * self.world_size
                except ZeroDivisionError:
                    speed_total = float('inf')

                # time_now = (time.time() - self.time_start) / 3600
                # time_total = time_now / ((global_step + 1) / self.total_step)
                # time_for_end = time_total - time_now
                time_now = time.time()
                time_sec = int(time_now - self.time_start)
                time_sec_avg = time_sec / (global_step - self.start_step + 1)
                eta_sec = time_sec_avg * (self.total_step - global_step - 1)
                time_for_end = eta_sec / 3600
                if self.writer is not None:
                    self.writer.add_scalar('time_for_end', time_for_end, global_step)
                    self.writer.add_scalar('learning_rate', learning_rate, global_step)
                    self.writer.add_scalar('loss', loss.avg, global_step)
                if fp16:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Fp16 Grad Scale: %2.f   Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step,
                              grad_scaler.get_scale(), time_for_end
                          )
                else:
                    msg = "Speed %.2f samples/sec   Loss %.4f   LearningRate %.6f   Epoch: %d   Global Step: %d   " \
                          "Required: %1.f hours" % (
                              speed_total, loss.avg, learning_rate, epoch, global_step, time_for_end
                          )
                logging.info(msg)
                loss.reset()
                self.tic = time.time()
            else:
                self.init = True
                self.tic = time.time()

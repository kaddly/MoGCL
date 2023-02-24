import os
import shutil
import time
import torch
from torch import nn
from train.metric_utils import AverageMeter, ProgressMeter
from train.optimizer_utils import create_lr_scheduler


def save_checkpoint(state, is_best, file_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join(os.path.abspath('.'), file_dir, filename))
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def train():
    pass


def train_one_epoch(train_loader, model, criterion, optimizer, lr_scheduler, epoch, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.4e")
    learning_rate = AverageMeter("lr", ':6.3f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, learning_rate],
        prefix="Epoch: [{}]".format(epoch),
    )

    # switch to train mode
    model.train()
    end = time.time()
    for i, batch in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        nodes, nodes_neigh, pos_nodes, pos_nodes_neigh, neg_index = [data.to(device) for data in batch]
        output = model((nodes, nodes_neigh), (pos_nodes, pos_nodes_neigh), neg_index)
        loss = criterion(output)
        losses.update(loss.item(), batch[0].size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        learning_rate.update(optimizer.param_groups[0]["lr"])
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

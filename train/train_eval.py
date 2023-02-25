import os
import shutil
import time
import datetime
import torch
from torch import nn
from utils import setup_logging
from module import MoGCL
from train.metric_utils import AverageMeter, ProgressMeter
from train.optimizer_utils import create_lr_scheduler
from train.loss_utils import SigmoidCELoss


def save_checkpoint(state, is_best, file_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join("models", file_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join("models", file_dir, filename),
                        os.path.join("models", file_dir, "model_best.pth.tar"))


def train(train_iter, feat_data, val_loader, index_loader, args):
    setup_logging(args.dataset)
    # save train info
    results_file = os.path.join("results", args.dataset,
                                "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    device = args.device
    model = MoGCL(feat_data, args.dim, args.num_view, args.num_pos, args.num_neigh, args.attn_size, args.feat_drop,
                  args.attn_drop, len(feat_data), args.mco_m, args.moco_t, args.is_mlp)
    criterion = SigmoidCELoss(args.num_pos)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, 1, args.num_epoch)

    cnt_wait = 0
    best = 1e9
    best_t = 0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            print("=> loaded checkpoint '{}' (epoch {})".format(args.resume, checkpoint["epoch"]))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
    model.to(device)
    for epoch in range(args.start_eopch, args.epochs):
        mean_loss, lr = train_one_epoch(train_iter, model, criterion, optimizer, lr_scheduler, epoch, device, args)
        val_loss = val_evaluate(model, criterion, val_loader, device)
        val_info = f"val_loss: {val_loss:>5.4f}"
        print(val_info)
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {mean_loss:>5.4f},  lr: {lr:>5.4f}\n"
            f.write(train_info + val_info + "\n\n")
        if val_loss < best:
            best = val_loss
            is_best = True
            best_t = epoch
            cnt_wait = 0
            print("save best parameters")
        else:
            is_best = False
            cnt_wait += 1
        save_checkpoint(
            {
                "epoch": epoch + 1,
                "arch": args.arch,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": lr_scheduler.state_dict()
            },
            is_best=is_best,
            file_dir=args.dataset,
            filename="checkpoint_{:04d}.pth.tar".format(epoch),
        )
        if cnt_wait == args.patience:
            print('Early stopping!')
            args.resume = os.path.join("models", args.dataset, "checkpoint_{:04d}.pth.tar".format(best_t))
            break
    embeds = get_embeds(model, index_loader, device, args)


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

        lr = optimizer.param_groups[0]["lr"]
        learning_rate.update(lr)
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
    return losses.avg, lr


def val_evaluate(model, criterion, val_loader, device=None):
    if isinstance(model, nn.Module):
        model.eval()
        if not device:
            device = next(iter(model.parameters())).device
    nodes, nodes_neigh, pos_nodes, pos_nodes_neigh, neg_index = [data.to(device) for data in val_loader]
    output = model((nodes, nodes_neigh), (pos_nodes, pos_nodes_neigh), neg_index)
    return criterion(output)


def get_embeds(model, index_loader, device, args):
    checkpoint = torch.load(args.resume)
    model.load_state_dict(checkpoint["state_dict"])
    if isinstance(model, nn.Module):
        model.eval()
        if not device:
            device = next(iter(model.parameters())).device
    nodes, nodes_neigh = [data.to(device) for data in index_loader]
    embeds = model.get_embeds(nodes, nodes_neigh)
    with open(os.path.join("embeds", args.dataset, 'nodes_embeds.txt'), "wb") as f:
        f.writelines(embeds.cpu().data.numpy())
        f.close()
    return embeds


def evaluate(embeds):
    pass

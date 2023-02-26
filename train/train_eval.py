import os
import shutil
import time
import datetime
import numpy as np
import pickle
from sklearn.metrics import f1_score, roc_auc_score, recall_score
from imblearn.over_sampling import RandomOverSampler
import torch
from torch import nn
from utils import load_data, setup_logging
from module import MoGCL, LogReg
from train.metric_utils import AverageMeter, ProgressMeter
from train.optimizer_utils import create_lr_scheduler
from train.loss_utils import SigmoidCELoss


def save_checkpoint(state, is_best, file_dir, filename="checkpoint.pth.tar"):
    torch.save(state, os.path.join("models", file_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join("models", file_dir, filename),
                        os.path.join("models", file_dir, "model_best.pth.tar"))


def train(args):
    setup_logging(args.dataset)
    # save train info
    results_file = os.path.join("results", args.dataset,
                                "results{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    # load data
    train_iter, feat_data, val_loader, index_loader, test_loader = load_data(args)
    device = args.device
    model = MoGCL(feat_data, args.dim, args.num_view, args.num_pos, args.num_neigh,
                  args.attn_size, args.feat_drop, args.attn_drop, len(feat_data), args.moco_m, args.moco_t, args.is_mlp)
    criterion = SigmoidCELoss(args.num_pos)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=args.weight_decay)
    lr_scheduler = create_lr_scheduler(optimizer, 1, args.epochs)

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
    for epoch in range(args.start_epoch, args.epochs):
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
    all_embeds = get_embeds(model, index_loader, device, args)
    evaluate(*test_loader, args, all_embeds)


def train_one_epoch(train_loader, model, criterion, optimizer, lr_scheduler, epoch, device, args):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.4f")
    learning_rate = AverageMeter("lr", ':6.4f')
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
    with torch.no_grad():
        output = model((nodes, nodes_neigh), (pos_nodes, pos_nodes_neigh), neg_index)
        loss = criterion(output)
    return loss


def get_embeds(model, index_loader, device, args):
    checkpoint = torch.load(os.path.join('models', args.dataset, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint["state_dict"])
    if isinstance(model, nn.Module):
        model.eval()
        if not device:
            device = next(iter(model.parameters())).device
    nodes, nodes_neigh = [torch.tensor(data).to(device) for data in index_loader]
    with torch.no_grad():
        embeds = model.get_embeds(nodes, nodes_neigh)
    all_embeds = {}
    embeds = embeds.cpu().data.numpy()
    for i, node in enumerate(index_loader[0]):
        all_embeds[node] = embeds[i]
    with open(os.path.join("embeds", args.dataset, 'nodes_embeds.pkl'), "wb") as f:
        pickle.dump(all_embeds, f)
        f.close()
    return all_embeds


def evaluate(train_idx, val_idx, train_labels, val_labels, args, all_embeds=None):
    if all_embeds is None:
        if os.path.isfile(os.path.join('embeds', args.dataset, 'nodes_embeds.pkl')):
            with open(os.path.join("embeds", args.dataset, 'nodes_embeds.pkl'), "rb") as f:
                all_embeds = pickle.load(f)
                f.close()
        else:
            raise FileExistsError('please train before')
    ros = RandomOverSampler(random_state=args.seed)
    train_resample_x, train_resample_y = ros.fit(train_idx, train_labels)
    val_resample_x, val_resample_y = ros.fit(val_idx, val_labels)
    train_embeds, val_embeds = map(all_embeds, train_resample_x), map(all_embeds, val_resample_x)
    device = args.device

    auc_score_list = []
    macro_f1s = []
    macro_recalls = []

    criterion = nn.CrossEntropyLoss()

    for _ in range(50):
        log = LogReg(args.dim, 2)
        opt = torch.optim.AdamW(log.parameters(), args.eva_lr, weight_decay=args.eva_wd)
        log.to(device)

        val_macro_f1s = []
        val_macro_recalls = []

        logits_list = []
        for iter_ in range(200):
            # train
            log.train()
            opt.zero_grad()

            logits = log(train_embeds)
            loss = criterion(logits, train_resample_y)

            loss.backward()
            opt.step()

            # val
            log.eval()
            with torch.no_grad():
                logits = log(val_embeds)
                preds = torch.argmax(logits, dim=1)

            val_f1_macro = f1_score(val_resample_y.cpu(), preds.cpu(), average='macro')
            val_recall_macro = recall_score(val_resample_y.cpu(), preds.cpu(), average='macro')

            val_macro_f1s.append(val_f1_macro)
            val_macro_recalls.append(val_recall_macro)

            logits_list.append(logits)

        macro_f1s.append(max(val_macro_f1s))
        macro_recalls.append(max(val_macro_recalls))

        max_iter = val_macro_f1s.index(max(val_macro_f1s))

        # auc
        best_logits = logits_list[max_iter]
        best_proba = nn.functional.softmax(best_logits, dim=1)
        auc_score_list.append(
            roc_auc_score(y_true=val_resample_y.detach().cpu().numpy(), y_score=best_proba.detach().cpu().numpy()))
    print(
        "\t[Classification] Macro-F1_mean: {:.4f} var: {:.4f} max: {:.4f}\nMacro-F1_mean: {:.4f} var: {:.4f} max: {:.4f}\nauc_mean: {:.4f} var: {:.4f} max: {:.4f}"
            .format(np.mean(macro_f1s), np.var(macro_f1s), np.max(macro_f1s),
                    np.mean(macro_recalls), np.var(macro_recalls), np.max(macro_recalls),
                    np.mean(auc_score_list), np.var(auc_score_list), np.max(auc_score_list)))
    f = open(os.path.join("result", args.dataset, "result.txt"), "a")
    f.write(str(np.mean(macro_f1s)) + "\t" + str(np.mean(macro_recalls)) + "\t" + str(np.mean(auc_score_list)) + "\n" +
            str(np.max(macro_f1s)) + "\t" + str(np.max(macro_recalls)) + "\t" + str(np.max(auc_score_list)) + "\n")
    f.close()

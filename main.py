from liftformer.utils import AverageMeter
from liftformer import utils
import torch
from tqdm import tqdm
from datasets.process import unNormalizeData
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datasets.Human36M import Human36M, actions, HorizentalFlip
from torch.utils.data import DataLoader
import os
from liftformer.Model import Liftformer
import torch.nn as nn
from liftformer.Optim import NoamLR
import matplotlib.pyplot as plt
from datasets.Human36M import show3Dpose, show2Dpose
import math
from opt import Options


def main(opt):

    utils.save_options(opt, opt.log_path)

    if torch.cuda.is_available():
        print(">>> Using cuda backend")
        torch.backends.cudnn.benchmark = True

    print(">>> Using tensorboard")
    writer = SummaryWriter(opt.log_path)

    print(">>> creating model")
    model = Liftformer(d_model=opt.d_model,
                       receptive_field=opt.receptive_field,
                       n_layers=opt.n_layers,
                       n_head=opt.n_head,
                       d_in=opt.d_in,
                       d_out=opt.d_out,
                       d_inner=opt.d_inner,
                       d_k=opt.d_k,
                       d_v=opt.d_v,
                       dropout=opt.dropout,
                       pre_LN=opt.pre_LN)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    criterion = nn.MSELoss()
    if torch.cuda.is_available():
        model = model.cuda()
        criterion = criterion.cuda()
    # create optimizers
    # optimizer = NoamOpt(torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-09), 2.0, 512, 4000)
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.98), eps=1e-09)
    if opt.use_noam:
        scheduler = NoamLR(optimizer, 4000)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True)

    # load checkpoint if needed/ wanted
    start_epoch = 0
    glob_step = 0
    best_error = 10000
    if opt.load:
        print(">>> loading ckpt from '{}'".format(opt.load))
        ckpt = torch.load(opt.load)
        model.load_state_dict(ckpt['model'])
        start_epoch = ckpt['epoch']
        glob_step = ckpt['glob_step']
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        best_error = ckpt['best_error']
        print(">>> ckpt loaded (epoch: {} | err: {})".format(start_epoch, best_error))

    print(">>> loading data")
    stat_3d = torch.load(os.path.join(opt.data_path, 'stat_3d.pth.tar'))
    stat_2d = torch.load(os.path.join(opt.data_path, 'stat_2d.pth.tar'))

    if opt.test:
        err_set = []
        for action in actions:
            print(">>> TEST on {}".format(action))
            test_data_loader = DataLoader(dataset=Human36M(actions=[action],
                                                           data_path=opt.data_path,
                                                           receptive_field=opt.receptive_field,
                                                           is_train=False),
                                          batch_size=opt.test_batchsize,
                                          shuffle=False,
                                          num_workers=opt.num_workers,
                                          pin_memory=opt.pin_memory)
            _, err_test = test(test_data_loader, model, criterion, opt.receptive_field, stat_2d, stat_3d,
                               opt.log_img_freq)
            err_set.append(err_test)
        print(">>>>>> TEST results:")
        for action in actions:
            print("{}".format(action), end='\t')
        print("\n")
        for err in err_set:
            print("{:.4f}".format(err), end='\t')
        print(">>>\nERRORS: {}".format(np.array(err_set).mean()))
        return

    transform = HorizentalFlip()
    train_dataset = Human36M(actions=actions,
                             data_path=opt.data_path,
                             receptive_field=27,
                             is_train=True,
                             transform=transform)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=opt.train_batchsize,
                                   shuffle=True,
                                   drop_last=True,
                                   num_workers=opt.num_workers,
                                   pin_memory=opt.pin_memory)
    test_dataset = Human36M(actions=actions,
                            data_path=opt.data_path,
                            receptive_field=27,
                            is_train=False,
                            transform=None)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=opt.test_batchsize,
                                  shuffle=False,
                                  drop_last=True,
                                  num_workers=opt.num_workers,
                                  pin_memory=opt.pin_memory)
    print(">>> data loaded !")

    for epoch in range(start_epoch, opt.epochs):
        writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
        glob_step, train_loss = train(train_data_loader, model, criterion, optimizer, writer, glob_step, epoch, stat_2d,
                                      stat_3d, opt.receptive_field, opt.log_img_freq)
        test_loss, test_error = test(test_data_loader, model, criterion, opt.receptive_field, stat_2d, stat_3d,
                                     opt.log_img_freq, writer, epoch)
        scheduler.step(test_loss)
        is_best = 'best' if best_error > test_error else ''
        best_error = min(best_error, test_error)

        # save checkpoint if needed
        cpkt = {
            'model': model.state_dict(),
            'epoch': epoch,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'glob_step': glob_step,
            'best_error': best_error
        }
        torch.save(cpkt, os.path.join(opt.log_path, '{}{}.ckpt'.format(epoch, is_best)))


def show_fig(input, output, gt):
    fig = plt.figure(figsize=(9, 3))
    ax1 = fig.add_subplot(131)
    ax1.title.set_text('2D input')
    show2Dpose(input, ax1)
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.title.set_text('3D pred')
    show3Dpose(output, ax2)
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.title.set_text('3D gt')
    show3Dpose(gt, ax3)
    return fig


def train(train_loader, model, criterion, optimizer, writer, train_n_iter, epoch, stat_2d, stat_3d, receptive_field,
          log_img_freq):

    receptive_center = math.floor(receptive_field / 2)
    losses = AverageMeter()
    model.train()

    with tqdm(train_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Train Epoch {epoch}")
        for i, (inps, tars) in enumerate(tepoch):
            if torch.cuda.is_available():
                inps = inps.cuda()
                tars = tars.cuda()

            outputs = model(inps)

            # calculate loss
            optimizer.zero_grad()
            loss = criterion(outputs, tars)
            loss.backward()
            losses.update(loss.item(), inps.size(0))
            optimizer.step()

            writer.add_scalar('train_loss', loss.item(), train_n_iter)
            if train_n_iter % log_img_freq == 1:
                show_input = inps[0][receptive_center:receptive_center + 1].cpu().detach().numpy()
                show_input = unNormalizeData(show_input, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use']).flatten()

                show_output = outputs[0].cpu().detach().numpy()
                show_output = unNormalizeData(show_output, stat_3d['mean'], stat_3d['std'],
                                              stat_3d['dim_use']).flatten()

                show_gt = tars[0].cpu().detach().numpy()
                show_gt = unNormalizeData(show_gt, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use']).flatten()

                fig = show_fig(show_input, show_output, show_gt)
                writer.add_figure('train figure', fig, train_n_iter)
            train_n_iter += 1

            tepoch.set_postfix(loss=losses.avg())

    return train_n_iter, losses.avg()


def test(test_loader, model, criterion, receptive_field, stat_2d, stat_3d, log_img_freq, writer=None, epoch=0):
    receptive_center = math.floor(receptive_field / 2)
    losses = AverageMeter()
    errors = AverageMeter()
    test_n_iter = 0
    model.eval()

    all_dist = []

    with tqdm(test_loader, unit="batch") as tepoch:
        tepoch.set_description(f"Test  Epoch {epoch}")
        for i, (inps, tars) in enumerate(tepoch):
            if torch.cuda.is_available():
                inps = inps.cuda()
                tars = tars.cuda()

            outputs = model(inps)

            # calculate loss
            loss = criterion(outputs, tars)
            losses.update(loss.item(), inps.size(0))

            if writer is not None and test_n_iter % log_img_freq == 1:
                show_input = inps[0][receptive_center:receptive_center + 1].cpu().detach().numpy()
                show_input = unNormalizeData(show_input, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use']).flatten()

                show_output = outputs[0].cpu().detach().numpy()
                show_output = unNormalizeData(show_output, stat_3d['mean'], stat_3d['std'],
                                              stat_3d['dim_use']).flatten()

                show_gt = tars[0].cpu().detach().numpy()
                show_gt = unNormalizeData(show_gt, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use']).flatten()

                fig = show_fig(show_input, show_output, show_gt)
                writer.add_figure('test figure', fig, epoch * 100 + test_n_iter)
            test_n_iter += 1

            # calculate erruracy
            targets_unnorm = unNormalizeData(tars.data.cpu().numpy().squeeze(), stat_3d['mean'], stat_3d['std'],
                                             stat_3d['dim_use'])
            outputs_unnorm = unNormalizeData(outputs.data.cpu().numpy().squeeze(), stat_3d['mean'], stat_3d['std'],
                                             stat_3d['dim_use'])

            # remove dim ignored
            dim_use = np.hstack((np.arange(3), stat_3d['dim_use']))

            outputs_use = outputs_unnorm[:, dim_use]
            targets_use = targets_unnorm[:, dim_use]

            sqerr = (outputs_use - targets_use)**2

            distance = np.zeros((sqerr.shape[0], 17))
            dist_idx = 0
            for k in np.arange(0, 17 * 3, 3):
                distance[:, dist_idx] = np.sqrt(np.sum(sqerr[:, k:k + 3], axis=1))
                dist_idx += 1
            errors.update(np.mean(distance), distance.size)

            tepoch.set_postfix(loss=losses.avg(), error=errors.avg())
        if writer is not None:
            writer.add_scalar('MPJPE error', errors.avg(), epoch)
            writer.add_scalar('val_loss', losses.avg(), epoch)

    return losses.avg(), errors.avg()


if __name__ == '__main__':
    option = Options().parse()
    main(option)

import argparse
import os
import torch
import json
from liftformer.Model import Liftformer
import numpy as np
import math
from datasets.Human36M import show3Dpose, show2Dpose
import matplotlib.pyplot as plt


def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_use):

    T = normalized_data.shape[0]  # Batch size
    D = data_mean.shape[0]  # 96

    orig_data = np.zeros((T, D), dtype=np.float32)

    orig_data[:, dimensions_to_use] = normalized_data

    # Multiply times stdev and add the mean
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    orig_data = np.multiply(orig_data, stdMat) + meanMat
    return orig_data


def show2DposeLandmk(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False):
    """Visualize a 2d skeleton
    Args
      channels: 64x1 vector. The pose to plot.
      ax: matplotlib axis to draw on
      lcolor: color for left part of the body
      rcolor: color for right part of the body
      add_labels: whether to add coordinate labels
    Returns
      Nothing. Draws on ax.

      MIT License

    Copyright (c) 2016 Julieta Martinez, Rayat Hossain, Javier Romero

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """

    assert channels.size == 33 * 2
    vals = np.reshape(channels, (33, -1))
    # yapf: disable
    I = np.array([ 0, 11, 13,  0, 12, 14, 11, 23, 25, 12, 24, 26])  # start points
    J = np.array([11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28])  # end points
    LR = np.array([1,  1,  1,  0,  0,  0,  0,  0,  0,  1,  1,  1], dtype=bool)
    # yapf: enable
    # Make connection matrix
    for i in np.arange(len(I)):
        x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
        ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 350  # space around the subject
    xroot, yroot = vals[0, 0], vals[0, 1]
    #ax.set_xlim([-RADIUS + xroot, RADIUS + xroot])
    #ax.set_ylim([-RADIUS + yroot, RADIUS + yroot])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    ax.set_aspect('equal')


def show_fig(input, output):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('2D input')
    show2Dpose(input, ax1, hide_ticks=False)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.title.set_text('3D pred')
    show3Dpose(output, ax2, hide_ticks=False)
    return fig


def main():

    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default='', help='path to load a pretrained checkpoint')
    parser.add_argument('--input',
                        '-i',
                        type=str,
                        default='all_poses_modified.npy',
                        help='path to load a pretrained checkpoint')
    opt = parser.parse_args()

    # load input
    input = np.load(opt.input)
    print(input.shape)
    fig, ax = plt.subplots()
    show2DposeLandmk(input.reshape(-1, 66)[0], ax)
    fig.savefig('data/demo/input.png')
    del fig, ax

    # load model options
    ckpt = torch.load(opt.load)
    jsonFile = os.path.join(os.path.dirname(opt.load), 'opt.json')
    f = open(jsonFile)
    opt = json.load(f)
    f.close()

    # load model
    model = Liftformer(d_model=opt['d_model'],
                       receptive_field=opt['receptive_field'],
                       n_layers=opt['n_layers'],
                       n_head=opt['n_head'],
                       d_in=opt['d_in'],
                       d_out=opt['d_out'],
                       d_inner=opt['d_inner'],
                       d_k=opt['d_k'],
                       d_v=opt['d_v'],
                       dropout=opt['dropout'],
                       pre_LN=opt['pre_LN']).cuda()

    model.load_state_dict(ckpt['model'])
    model.eval()

    # data preprocess
    stat_3d = torch.load(os.path.join(opt['data_path'], 'stat_3d.pth.tar'))
    stat_2d = torch.load(os.path.join(opt['data_path'], 'stat_2d.pth.tar'))

    input_length = input.shape[0]
    input_h36m = np.zeros((17, input_length, 2))
    input = input.transpose(1, 0, 2)
    # https://github.com/una-dinosauria/3d-pose-baseline/issues/30
    input_h36m[0] = (input[23] + input[24]) / 2  # Hip
    input_h36m[1] = input[24]  # RHip
    input_h36m[2] = input[26]  # RKnee
    input_h36m[3] = input[32]  # RFoot
    input_h36m[4] = input[23]  # LHip
    input_h36m[5] = input[25]  # LKnee
    input_h36m[6] = input[31]  # LFoot
    input_h36m[7] = (input[11] + input[12] + input[23] + input[24]) / 4  # Spine
    input_h36m[8] = (input[9] + input[10] + input[11] + input[12]) / 4  # Thorax
    input_h36m[9] = input[0]  # Nose
    input_h36m[10] = (input[2] + input[5] / 2)  # head
    input_h36m[11] = input[11]  # LShoulder
    input_h36m[12] = input[13]  # LElbow
    input_h36m[13] = input[15]  # LWrist
    input_h36m[14] = input[12]  # RShoulder
    input_h36m[15] = input[14]  # RElbow
    input_h36m[16] = input[16]  # RWrist
    input_h36m = input_h36m.transpose(1, 0, 2).reshape(-1, 17 * 2, order='c')
    input_h36m = input_h36m - np.tile(input_h36m[:, :2], [1, 17])

    input_mean = np.mean(input_h36m, axis=0)
    input_std = np.std(input_h36m, axis=0)
    print(input_std)
    input_h36m = np.divide(input_h36m - input_mean, input_std, out=np.zeros_like(input_h36m), where=input_std != 0)

    del input
    input_h36m = torch.from_numpy(input_h36m).float().cuda()

    receptive_field = opt['receptive_field']
    receptive_center = math.floor(receptive_field / 2)

    for i in range(input_h36m.shape[0] - receptive_field + 1):
        batch_input = input_h36m[i:i + receptive_field].unsqueeze(dim=0)
        batch_output = model(batch_input)

        show_input = batch_input[0][receptive_center:receptive_center + 1].cpu().detach().numpy()
        show_input = unNormalizeData(show_input, stat_2d['mean'], stat_2d['std'], stat_2d['dim_use']).flatten()

        show_output = batch_output[0].cpu().detach().numpy()
        show_output = unNormalizeData(show_output, stat_3d['mean'], stat_3d['std'], stat_3d['dim_use']).flatten()
        fig = show_fig(show_input, show_output)
        fig.savefig('data/demo/fig{}.png'.format(i))
        del fig
        break


if __name__ == '__main__':
    main()
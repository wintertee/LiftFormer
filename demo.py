import argparse
import os
from pprint import pprint
import torch
import json
from liftformer.Model import Liftformer
from datasets.process import H36M_to_17, unNormalizeData
import numpy as np
import math
from datasets.Human36M import show3Dpose, show2Dpose
import matplotlib.pyplot as plt


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

        self.parser.add_argument('--load', type=str, default='', help='path to load a pretrained checkpoint')

    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self.opt = self.parser.parse_args()
        self._print()
        if not os.path.exists(self.opt.log_path):
            os.mkdir(self.opt.log_path)
        return self.opt


def normalize2d(raw_data, data_mean, data_std):

    dimensions_to_use = np.array(list(H36M_to_17.keys()))
    dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 2, dimensions_to_use * 2 + 1)))

    data_mean = data_mean[dimensions_to_use]
    data_std = data_std[dimensions_to_use]

    return np.divide(raw_data - data_mean, data_std)


def show_fig(input, output):
    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(121)
    ax1.title.set_text('2D input')
    show2Dpose(input, ax1)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.title.set_text('3D pred')
    show3Dpose(output, ax2)
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
    input_h36m = input_h36m.transpose(1, 0, 2).reshape(-1, 17 * 2, order='C')
    input_h36m = normalize2d(input_h36m, stat_2d['mean'], stat_2d['std'])

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
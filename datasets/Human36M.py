from torch.utils.data import Dataset
import torch
import math
import os
import numpy as np

TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

actions = [
    "Directions", "Discussion", "Eating", "Greeting", "Phoning", "Photo", "Posing", "Purchases", "Sitting",
    "SittingDown", "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"
]


class Human36M(Dataset):
    """
    MIT License

    Copyright (c) 2018 weigq.

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
    def __init__(self, actions, data_path, receptive_field, is_train=True, transform=None):
        """
        :param actions: list of actions to use
        :param data_path: path to dataset
        :param is_train: load train/test dataset
        """

        self.actions = actions
        self.data_path = data_path
        self.receptive_field = receptive_field
        self.is_train = is_train
        self.transform = transform

        self.train_inp, self.train_out, self.test_inp, self.test_out = [], [], [], []
        self.train_meta, self.test_meta = [], []

        # loading data
        train_2d_file = 'train_2d.pth.tar'
        test_2d_file = 'test_2d.pth.tar'

        train_3d_file = 'train_3d.pth.tar'
        test_3d_file = 'test_3d.pth.tar'

        if self.is_train:
            # load train data
            self.train_3d = torch.load(os.path.join(os.getcwd(), data_path, train_3d_file))
            self.train_2d = torch.load(os.path.join(os.getcwd(), data_path, train_2d_file))
            for k2d in self.train_2d.keys():
                (sub, act, fname) = k2d
                if act not in self.actions:
                    continue
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                num_f, _ = self.train_2d[k2d].shape
                assert self.train_3d[k3d].shape[0] == self.train_2d[k2d].shape[
                    0], '(training) 3d & 2d shape not matched'
                for i in range(num_f - self.receptive_field + 1):
                    self.train_inp.append(self.train_2d[k2d][i:i + receptive_field])
                    self.train_out.append(self.train_3d[k3d][i + math.floor(receptive_field / 2):i +
                                                             math.ceil(self.receptive_field / 2)])

        else:
            # load test data
            self.test_3d = torch.load(os.path.join(data_path, test_3d_file))
            self.test_2d = torch.load(os.path.join(data_path, test_2d_file))
            for k2d in self.test_2d.keys():
                (sub, act, fname) = k2d
                if act not in self.actions:
                    continue
                k3d = k2d
                k3d = (sub, act, fname[:-3]) if fname.endswith('-sh') else k3d
                num_f, _ = self.test_2d[k2d].shape
                assert self.test_2d[k2d].shape[0] == self.test_3d[k3d].shape[0], '(test) 3d & 2d shape not matched'
                for i in range(num_f - self.receptive_field + 1):
                    self.test_inp.append(self.test_2d[k2d][i:i + receptive_field])
                    self.test_out.append(self.test_3d[k3d][i + math.floor(receptive_field / 2):i +
                                                           math.ceil(self.receptive_field / 2)])

    def __getitem__(self, index):
        if self.is_train:
            inputs = torch.from_numpy(self.train_inp[index]).float()
            outputs = torch.from_numpy(self.train_out[index]).float()

        else:
            inputs = torch.from_numpy(self.test_inp[index]).float()
            outputs = torch.from_numpy(self.test_out[index]).float()

        if self.transform:
            inputs, outputs = self.transform(inputs, outputs)

        return inputs, outputs

    def __len__(self):
        if self.is_train:
            return len(self.train_inp)
        else:
            return len(self.test_inp)


class HorizentalFlip(object):
    def __init__(self, p=0.5):
        self.p = p
        self.order = np.array([0, 4, 5, 6, 1, 2, 3, 7, 8, 9, 10, 14, 15, 16, 11, 12, 13])
        self.order_2d = np.vstack((self.order * 2, self.order * 2 + 1)).flatten(order='F')
        self.order_3d = np.vstack((self.order * 3, self.order * 3 + 1, self.order * 3 + 2)).flatten(order='F')

    def __call__(self, inputs2d, outputs3d):
        if torch.rand(1) < self.p:
            inputs2d = inputs2d[:, self.order_2d]
            outputs3d = outputs3d[:, self.order_3d]
        return inputs2d, outputs3d


def show3Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False, hide_ticks=True):  # blue, orange
    """
    Visualize a 3d skeleton
    Args
      channels: 96x1 vector. The pose to plot.
      ax: matplotlib 3d axis to draw on
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

    assert channels.size == 32 * 3
    vals = np.reshape(channels, (32, -1))

    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 14, 18, 19, 14, 26, 27]) - 1  # start points
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 16, 18, 19, 20, 26, 27, 28]) - 1  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y, z = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(3)]
        ax.plot(x, y, z, lw=2, c=lcolor if LR[i] else rcolor)

    RADIUS = 750  # space around the subject
    xroot, yroot, zroot = vals[0, 0], vals[0, 1], vals[0, 2]
    ax.set_xlim3d([-RADIUS + xroot, RADIUS + xroot])
    ax.set_zlim3d([-RADIUS + zroot, RADIUS + zroot])
    ax.set_ylim3d([-RADIUS + yroot, RADIUS + yroot])

    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    if hide_ticks:
        # Get rid of the ticks and tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])
        ax.set_zticklabels([])

    # Get rid of the panes (actually, make them white)
    white = (1.0, 1.0, 1.0, 0.0)
    ax.w_xaxis.set_pane_color(white)
    ax.w_yaxis.set_pane_color(white)
    # Keep z pane

    # Get rid of the lines in 3d
    ax.w_xaxis.line.set_color(white)
    ax.w_yaxis.line.set_color(white)
    ax.w_zaxis.line.set_color(white)


def show2Dpose(channels, ax, lcolor="#3498db", rcolor="#e74c3c", add_labels=False, hide_ticks=True):
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

    assert channels.size == 32 * 2
    vals = np.reshape(channels, (32, -1))

    I = np.array([1, 2, 3, 1, 7, 8, 1, 13, 14, 14, 18, 19, 14, 26, 27]) - 1  # start points
    J = np.array([2, 3, 4, 7, 8, 9, 13, 14, 16, 18, 19, 20, 26, 27, 28]) - 1  # end points
    LR = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=bool)

    # Make connection matrix
    for i in np.arange(len(I)):
        x, y = [np.array([vals[I[i], j], vals[J[i], j]]) for j in range(2)]
        ax.plot(x, y, lw=2, c=lcolor if LR[i] else rcolor)

    if hide_ticks:
        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])

        # Get rid of tick labels
        ax.get_xaxis().set_ticklabels([])
        ax.get_yaxis().set_ticklabels([])

    RADIUS = 350  # space around the subject
    xroot, yroot = vals[0, 0], vals[0, 1]
    ax.set_xlim([-RADIUS + xroot, RADIUS + xroot])
    ax.set_ylim([-RADIUS + yroot, RADIUS + yroot])
    if add_labels:
        ax.set_xlabel("x")
        ax.set_ylabel("z")

    ax.set_aspect('equal')

"""
Utility functions for dealing with human3.6m data.
Original version: https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/data_utils.py
Adapted from: https://github.com/weigq/3d_pose_baseline_pytorch/issues/22#issuecomment-751764295

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

from __future__ import division
import torch
import copy
import glob
import cdflib
from . import camera

import os
import numpy as np

np.seterr(all='raise')

TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'  # 0
H36M_NAMES[1] = 'RHip'  # 1
H36M_NAMES[2] = 'RKnee'  # 2
H36M_NAMES[3] = 'RFoot'  # 3
H36M_NAMES[6] = 'LHip'  # 4
H36M_NAMES[7] = 'LKnee'  # 5
H36M_NAMES[8] = 'LFoot'  # 6
H36M_NAMES[12] = 'Spine'  # 7
H36M_NAMES[13] = 'Thorax'  # 8
H36M_NAMES[14] = 'Neck/Nose'  # 9
H36M_NAMES[15] = 'Head'  # 10
H36M_NAMES[17] = 'LShoulder'  # 11
H36M_NAMES[18] = 'LElbow'  # 12
H36M_NAMES[19] = 'LWrist'  # 13
H36M_NAMES[25] = 'RShoulder'  # 14
H36M_NAMES[26] = 'RElbow'  # 15
H36M_NAMES[27] = 'RWrist'  # 16

H36M_to_17 = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    6: 4,
    7: 5,
    8: 6,
    12: 7,
    13: 8,
    14: 9,
    15: 10,
    17: 11,
    18: 12,
    19: 13,
    25: 14,
    26: 15,
    27: 16
}

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = [''] * 16
SH_NAMES[0] = 'RFoot'
SH_NAMES[1] = 'RKnee'
SH_NAMES[2] = 'RHip'
SH_NAMES[3] = 'LHip'
SH_NAMES[4] = 'LKnee'
SH_NAMES[5] = 'LFoot'
SH_NAMES[6] = 'Hip'
SH_NAMES[7] = 'Spine'
SH_NAMES[8] = 'Thorax'
SH_NAMES[9] = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'


def main():
    data_dir = './data/h36m/'
    actions = define_actions('all')
    SUBJECT_IDS = [1, 5, 6, 7, 8, 9, 11]
    rcams = camera.load_cameras("./data/h36m/metadata.xml", SUBJECT_IDS)
    camera_frame = True  # boolean. Whether to convert the data to camera coordinates

    # 2D ground truth
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use = \
        create_2d_data(actions, data_dir, rcams)

    stat_2d = {'mean': data_mean, 'std': data_std, 'dim_use': dim_to_use, 'dim_ignore': dim_to_ignore}
    torch.save(stat_2d, './data/h36m/stat_2d.pth.tar')
    torch.save(train_set, './data/h36m/train_2d.pth.tar')
    torch.save(test_set, './data/h36m/test_2d.pth.tar')

    # 3D ground truth
    train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, \
        train_root_positions, test_root_positions = \
        read_3d_data(actions, data_dir, camera_frame, rcams)

    stat_3d = {'mean': data_mean, 'std': data_std, 'dim_use': dim_to_use, 'dim_ignore': dim_to_ignore}
    torch.save(stat_3d, './data/h36m/stat_3d.pth.tar')
    torch.save(train_set, './data/h36m/train_3d.pth.tar')
    torch.save(test_set, './data/h36m/test_3d.pth.tar')


def load_data(bpath, subjects, actions, dim=3):
    """Loads 2d ground truth from disk, and puts it in an easy-to-acess dictionary

  Args
    bpath: String. Path where to load the data from
    subjects: List of integers. Subjects whose data will be loaded
    actions: List of strings. The actions to load
    dim: Integer={2,3}. Load 2 or 3-dimensional data
  Returns:
    data: Dictionary with keys k=(subject, action, seqname)
      values v=(nx(32*2) matrix of 2d ground truth)
      There will be 2 entries per subject/action if loading 3d data
      There will be 8 entries per subject/action if loading 2d data
  """

    if dim not in [2, 3]:
        raise ValueError('dim must be 2 or 3')

    data = {}

    for subj in subjects:
        for action in actions:

            print('Reading subject {0}, action {1}'.format(subj, action))

            dpath = os.path.join(bpath, 'S{0}'.format(subj), 'MyPoseFeatures/D{0}_Positions'.format(dim),
                                 '{0}*.cdf'.format(action))
            print(dpath)

            fnames = glob.glob(dpath)

            loaded_seqs = 0
            for fname in fnames:
                seqname = os.path.basename(fname)

                # This rule makes sure SittingDown is not loaded when Sitting is requested
                if action == "Sitting" and seqname.startswith("SittingDown"):
                    continue

                # This rule makes sure that WalkDog and WalkTogeter are not loaded when
                # Walking is requested.
                if seqname.startswith(action):
                    print(fname)
                    loaded_seqs = loaded_seqs + 1

                    cdf_file = cdflib.CDF(fname)
                    poses = cdf_file.varget("Pose").squeeze()
                    cdf_file.close()

                    data[(subj, action, seqname)] = poses

            if dim == 2:
                assert loaded_seqs == 8, "Expecting 8 sequences, found {0} instead".format(loaded_seqs)
            else:
                assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format(loaded_seqs)

    return data


def normalization_stats(complete_data, dim):
    """Computes normalization statistics: mean and stdev, dimensions used and ignored

  Args
    complete_data: nxd np array with poses
    dim. integer={2,3} dimensionality of the data
    predict_14. boolean. Whether to use only 14 joints
  Returns
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dimensions_to_ignore: list of dimensions not used in the model
    dimensions_to_use: list of dimensions used in the model
  """
    if dim not in [2, 3]:
        raise ValueError('dim must be 2 or 3')

    data_mean = np.mean(complete_data, axis=0)
    data_std = np.std(complete_data, axis=0)

    # Encodes which 17 (or 14) 2d-3d pairs we are predicting
    dimensions_to_ignore = []
    if dim == 2:
        dimensions_to_use = np.array(list(H36M_to_17.keys()))
        dimensions_to_use = np.sort(np.hstack((dimensions_to_use * 2, dimensions_to_use * 2 + 1)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 2), dimensions_to_use)
    else:  # dim == 3
        dimensions_to_use = np.array(list(H36M_to_17.keys()))

        dimensions_to_use = np.sort(
            np.hstack((dimensions_to_use * 3, dimensions_to_use * 3 + 1, dimensions_to_use * 3 + 2)))
        dimensions_to_ignore = np.delete(np.arange(len(H36M_NAMES) * 3), dimensions_to_use)

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def transform_world_to_camera(poses_set, cams, ncams=4):
    """Project 3d poses from world coordinate to camera coordinate system

    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with camera
      ncams: number of camera per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted(poses_set.keys()):

        subj, action, seqname = t3dk
        t3d_world = poses_set[t3dk]

        for c in range(ncams):
            R, T, _, _, _, _, name = cams[(subj, c + 1)]
            camera_coord = camera.world_to_camera_frame(np.reshape(t3d_world, [-1, 3]), R, T)
            camera_coord = np.reshape(camera_coord, [-1, len(H36M_NAMES) * 3])

            sname = seqname[:-3] + name + ".h5"  # e.g.: Waiting 1.58860488.h5
            t3d_camera[(subj, action, sname)] = camera_coord

    return t3d_camera


def normalize_data(data, data_mean, data_std, dim_to_use):
    """Normalizes a dictionary of poses

  Args
    data: dictionary where values are
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
    dim_to_use: list of dimensions to keep in the data
  Returns
    data_out: dictionary with same keys as data, but values have been normalized
  """
    data_out = {}

    for key in data.keys():
        data[key] = data[key][:, dim_to_use]
        mu = data_mean[dim_to_use]
        stddev = data_std[dim_to_use]
        data_out[key] = np.divide((data[key] - mu), stddev, out=np.zeros_like(data[key]), where=stddev != 0)

    return data_out


def unNormalizeData(normalized_data, data_mean, data_std, dimensions_to_use):
    """Un-normalizes a matrix whose mean has been substracted and that has been divided by
  standard deviation. Some dimensions might also be missing

  Args
    normalized_data: nxd matrix to unnormalize
    data_mean: np vector with the mean of the data
    data_std: np vector with the standard deviation of the data
  Returns
    orig_data: the input normalized_data, but unnormalized

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


def define_actions(action):
    """Given an action string, returns a list of corresponding actions.

  Args
    action: String. either "all" or one of the h36m actions
  Returns
    actions: List of strings. Actions to use.
  Raises
    ValueError: if the action is not a valid action in Human 3.6M
  """
    actions = [
        "Directions", "Discussion", "Eating", "Greeting", "Phoning", "Photo", "Posing", "Purchases", "Sitting",
        "SittingDown", "Smoking", "Waiting", "WalkDog", "Walking", "WalkTogether"
    ]

    if action == "All" or action == "all":
        return actions

    if action not in actions:
        raise ValueError("Unrecognized action: %s" % action)

    return [action]


def project_to_camera(poses_set, cams, ncams=4):
    """
  Project 3d poses using camera parameters

  Args
    poses_set: dictionary with 3d poses
    cams: dictionary with camera parameters
    ncams: number of camera per subject
  Returns
    t2d: dictionary with 2d poses
  """
    t2d = {}

    for t3dk in sorted(poses_set.keys()):
        subj, a, seqname = t3dk
        t3d = poses_set[t3dk]

        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, cam + 1)]
            pts2d, _, _, _, _ = camera.project_point_radial(np.reshape(t3d, [-1, 3]), R, T, f, c, k, p)

            pts2d = np.reshape(pts2d, [-1, len(H36M_NAMES) * 2])
            sname = seqname[:-3] + name + ".h5"  # e.g.: Waiting 1.58860488.h5
            t2d[(subj, a, sname)] = pts2d

    return t2d


def create_2d_data(actions, data_dir, rcams):
    """Creates 2d poses by projecting 3d poses with the corresponding camera
  parameters. Also normalizes the 2d poses

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    rcams: dictionary with camera parameters
  Returns
    train_set: dictionary with projected 2d poses for training
    test_set: dictionary with projected 2d poses for testing
    data_mean: vector with the mean of the 2d training data
    data_std: vector with the standard deviation of the 2d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
  """

    # Load 3d data
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)

    # Create 2d data by projecting with camera parameters
    train_set = project_to_camera(train_set, rcams)
    test_set = project_to_camera(test_set, rcams)

    # Compute normalization statistics.
    complete_train = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(complete_train, dim=2)

    # Divide every dimension independently
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use


def read_3d_data(actions, data_dir, camera_frame, rcams):
    """Loads 3d poses, zero-centres and normalizes them

  Args
    actions: list of strings. Actions to load
    data_dir: string. Directory where the data can be loaded from
    camera_frame: boolean. Whether to convert the data to camera coordinates
    rcams: dictionary with camera parameters
    predict_14: boolean. Whether to predict only 14 joints
  Returns
    train_set: dictionary with loaded 3d poses for training
    test_set: dictionary with loaded 3d poses for testing
    data_mean: vector with the mean of the 3d training data
    data_std: vector with the standard deviation of the 3d training data
    dim_to_ignore: list with the dimensions to not predict
    dim_to_use: list with the dimensions to predict
    train_root_positions: dictionary with the 3d positions of the root in train
    test_root_positions: dictionary with the 3d positions of the root in test
  """
    # Load 3d data
    train_set = load_data(data_dir, TRAIN_SUBJECTS, actions, dim=3)
    test_set = load_data(data_dir, TEST_SUBJECTS, actions, dim=3)

    if camera_frame:
        train_set = transform_world_to_camera(train_set, rcams)
        test_set = transform_world_to_camera(test_set, rcams)

    # Apply 3d post-processing (centering around root)
    train_set, train_root_positions = postprocess_3d(train_set)
    test_set, test_root_positions = postprocess_3d(test_set)

    # Compute normalization statistics
    complete_train = copy.deepcopy(np.vstack(list(train_set.values())))
    data_mean, data_std, dim_to_ignore, dim_to_use = normalization_stats(complete_train, dim=3)

    # Divide every dimension independently
    train_set = normalize_data(train_set, data_mean, data_std, dim_to_use)
    test_set = normalize_data(test_set, data_mean, data_std, dim_to_use)

    return train_set, test_set, data_mean, data_std, dim_to_ignore, dim_to_use, train_root_positions, test_root_positions


def postprocess_3d(poses_set):
    """Center 3d points around root

  Args
    poses_set: dictionary with 3d data
  Returns
    poses_set: dictionary with 3d data centred around root (center hip) joint
    root_positions: dictionary with the original 3d position of each pose
  """
    root_positions = {}
    for k in poses_set.keys():
        # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:, :3])

        # Remove the root from the 3d position
        poses = poses_set[k]
        poses = poses - np.tile(poses[:, :3], [1, len(H36M_NAMES)])
        poses_set[k] = poses

    return poses_set, root_positions


if __name__ == "__main__":
    main()

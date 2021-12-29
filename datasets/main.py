from . import process


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
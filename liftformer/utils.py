import os
import json


def save_options(opt, path):
    file_path = os.path.join(path, 'opt.json')
    with open(file_path, 'w+') as f:
        f.write(json.dumps(vars(opt), sort_keys=True, indent=4))


class AverageMeter(object):
    def __init__(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n

    def avg(self):
        return self.sum / self.count

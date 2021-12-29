import argparse
import os
from pprint import pprint


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--data_path', type=str, default='data/h36m', help='path to dataset')
        self.parser.add_argument('--log_path', type=str, default='runs/default/')
        self.parser.add_argument('--load', type=str, default='', help='path to load a pretrained checkpoint')

        self.parser.add_argument('--test', dest='test', action='store_true', help='test')

        # ===============================================================
        #                     Model options
        # ===============================================================

        self.parser.add_argument('--d_model', type=int, default=512)
        self.parser.add_argument('--receptive_field', type=int, default=27)
        self.parser.add_argument('--n_layers', type=int, default=6)
        self.parser.add_argument('--n_head', type=int, default=8)
        self.parser.add_argument('--d_in', type=int, default=34)
        self.parser.add_argument('--d_out', type=int, default=51)
        self.parser.add_argument('--d_inner', type=int, default=2048)
        self.parser.add_argument('--d_k', type=int, default=64)
        self.parser.add_argument('--d_v', type=int, default=64)
        self.parser.add_argument('--dropout', type=float, default=0.1)

        self.parser.add_argument('--pre_LN', dest='pre_LN', action='store_true')
        self.parser.add_argument('--no-pre_LN', dest='pre_LN', action='store_false')
        self.parser.set_defaults(pre_LN=True)
        self.parser.add_argument('--use_noam', dest='use_noam', action='store_true')
        self.parser.add_argument('--no-use_noam', dest='use_noam', action='store_false')
        self.parser.set_defaults(use_noam=False)

        # ===============================================================
        #                     Running options
        # ===============================================================

        self.parser.add_argument('--lr', type=float, default=1.0e-3)
        self.parser.add_argument('--epochs', type=int, default=80)
        self.parser.add_argument('--train_batchsize', type=int, default=5120)
        self.parser.add_argument('--test_batchsize', type=int, default=5120)
        self.parser.add_argument('--num_workers', type=int, default=6, help='# subprocesses to use for data loading')
        self.parser.add_argument('--pin_memory', dest='pin_memory', action='store_true')
        self.parser.add_argument('--no-pin_memory', dest='pin_memory', action='store_false')
        self.parser.set_defaults(pin_memory=True)
        self.parser.add_argument('--log_img_freq', type=int, default=250)

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

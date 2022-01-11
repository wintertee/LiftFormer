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
        self.parser.add_argument(
            '--data_path', type=str, default='data/h36m', help='path to dataset')
        self.parser.add_argument(
            '--log_path', type=str, default='runs/default/')
        self.parser.add_argument(
            '--load', type=str, default='', help='path to load a pretrained checkpoint')

        self.parser.add_argument(
            '--test', dest='test', action='store_true', help='test')

        # ===============================================================
        #                     Model options
        # ===============================================================

        self.parser.add_argument(
            '--d_model', type=int, default=512, help="demension in tranformer model")
        self.parser.add_argument(
            '--receptive_field', type=int, default=27, help="receptive field of network")
        self.parser.add_argument(
            '--n_layers', type=int, default=6, help="number of transformer encoder")
        self.parser.add_argument(
            '--n_head', type=int, default=8, help="number of head in attention module")
        self.parser.add_argument('--d_in', type=int, default=34,
                                 help="demension of input, should be 2 * number of joints")
        self.parser.add_argument('--d_out', type=int, default=51,
                                 help="demension of output, should be 3 * number of joints")
        self.parser.add_argument('--d_inner', type=int, default=2048,
                                 help="demension of feed-forward network module")
        self.parser.add_argument('--d_k', type=int, default=64,
                                 help="demension of key matrix")
        self.parser.add_argument('--d_v', type=int, default=64,
                                 help="demension of value matrix")
        self.parser.add_argument('--dropout', type=float, default=0.1,
                                 help="probability of dropout")

        self.parser.add_argument(
            '--pre_LN', dest='pre_LN', action='store_true',
            help="use pre layer normalisation (default)")
        self.parser.add_argument(
            '--no-pre_LN', dest='pre_LN', action='store_false',
            help="not use pre layer normalisation")
        self.parser.set_defaults(pre_LN=True)

        self.parser.add_argument(
            '--use_noam', dest='use_noam', action='store_true',
            help="use Noam learning rate warm up")
        self.parser.add_argument(
            '--no-use_noam', dest='use_noam', action='store_false',
            help="not use Noam learning rate warm up (default)")
        self.parser.set_defaults(use_noam=False)

        self.parser.add_argument(
            '--ReduceLROnPlateau', dest='ReduceLROnPlateau', action='store_true',
            help="use ReduceLROnPlateau learning rate scheduler (default)")
        self.parser.add_argument(
            '--no-ReduceLROnPlateau', dest='ReduceLROnPlateau', action='store_false',
            help="not use ReduceLROnPlateau learning rate scheduler")
        self.parser.set_defaults(ReduceLROnPlateau=True)

        # ===============================================================
        #                     Running options
        # ===============================================================

        self.parser.add_argument('--lr', type=float, default=1.0e-3,
                                 help="learning rate")
        self.parser.add_argument('--epochs', type=int, default=80,
                                 help="number of epochs")
        self.parser.add_argument('--train_batchsize', type=int, default=5120,
                                 help="batch size used for train")
        self.parser.add_argument('--test_batchsize', type=int, default=5120,
                                 help="batch size for used test")
        self.parser.add_argument(
            '--num_workers', type=int, default=6, help='subprocesses to use for data loading')
        self.parser.add_argument(
            '--pin_memory', dest='pin_memory', action='store_true', help="use pin_memory in DataLoader (default)")
        self.parser.add_argument(
            '--no-pin_memory', dest='pin_memory', action='store_false', help="not use pin_memory in DataLoader")
        self.parser.set_defaults(pin_memory=True)
        self.parser.add_argument('--log_img_freq', type=int, default=250,
                                 help="frequency for logging image in tensorboard (steps)")

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

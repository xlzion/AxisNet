import datetime
import argparse
import random
import numpy as np
import torch


class AxisNetOptions:
    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description="AxisNet refactor options")
        parser.add_argument("--train", default=1, type=int, help="train(default) or evaluate")
        parser.add_argument("--use_cpu", action="store_true", help="use cpu?")

        parser.add_argument("--hgc", type=int, default=16, help="hidden units of gconv layer")
        parser.add_argument("--lg", type=int, default=4, help="number of gconv layers")
        parser.add_argument("--lr", default=0.01, type=float, help="initial learning rate")
        parser.add_argument("--wd", default=5e-5, type=float, help="weight decay")
        parser.add_argument("--num_iter", default=300, type=int, help="number of epochs for training")
        parser.add_argument("--edropout", type=float, default=0.3, help="edge dropout rate")
        parser.add_argument("--dropout", default=0.2, type=float, help="ratio of dropout")
        parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
        parser.add_argument("--ckpt_path", type=str, default="./save_models/axisnet", help="checkpoint path")

        parser.add_argument("--use_multimodal", action="store_true", help="enable microbiome data")
        parser.add_argument("--microbiome_path", type=str, default=None, help="path to microbiome data")
        parser.add_argument("--contrastive_weight", type=float, default=0.5, help="contrastive loss weight")
        parser.add_argument("--microbiome_dim", type=int, default=2503, help="microbiome feature dim")
        parser.add_argument("--microbiome_reg_weight", type=float, default=0.05, help="graph consistency loss")
        parser.add_argument("--microbiome_warmup_epochs", type=int, default=10, help="warmup epochs")
        parser.add_argument("--microbiome_top_k", type=int, default=5, help="top-k pseudo-pairing")
        parser.add_argument("--microbiome_pca_dim", type=int, default=64, help="PCA dimension")
        parser.add_argument("--drop_age", action="store_true", help="drop age from phenotypes")
        parser.add_argument("--drop_sex", action="store_true", help="drop sex from phenotypes")
        parser.add_argument("--seed", type=int, default=123, help="random seed")
        parser.add_argument("--model_type", type=str, default="enhanced",
                            choices=["enhanced", "transformer", "gcn_transformer"],
                            help="model choice")

        args = parser.parse_args(argv)
        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device("cpu")
        else:
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(" Using GPU in torch")

        self.args = args

    def print_args(self):
        print("==========       CONFIG      =============")
        for arg, content in self.args.__dict__.items():
            print("{}:{}".format(arg, content))
        print("==========     CONFIG END    =============")
        print("\n")
        phase = "train" if self.args.train == 1 else "eval"
        print("===> Phase is {}.".format(phase))

    def initialize(self):
        self.set_seed(self.args.seed)
        self.print_args()
        return self.args

    def set_seed(self, seed=0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

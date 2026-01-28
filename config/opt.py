import datetime
import argparse
import random
import numpy as np
import torch


class AxisNetOptions:
    def __init__(self, argv=None):
        parser = argparse.ArgumentParser(description="AxisNet: Multimodal Microbiome-Brain Network Fusion")

        # --- Execution Mode ---
        mode_group = parser.add_argument_group("Execution Mode")
        mode_group.add_argument("--mode", type=str, default="train", choices=["train", "eval"],
                                help="Execution mode: 'train' for model training, 'eval' for evaluation")
        mode_group.add_argument("--seed", type=int, default=123, help="Random seed for reproducibility")
        mode_group.add_argument("--use_cpu", action="store_true", help="Force CPU usage even if GPU is available")

        # --- Model Architecture ---
        arch_group = parser.add_argument_group("Model Architecture")
        arch_group.add_argument("--model_type", type=str, default="enhanced",
                                choices=["enhanced", "transformer", "gcn_transformer"],
                                help="Architecture variant to use")
        arch_group.add_argument("--hidden_dim", type=int, default=16, help="Hidden units in graph convolution layers")
        arch_group.add_argument("--num_layers", type=int, default=4, help="Number of graph convolution layers")
        arch_group.add_argument("--dropout", default=0.2, type=float, help="Node feature dropout rate")
        arch_group.add_argument("--edge_dropout", type=float, default=0.3, help="Edge dropout rate (variational)")
        arch_group.add_argument("--num_classes", type=int, default=2, help="Number of output classification classes")

        # --- Training Hyperparameters ---
        train_group = parser.add_argument_group("Training Hyperparameters")
        train_group.add_argument("--lr", default=0.01, type=float, help="Initial learning rate")
        train_group.add_argument("--weight_decay", default=5e-5, type=float, help="Weight decay (L2 penalty)")
        train_group.add_argument("--epochs", default=300, type=int, help="Maximum number of training epochs")
        train_group.add_argument("--ckpt_path", type=str, default="./save_models/axisnet",
                                help="Directory to save/load model checkpoints")

        # --- Multimodal & Microbiome ---
        multi_group = parser.add_argument_group("Multimodal Integration")
        multi_group.add_argument("--use_multimodal", action="store_true", help="Enable microbiome-brain fusion")
        multi_group.add_argument("--microbiome_path", type=str, default=None, help="Path to microbiome data file")
        multi_group.add_argument("--microbiome_pca_dim", type=int, default=64, help="Dimension of PCA-reduced microbiome features")
        multi_group.add_argument("--microbiome_top_k", type=int, default=5, help="Top-K neighbors for pseudo-pairing")
        multi_group.add_argument("--contrastive_weight", type=float, default=0.5, help="Weight for cross-modal contrastive loss")
        multi_group.add_argument("--consistency_weight", type=float, default=0.05, help="Weight for graph consistency regularization")
        multi_group.add_argument("--warmup_epochs", type=int, default=10, help="Epochs before starting graph consistency")
        multi_group.add_argument("--drop_age", action="store_true", help="Exclude age from phenotypic features")
        multi_group.add_argument("--drop_sex", action="store_true", help="Exclude sex from phenotypic features")

        args = parser.parse_args(argv)
        args.time = datetime.datetime.now().strftime("%y%m%d")

        if args.use_cpu:
            args.device = torch.device("cpu")
        else:
            args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.args = args

    def print_args(self):
        print("\n" + "="*20 + " AxisNet Configuration " + "="*20)
        for arg, content in self.args.__dict__.items():
            if arg not in ['device']: # Skip device object
                print(f"{arg: <25}: {content}")
        print("="*63 + "\n")
        print(f"===> Phase: {self.args.mode.upper()}")

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

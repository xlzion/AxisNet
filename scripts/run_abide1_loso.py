#!/usr/bin/env python3
"""
Run AxisNet (GCN+Transformer) on ABIDE I with LOSO (Leave-One-Site-Out).

ABIDE I data paths are the repository defaults in `data/abide_parser.py`,
so you typically don't need to pass --data_folder/--phenotype_path.

Examples:

  # Unimodal (fMRI only), LOSO
  python scripts/run_abide1_loso.py

  # Multimodal, LOSO
  python scripts/run_abide1_loso.py --use_multimodal --microbiome_path data/feature-table.biom
"""

import argparse

from AxisNet_refactor.config.opt import AxisNetOptions
from AxisNet_refactor.scripts.train_eval import run_cv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ABIDE I with LOSO (one site held out per fold).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    parser.add_argument("--ckpt_path", type=str, default="./save_models/axisnet_abide1_loso")
    parser.add_argument("--use_multimodal", action="store_true")
    parser.add_argument("--microbiome_path", type=str, default=None)
    parser.add_argument("--drop_age", action="store_true")
    parser.add_argument("--drop_sex", action="store_true")
    args = parser.parse_args()

    argv = [
        "--cv_type", "loso",
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--mode", args.mode,
        "--ckpt_path", args.ckpt_path,
    ]
    if args.use_multimodal:
        argv.append("--use_multimodal")
        if args.microbiome_path:
            argv.extend(["--microbiome_path", args.microbiome_path])
    if args.drop_age:
        argv.append("--drop_age")
    if args.drop_sex:
        argv.append("--drop_sex")

    opt = AxisNetOptions(argv=argv).initialize()
    run_cv(opt)


if __name__ == "__main__":
    main()


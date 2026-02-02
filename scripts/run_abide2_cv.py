#!/usr/bin/env python3
"""
Run GCN+Transformer on ABIDE II with cross-validation or LOSO (Leave-One-Site-Out).

Examples:

  # ABIDE II, 10-fold stratified CV (default)
  python scripts/run_abide2_cv.py \
    --data_folder /path/to/ABIDE_II/cpac/filt_noglobal \
    --phenotype_path /path/to/ABIDE_II_phenotype.csv

  # ABIDE II, Leave-One-Site-Out (one site as test each fold)
  python scripts/run_abide2_cv.py \
    --data_folder /path/to/ABIDE_II/cpac/filt_noglobal \
    --phenotype_path /path/to/ABIDE_II_phenotype.csv \
    --cv_type loso

  # With custom subject list and multimodal
  python scripts/run_abide2_cv.py \
    --data_folder /path/to/ABIDE_II/cpac/filt_noglobal \
    --phenotype_path /path/to/ABIDE_II_phenotype.csv \
    --subject_ids_path /path/to/ABIDE_II_subject_ids.txt \
    --cv_type loso \
    --use_multimodal
"""

import argparse
import sys

from AxisNet_refactor.config.opt import AxisNetOptions
from AxisNet_refactor.scripts.train_eval import run_cv


def main():
    parser = argparse.ArgumentParser(
        description="Run AxisNet (GCN+Transformer) on ABIDE II with CV or LOSO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_folder",
        type=str,
        required=True,
        help="Root folder for connectivity data (e.g. ABIDE II cpac/filt_noglobal)",
    )
    parser.add_argument(
        "--phenotype_path",
        type=str,
        required=True,
        help="Path to phenotype CSV (columns: SUB_ID, FILE_ID, SITE_ID, DX_GROUP, AGE_AT_SCAN, SEX)",
    )
    parser.add_argument(
        "--subject_ids_path",
        type=str,
        default=None,
        help="Subject IDs file (one per line). Default: <data_folder>/subject_IDs.txt",
    )
    parser.add_argument(
        "--cv_type",
        type=str,
        default="stratified_kfold",
        choices=["stratified_kfold", "loso"],
        help="stratified_kfold = K-fold CV; loso = Leave-One-Site-Out",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=10,
        help="Number of folds when cv_type=stratified_kfold",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--use_multimodal", action="store_true")
    parser.add_argument("--microbiome_path", type=str, default=None)
    parser.add_argument("--drop_age", action="store_true")
    parser.add_argument("--drop_sex", action="store_true")
    parser.add_argument("--ckpt_path", type=str, default="./save_models/axisnet_abide2")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval"])
    args = parser.parse_args()

    argv = [
        "--data_folder", args.data_folder,
        "--phenotype_path", args.phenotype_path,
        "--cv_type", args.cv_type,
        "--n_folds", str(args.n_folds),
        "--seed", str(args.seed),
        "--epochs", str(args.epochs),
        "--ckpt_path", args.ckpt_path,
        "--mode", args.mode,
    ]
    if args.subject_ids_path:
        argv.extend(["--subject_ids_path", args.subject_ids_path])
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

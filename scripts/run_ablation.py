#!/usr/bin/env python3
"""
Run GCN+Transformer ablation experiments and save CSV.

Runs unimodal/multimodal for variants: full, drop_age, drop_sex, drop_age_sex.
Output: result0_gcn_transformer.csv (or --out_prefix_gcn_transformer.csv).
Default seed: 123.
"""

import argparse
import csv

from AxisNet_refactor.config.opt import AxisNetOptions
from AxisNet_refactor.scripts.train_eval import run_cv

MODEL_TYPE = "gcn_transformer"


def build_argv(seed, use_multimodal, microbiome_path, drop_age, drop_sex):
    argv = ["--seed", str(seed), "--model_type", MODEL_TYPE]
    if use_multimodal:
        argv.append("--use_multimodal")
        if microbiome_path:
            argv.extend(["--microbiome_path", microbiome_path])
    if drop_age:
        argv.append("--drop_age")
    if drop_sex:
        argv.append("--drop_sex")
    return argv


def run_one(seed, use_multimodal, microbiome_path, drop_age, drop_sex):
    argv = build_argv(seed, use_multimodal, microbiome_path, drop_age, drop_sex)
    opt = AxisNetOptions(argv=argv).initialize()
    return run_cv(opt)


def main():
    parser = argparse.ArgumentParser(description="Run targeted ablation experiments (age/sex drop) and save CSV")
    parser.add_argument(
        "--microbiome_path",
        type=str,
        default="data/feature-table.biom",
        help="path to microbiome data (for multimodal runs)",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="",
        help="comma-separated seeds; if empty, use the default seed 123 (same as AxisNetOptions)",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="result0",
        help="prefix for output CSV (default: result0 -> result0_gcn_transformer.csv)",
    )
    args = parser.parse_args()

    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        seeds = [123]

    variant_configs = [
        ("full", False, False),
        ("drop_age", True, False),
        ("drop_sex", False, True),
        ("drop_age_sex", True, True),
    ]

    fieldnames = [
        "group", "modality", "variant", "seed", "drop_age", "drop_sex",
        "acc_mean", "acc_std", "auc_mean", "auc_std",
        "se_mean", "se_std", "sp_mean", "sp_std", "f1_mean", "f1_std",
    ]

    rows = []
    print(f"\n{'='*60}\nRunning GCN+Transformer ablation\n{'='*60}\n")

    for seed in seeds:
        for variant_name, d_age, d_sex in variant_configs:
            print(f"  Running: gcn_transformer | multimodal | {variant_name} | seed={seed}")
            metrics_mm = run_one(
                seed, use_multimodal=True, microbiome_path=args.microbiome_path,
                drop_age=d_age, drop_sex=d_sex,
            )
            rows.append({
                "group": f"ablation_gcn_transformer_mm",
                "modality": "multimodal",
                "variant": variant_name,
                "seed": seed,
                "drop_age": d_age,
                "drop_sex": d_sex,
                **metrics_mm,
            })

            print(f"  Running: gcn_transformer | unimodal | {variant_name} | seed={seed}")
            metrics_uni = run_one(
                seed, use_multimodal=False, microbiome_path=args.microbiome_path,
                drop_age=d_age, drop_sex=d_sex,
            )
            rows.append({
                "group": f"ablation_gcn_transformer_uni",
                "modality": "unimodal",
                "variant": variant_name,
                "seed": seed,
                "drop_age": d_age,
                "drop_sex": d_sex,
                **metrics_uni,
            })

    out_csv = f"{args.out_prefix}_gcn_transformer.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved GCN+Transformer ablation results to {out_csv}\n{'='*60}")


if __name__ == "__main__":
    main()


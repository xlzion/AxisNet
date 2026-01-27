#!/usr/bin/env python3
"""
Run ablation experiments for all three model types and save separate CSV files.

For each model type (enhanced, transformer, gcn_transformer), runs:
- unimodal (fMRI only), full (use all phenotypes)
- unimodal (fMRI only), drop_age
- unimodal (fMRI only), drop_sex
- multimodal (fMRI + microbiome), full (use all phenotypes)
- multimodal (fMRI + microbiome), drop_age
- multimodal (fMRI + microbiome), drop_sex

Output CSV files:
- result0_enhanced.csv
- result0_transformer.csv
- result0_gcn_transformer.csv

Random seed:
- By default uses seed=123 (same as AxisNetOptions default), so we do
  **not change** the random seed behavior.
- You can optionally override with --seeds, but if you don't pass it,
  the original/default seed is kept.
"""

import argparse
import csv

from AxisNet_refactor.config.opt import AxisNetOptions
from AxisNet_refactor.scripts.train_eval import run_cv


def build_argv(seed, use_multimodal, microbiome_path, drop_age, drop_sex, model_type="enhanced"):
    argv = ["--seed", str(seed), "--model_type", model_type]
    if use_multimodal:
        argv.append("--use_multimodal")
        if microbiome_path:
            argv.extend(["--microbiome_path", microbiome_path])
    if drop_age:
        argv.append("--drop_age")
    if drop_sex:
        argv.append("--drop_sex")
    return argv


def run_one(seed, use_multimodal, microbiome_path, drop_age, drop_sex, model_type="enhanced"):
    argv = build_argv(seed, use_multimodal, microbiome_path, drop_age, drop_sex, model_type)
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
        help="prefix for output CSV filenames (default: result0, will generate result0_enhanced.csv, etc.)",
    )
    args = parser.parse_args()

    # Keep the original/default seed behavior if user does not specify anything:
    # AxisNetOptions default is 123, so we use [123] here.
    if args.seeds.strip():
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    else:
        seeds = [123]

    # (variant_name, drop_age, drop_sex)
    variant_configs = [
        ("full", False, False),
        ("drop_age", True, False),
        ("drop_sex", False, True),
    ]

    # Run experiments for all three model types
    model_types = ["enhanced", "transformer", "gcn_transformer"]

    fieldnames = [
        "group",
        "modality",
        "variant",
        "seed",
        "drop_age",
        "drop_sex",
        "acc_mean",
        "acc_std",
        "auc_mean",
        "auc_std",
        "se_mean",
        "se_std",
        "sp_mean",
        "sp_std",
        "f1_mean",
        "f1_std",
    ]

    for model_type in model_types:
        rows = []
        print(f"\n{'='*60}")
        print(f"Running experiments for model type: {model_type}")
        print(f"{'='*60}\n")

        for seed in seeds:
            for variant_name, d_age, d_sex in variant_configs:
                # Multimodal (use microbiome)
                print(f"  Running: {model_type} | multimodal | {variant_name} | seed={seed}")
                metrics_mm = run_one(
                    seed,
                    use_multimodal=True,
                    microbiome_path=args.microbiome_path,
                    drop_age=d_age,
                    drop_sex=d_sex,
                    model_type=model_type,
                )
                rows.append(
                    {
                        "group": f"ablation_{model_type}_mm",
                        "modality": "multimodal",
                        "variant": variant_name,
                        "seed": seed,
                        "drop_age": d_age,
                        "drop_sex": d_sex,
                        **metrics_mm,
                    }
                )

                # Unimodal (fMRI only)
                print(f"  Running: {model_type} | unimodal | {variant_name} | seed={seed}")
                metrics_uni = run_one(
                    seed,
                    use_multimodal=False,
                    microbiome_path=args.microbiome_path,
                    drop_age=d_age,
                    drop_sex=d_sex,
                    model_type=model_type,
                )
                rows.append(
                    {
                        "group": f"ablation_{model_type}_uni",
                        "modality": "unimodal",
                        "variant": variant_name,
                        "seed": seed,
                        "drop_age": d_age,
                        "drop_sex": d_sex,
                        **metrics_uni,
                    }
                )

        out_csv = f"{args.out_prefix}_{model_type}.csv"
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"\nSaved {model_type} ablation results to {out_csv}")

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()


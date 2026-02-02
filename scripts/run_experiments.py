#!/usr/bin/env python3
import argparse
import csv

from AxisNet_refactor.config.opt import AxisNetOptions
from AxisNet_refactor.scripts.train_eval import run_cv


def build_argv(seed, use_multimodal, microbiome_path, drop_age, drop_sex, model_type="gcn_transformer"):
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


def run_one(seed, use_multimodal, microbiome_path, drop_age, drop_sex, model_type="gcn_transformer"):
    argv = build_argv(seed, use_multimodal, microbiome_path, drop_age, drop_sex, model_type)
    opt = AxisNetOptions(argv=argv).initialize()
    return run_cv(opt)


def run_variant(seed, use_mm, path, d_age, d_sex, variant_name, group_name, rows_list, group_rows_dict, model_type):
    metrics = run_one(seed, use_mm, path, d_age, d_sex, model_type)
    group_key = f"{model_type}_{'mm' if use_mm else 'uni'}_{variant_name}"
    row = {
        "group": f"{group_name}_{model_type}_{'mm' if use_mm else 'uni'}",
        "variant": variant_name,
        "seed": seed,
        "drop_age": d_age,
        "drop_sex": d_sex,
        **metrics,
    }
    rows_list.append(row)
    if group_key not in group_rows_dict:
        group_rows_dict[group_key] = []
    group_rows_dict[group_key].append(row)


def main():
    parser = argparse.ArgumentParser(description="Run CV experiments and save CSV")
    parser.add_argument("--microbiome_path", type=str, default="data/feature-table.biom")
    parser.add_argument("--seeds", type=str, default="", help="comma-separated seeds")
    parser.add_argument("--out_csv", type=str, default="experiment_results.csv")
    args = parser.parse_args()

    if args.seeds.strip():
        seed_groups = [("custom", [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""])]
    else:
        seed_groups = [
            ("seq_0_4", [0, 1, 2, 3, 4]),
            ("classic_42_2022", [42, 123, 2020, 2021, 2022]),
        ]

    rows = []
    variant_configs = [
        ("full", False, False),
        ("drop_age", True, False),
        ("drop_sex", False, True),
        ("drop_age_sex", True, True),
    ]
    model_types = ["gcn_transformer"]

    for group_name, seeds in seed_groups:
        group_rows_dict = {}
        for seed in seeds:
            for v_name, d_age, d_sex in variant_configs:
                for model_type in model_types:
                    run_variant(seed, True, args.microbiome_path, d_age, d_sex, v_name, group_name, rows, group_rows_dict, model_type)
                    run_variant(seed, False, args.microbiome_path, d_age, d_sex, v_name, group_name, rows, group_rows_dict, model_type)

        for key, g_rows in group_rows_dict.items():
            if not g_rows:
                continue
            parts = key.split("_")
            m_type = parts[0]
            is_mm = parts[1] == "mm"
            v_name = "_".join(parts[2:])
            d_age = g_rows[0]["drop_age"]
            d_sex = g_rows[0]["drop_sex"]

            avg_row = {
                "group": f"{group_name}_{m_type}_{'mm' if is_mm else 'uni'}_avg",
                "variant": v_name,
                "seed": "avg",
                "drop_age": d_age,
                "drop_sex": d_sex,
            }
            metric_keys = [
                "acc_mean", "acc_std", "auc_mean", "auc_std",
                "se_mean", "se_std", "sp_mean", "sp_std", "f1_mean", "f1_std",
            ]
            for mk in metric_keys:
                avg_row[mk] = sum(r[mk] for r in g_rows) / len(g_rows)
            rows.append(avg_row)

    fieldnames = [
        "group", "variant", "seed", "drop_age", "drop_sex",
        "acc_mean", "acc_std", "auc_mean", "auc_std",
        "se_mean", "se_std", "sp_mean", "sp_std", "f1_mean", "f1_std",
    ]
    with open(args.out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved results to {args.out_csv}")


if __name__ == "__main__":
    main()

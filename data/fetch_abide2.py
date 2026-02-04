#!/usr/bin/env python3
"""
Helper script for preparing ABIDE II data for AxisNet.

IMPORTANT
---------
ABIDE II imaging data and the official composite phenotypic file
(`ABIDEII_Composite_Phenotypic.csv`) are distributed via INDI/NITRC and
require you to accept the data usage terms. This script CANNOT bypass
that, it only:

- Shows you the expected folder layout for AxisNet
- Prints the relevant download URLs
- Tells you which command‑line flags to use afterwards

Typical usage
-------------

    python data/fetch_abide2.py --output_root ./data/ABIDE_II

Then follow the printed instructions.
"""

import argparse
from pathlib import Path


ABIDE2_PORTAL_URL = "https://fcon_1000.projects.nitrc.org/indi/abide/abide_II.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show instructions for downloading and organizing ABIDE II data for AxisNet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="./data/ABIDE_II",
        help="Root directory where you will place ABIDE II data",
    )
    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    cpac_folder = root / "cpac" / "filt_noglobal"

    print("\n================ ABIDE II Download / Setup Helper ================\n")
    print(f"1) Recommended root folder for ABIDE II data:")
    print(f"   {root}")
    print("\n   Inside this folder, AxisNet expects (for CPAC, filt_noglobal):")
    print(f"   {cpac_folder}/")
    print("     ├── <SUB_ID_1>/")
    print("     │    └── <SUB_ID_1>_ho_correlation.mat   (or similar connectivity .mat)")
    print("     ├── <SUB_ID_2>/")
    print("     └── ...")

    print("\n2) Official ABIDE II portal (manual download required):")
    print(f"   {ABIDE2_PORTAL_URL}")
    print("   On that page, locate and download at least:")
    print("   - ABIDEII_Composite_Phenotypic.csv")
    print("   - The MRI imaging collections for the sites you care about")

    print("\n3) Place files as follows (example):")
    print(f"   - Phenotype CSV:")
    print(f"       {root}/ABIDEII_Composite_Phenotypic.csv")
    print(f"   - Connectivity data (after preprocessing to correlation matrices):")
    print(f"       {cpac_folder}/<SUB_ID>/ <SUB_ID>_ho_correlation.mat")
    print("     (you may adapt the atlas/kind names, but keep them consistent")
    print("      with the arguments passed to the loader, e.g. kind='correlation', atlas='ho')")

    print("\n4) After data is in place, you can run AxisNet on ABIDE II, e.g.:")
    print("\n   10-fold stratified CV on ABIDE II:")
    print(f"   python scripts/run_abide2_cv.py \\")
    print(f"     --data_folder {cpac_folder} \\")
    print(f"     --phenotype_path {root}/ABIDEII_Composite_Phenotypic.csv")

    print("\n   Leave-One-Site-Out (LOSO) on ABIDE II:")
    print(f"   python scripts/run_abide2_cv.py \\")
    print(f"     --data_folder {cpac_folder} \\")
    print(f"     --phenotype_path {root}/ABIDEII_Composite_Phenotypic.csv \\")
    print("     --cv_type loso")

    print("\n   If your subject ID list is in a custom file:")
    print("     --subject_ids_path /path/to/ABIDE_II_subject_IDs.txt")

    print("\n5) Notes:")
    print("   - AxisNet assumes the phenotype CSV has at least columns:")
    print("       SUB_ID, FILE_ID, SITE_ID, DX_GROUP, AGE_AT_SCAN, SEX")
    print("   - If your ABIDE II phenotypic file uses different column names,")
    print("     you may need to adapt `data/abide_parser.py` accordingly.")
    print("   - Preprocessing from raw NIfTI to connectivity matrices (CPAC,")
    print("     filt_noglobal) is not handled by this script; it expects you ")
    print("     to run CPAC or other pipelines beforehand.")

    print("\n==================================================================\n")


if __name__ == "__main__":
    main()


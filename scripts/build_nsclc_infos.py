import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import SimpleITK as sitk


def extract_pid(path):
    name = os.path.basename(path).upper()
    m = re.search(r"(LUNG1[-_ ]?\d+)", name)
    if not m:
        return None
    token = m.group(1).replace("_", "-").replace(" ", "-")
    m2 = re.search(r"LUNG1-(\d+)", token)
    if not m2:
        return None
    return f"LUNG1-{int(m2.group(1)):03d}"


def read_mask_volume(mask_path):
    if not mask_path or not os.path.exists(mask_path):
        return 0.0
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_path))
    return float(np.sum(mask > 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ct-dir", required=True, help="Directory containing CT .nii/.nii.gz files")
    parser.add_argument("--mask-dir", required=True, help="Directory containing mask .nii/.nii.gz files")
    parser.add_argument("--clinical-csv", required=True, help="Clinical CSV path")
    parser.add_argument("--output", default="infos.json", help="Output infos.json path")
    args = parser.parse_args()

    df = pd.read_csv(args.clinical_csv)
    if "PatientID" not in df.columns or "deadstatus.event" not in df.columns:
        raise ValueError("Clinical CSV must contain PatientID and deadstatus.event columns")
    df = df[["PatientID", "deadstatus.event"]].dropna()
    pid_to_label = {str(r["PatientID"]): int(r["deadstatus.event"]) for _, r in df.iterrows()}

    ct_files = [
        os.path.join(args.ct_dir, x) for x in os.listdir(args.ct_dir)
        if x.endswith(".nii") or x.endswith(".nii.gz")
    ]
    mask_files = [
        os.path.join(args.mask_dir, x) for x in os.listdir(args.mask_dir)
        if x.endswith(".nii") or x.endswith(".nii.gz")
    ]
    mask_map = {extract_pid(p): p for p in mask_files if extract_pid(p)}

    infos = []
    for ct in ct_files:
        pid = extract_pid(ct)
        if not pid or pid not in pid_to_label:
            continue
        mask_path = mask_map.get(pid)
        if not mask_path:
            continue
        sid = os.path.basename(ct)
        if sid.endswith(".nii.gz"):
            sid = sid[:-7]
        elif sid.endswith(".nii"):
            sid = sid[:-4]
        infos.append(
            {
                "sid": sid,
                "pid": pid,
                "label": int(pid_to_label[pid]),
                "volume": read_mask_volume(mask_path),
            }
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(infos, f, indent=2)
    print(f"Saved {len(infos)} records to {args.output}")


if __name__ == "__main__":
    main()

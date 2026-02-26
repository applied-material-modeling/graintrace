import numpy as np
import pandas as pd

def select_smallest_cluster(df, nsmallest=10, min_size=1):
    d = df[df["n"] >= min_size]
    d = d.nsmallest(nsmallest, "n")
    return d["cluster_label"].to_numpy()

def select_highest_von_mises_from_components(df: pd.DataFrame, k: int = 1, required_cols=None):
    
    if required_cols is None:
        required_cols = ["sxx_mean_mean", "syy_mean_mean", "szz_mean_mean", "sxy_mean_mean", "syz_mean_mean", "sxz_mean_mean"]
    
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for von Mises: {missing}")

    d = df.copy()
    for c in required_cols:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=required_cols)

    sxx, syy, szz, sxy, syz, sxz = (d[c].to_numpy(dtype=float) for c in required_cols)

    vm = np.sqrt(
        0.5 * ((sxx - syy) ** 2 + (syy - szz) ** 2 + (szz - sxx) ** 2)
        + 3.0 * (sxy ** 2 + syz ** 2 + sxz ** 2)
    )

    d["_von_mises_from_means"] = vm
    return d.nlargest(k, "_von_mises_from_means")["cluster_label"].to_numpy()
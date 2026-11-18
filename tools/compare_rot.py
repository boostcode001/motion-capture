import pandas as pd
from pathlib import Path

D5 = Path("reports/day05")
FLAGS = ["fail", "no_pose", "low_conf", "legs_merged", "arms_merged", "bone_abnormal"]

def stats(table_csv, flags_csv):
    r = {}
    if table_csv.exists():
        t = pd.read_csv(table_csv)
        v = t["valid_pose"].astype(float)
        r["valid"] = v.mean()
        r["conf"] = t.loc[v == 1, "mean_keypoint_conf"].mean()
        if "rotation" in t:
            r["rot_used"] = (t.loc[v == 1, "rotation"] != 0).mean()
    if flags_csv.exists():
        f = pd.read_csv(flags_csv)
        for c in FLAGS:
            if c in f:
                r[c] = f[c].astype(float).mean()
    return r

rows = []
for SET in ["frontal_videos", "side_videos"]:
    for tcsv in sorted((D5 / "rot" / SET / "tables").glob("*_frame_features.csv")):
        n = tcsv.name.replace("_frame_features.csv", "")
        base = stats(D5 / SET / "tables" / tcsv.name, D5 / SET / "fail" / n / "frame_flags.csv")
        rot = stats(tcsv, D5 / "rot" / SET / "fail" / n / "frame_flags.csv")
        row = {"set": SET.replace("_videos", ""), "file": n}
        for k in ["valid", "conf", "no_pose", "legs_merged", "arms_merged", "fail"]:
            row[f"{k}_base"] = base.get(k)
            row[f"{k}_rot"] = rot.get(k)
        row["rot_used"] = rot.get("rot_used")
        rows.append(row)

df = pd.DataFrame(rows).round(3)
out = D5 / "rot" / "compare.csv"
df.to_csv(out, index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(df.to_string(index=False))
print("\n[평균]")
print(df.drop(columns=["file"]).groupby("set").mean().round(3).to_string())
print(f"\n저장: {out}")

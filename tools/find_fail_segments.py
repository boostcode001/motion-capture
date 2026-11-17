from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    p = argparse.ArgumentParser(description="18번 Feature CSV에서 관절 실패 구간 찾기")
    p.add_argument("--csv", required=True, help="reports/day03/tables/*_frame_features.csv")
    p.add_argument("--video", default=None, help="(선택) 18번 --save-video 결과 mp4")
    p.add_argument("--conf", type=float, default=0.60, help="평균 Keypoint Confidence 하한")
    p.add_argument("--low-count", type=int, default=3, help="LOW 관절 개수 상한")
    p.add_argument("--merge", type=float, default=0.15, help="좌우 관절 간격/몸통길이 하한")
    p.add_argument("--bone-min", type=float, default=0.5, help="뼈 길이/중앙값 하한")
    p.add_argument("--bone-max", type=float, default=1.8, help="뼈 길이/중앙값 상한")
    p.add_argument("--gap", type=int, default=2, help="이 Frame 수 이하로 끊긴 구간은 이어 붙임")
    p.add_argument("--min-len", type=int, default=3, help="이보다 짧은 구간은 무시")
    return p.parse_args()


def pt(df, name):
    return np.c_[df[f"{name}_x"].to_numpy(float), df[f"{name}_y"].to_numpy(float)]


def dist(a, b):
    return np.linalg.norm(a - b, axis=1)


def main():
    args = parse_args()
    csv_path = Path(args.csv)
    df = pd.read_csv(csv_path)
    n = len(df)

    flags = {k: np.zeros(n, bool) for k in
             ["no_pose", "low_conf", "legs_merged", "arms_merged", "bone_abnormal"]}

    # 1) 미검출 / 어깨 폭 붕괴 (18번 valid_pose = 0)
    flags["no_pose"] = df["valid_pose"].to_numpy() == 0

    v = ~flags["no_pose"]

    # 2) 낮은 Confidence
    conf = df["mean_keypoint_conf"].fillna(0).to_numpy()
    low = df["low_keypoint_count"].fillna(99).to_numpy()
    flags["low_conf"] = v & ((conf < args.conf) | (low >= args.low_count))

    # 몸통 길이(어깨 중심~골반 중심)로 다시 나눠 각도 영향 줄이기
    sh_c = (pt(df, "left_shoulder") + pt(df, "right_shoulder")) / 2
    hip_c = (pt(df, "left_hip") + pt(df, "right_hip")) / 2
    torso = dist(sh_c, hip_c) + 1e-6

    # 3) 좌우 다리 합쳐짐 (무릎 또는 발목 간격이 너무 좁음)
    knee_gap = dist(pt(df, "left_knee"), pt(df, "right_knee")) / torso
    ankle_gap = dist(pt(df, "left_ankle"), pt(df, "right_ankle")) / torso
    flags["legs_merged"] = v & ((knee_gap < args.merge) | (ankle_gap < args.merge))

    # 4) 좌우 팔 합쳐짐
    wrist_gap = dist(pt(df, "left_wrist"), pt(df, "right_wrist")) / torso
    elbow_gap = dist(pt(df, "left_elbow"), pt(df, "right_elbow")) / torso
    flags["arms_merged"] = v & ((wrist_gap < args.merge) | (elbow_gap < args.merge))

    # 5) 뼈 길이 이상 (영상 전체 중앙값 대비)
    bones = {
        "l_thigh": ("left_hip", "left_knee"), "r_thigh": ("right_hip", "right_knee"),
        "l_shin": ("left_knee", "left_ankle"), "r_shin": ("right_knee", "right_ankle"),
        "l_upper": ("left_shoulder", "left_elbow"), "r_upper": ("right_shoulder", "right_elbow"),
        "l_fore": ("left_elbow", "left_wrist"), "r_fore": ("right_elbow", "right_wrist"),
    }
    bad_bones = [[] for _ in range(n)]
    for bname, (a, b) in bones.items():
        length = dist(pt(df, a), pt(df, b)) / torso
        med = np.nanmedian(length[v]) if v.any() else np.nan
        ratio = length / (med + 1e-6)
        bad = v & ((ratio < args.bone_min) | (ratio > args.bone_max))
        flags["bone_abnormal"] |= bad
        for i in np.where(bad)[0]:
            bad_bones[i].append(bname)

    # Frame별 결과
    out = df[["frame_id", "time_sec", "valid_pose", "mean_keypoint_conf", "low_keypoint_count"]].copy()
    out["knee_gap"], out["ankle_gap"], out["wrist_gap"] = knee_gap.round(3), ankle_gap.round(3), wrist_gap.round(3)
    for k, f in flags.items():
        out[k] = f.astype(int)
    out["bad_bones"] = [",".join(b) for b in bad_bones]
    fail = np.any(np.vstack(list(flags.values())), axis=0)
    out["fail"] = fail.astype(int)

    # 연속 구간 묶기
    segs, start, last = [], None, None
    for i in np.where(fail)[0]:
        if start is None:
            start = last = i
        elif i - last <= args.gap + 1:
            last = i
        else:
            segs.append((start, last)); start = last = i
    if start is not None:
        segs.append((start, last))
    segs = [s for s in segs if s[1] - s[0] + 1 >= args.min_len]

    seg_rows = []
    for k, (s, e) in enumerate(segs, 1):
        part = out.iloc[s:e + 1]
        reasons = {name: int(part[name].sum()) for name in flags}
        main_reason = max(reasons, key=reasons.get)
        seg_rows.append({
            "segment": k,
            "start_frame": int(part.frame_id.iloc[0]), "end_frame": int(part.frame_id.iloc[-1]),
            "start_sec": float(part.time_sec.iloc[0]), "end_sec": float(part.time_sec.iloc[-1]),
            "frames": e - s + 1,
            "main_reason": main_reason,
            **{f"n_{r}": c for r, c in reasons.items()},
            "mid_frame": int(part.frame_id.iloc[len(part) // 2]),
        })
    seg_df = pd.DataFrame(seg_rows)

    # 저장
    out_dir = ROOT / "reports" / "day05" / "fail" / csv_path.stem.replace("_frame_features", "")
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_dir / "frame_flags.csv", index=False, encoding="utf-8-sig")
    seg_df.to_csv(out_dir / "fail_segments.csv", index=False, encoding="utf-8-sig")

    # (선택) 구간 대표 이미지
    if args.video and len(seg_df):
        cap = cv2.VideoCapture(args.video)
        for r in seg_rows:
            cap.set(cv2.CAP_PROP_POS_FRAMES, r["mid_frame"] - 1)
            ok, frame = cap.read()
            if ok:
                label = f"seg{r['segment']} {r['start_sec']:.1f}-{r['end_sec']:.1f}s {r['main_reason']}"
                cv2.putText(frame, label, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3)
                cv2.imwrite(str(out_dir / f"seg{r['segment']:02d}_f{r['mid_frame']}.jpg"), frame)
        cap.release()

    # 요약
    print(f"Source        : {csv_path.name}")
    print(f"Total frames  : {n}")
    print(f"Fail frames   : {int(fail.sum())} ({fail.mean():.1%})")
    for k, f in flags.items():
        print(f"  {k:14s}: {int(f.sum()):5d} ({f.mean():.1%})")
    print(f"Fail segments : {len(seg_df)}")
    if len(seg_df):
        print(seg_df[["segment", "start_sec", "end_sec", "frames", "main_reason"]].to_string(index=False))
    print(f"Saved         : {out_dir}")


if __name__ == "__main__":
    main()

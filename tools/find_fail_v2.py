"""
실패 구간 탐지 v2 — 픽셀 좌표 + 시간 연속성 기반
사용: python tools/find_fail_v2.py --csv reports/day05/.../tables/<stem>_frame_features.csv [--video 원본.mp4]
판정(모두 픽셀 좌표 기준, 몸통·어깨 정규화 사용 안 함):
  scale = 프레임별 관절 bbox 대각선(D)
  merged_legs / merged_arms : 좌우 무릎·발목(손목·팔꿈치) 간격 / D < --merge
  lr_swap   : 이전 프레임과 비교해 좌↔우 교차 배정 비용이 더 작음
  limb_swap : 손목이 이전 프레임의 발목(또는 발목이 손목) 위치로 순간이동
  bone_jump : 뼈 길이가 직전 유효 프레임 대비 ×--jump 이상 / 1/--jump 이하로 급변
  kp_jump   : 관절 이동량 / D > --move (튀는 관절)
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
I = {n: i for i, n in enumerate(NAMES)}
BONES = {"l_upper": (5, 7), "l_fore": (7, 9), "r_upper": (6, 8), "r_fore": (8, 10),
         "l_thigh": (11, 13), "l_shin": (13, 15), "r_thigh": (12, 14), "r_shin": (14, 16)}
PAIRS = {"leg": [(13, 14), (15, 16)], "arm": [(7, 8), (9, 10)]}
BODY = list(range(5, 17))  # 얼굴 제외


def find_cols(df):
    cols = {}
    for n in NAMES:
        for ax in ("x", "y"):
            cand = [c for c in df.columns if c.endswith(f"{n}_{ax}") and "conf" not in c]
            if not cand:
                raise SystemExit(f"좌표 컬럼을 찾지 못함: *{n}_{ax}  (CSV 컬럼명을 확인하세요)")
            cols[(n, ax)] = sorted(cand, key=len)[0]
    return cols


def to_pixel(df, cols):
    xy = np.full((len(df), 17, 2), np.nan)
    s = df["shoulder_width"].to_numpy(float)
    cx, cy = df["hip_center_x"].to_numpy(float), df["hip_center_y"].to_numpy(float)
    for n in NAMES:
        xy[:, I[n], 0] = df[cols[(n, "x")]].to_numpy(float) * s + cx
        xy[:, I[n], 1] = df[cols[(n, "y")]].to_numpy(float) * s + cy
    xy[df["valid_pose"].to_numpy() != 1] = np.nan
    return xy


def dist(a, b):
    return np.linalg.norm(a - b, axis=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--video")
    ap.add_argument("--merge", type=float, default=0.04)
    ap.add_argument("--jump", type=float, default=1.6)
    ap.add_argument("--move", type=float, default=0.15)
    ap.add_argument("--gap", type=int, default=2)
    ap.add_argument("--min-len", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()

    csv = Path(a.csv)
    df = pd.read_csv(csv)
    xy = to_pixel(df, find_cols(df))
    T = len(df)
    body = xy[:, BODY]
    D = np.linalg.norm(np.nanmax(body, 1) - np.nanmin(body, 1), axis=-1)  # 관절 bbox 대각선

    f = pd.DataFrame({"frame_id": df["frame_id"], "time_sec": df["time_sec"]})
    f["no_pose"] = (df["valid_pose"] != 1).astype(int)
    for k, prs in PAIRS.items():
        g = np.nanmin(np.stack([dist(xy[:, p], xy[:, q]) for p, q in prs], 1), 1) / D
        f[f"{k}_gap"] = np.round(g, 3)
        f[f"merged_{k}s"] = (g < a.merge).astype(int)

    lr = np.zeros(T, int); limb = np.zeros(T, int); bj = np.zeros(T, int); kj = np.zeros(T, int)
    bad = [""] * T
    prev = None
    for t in range(T):
        cur = xy[t]
        if np.isnan(cur).all():
            continue
        if prev is not None:
            P = xy[prev]; d = D[t]
            # 좌우 교차
            for p, q in PAIRS["leg"] + PAIRS["arm"]:
                keep = dist(cur[p], P[p]) + dist(cur[q], P[q])
                swap = dist(cur[p], P[q]) + dist(cur[q], P[p])
                if swap + 0.02 * d < keep and keep > 0.08 * d:
                    lr[t] = 1
            # 팔↔다리 순간이동 (손목 ↔ 발목)
            for w in (9, 10):
                for an in (15, 16):
                    if dist(cur[w], P[an]) < 0.5 * dist(cur[w], P[w]) and dist(cur[w], P[w]) > a.move * d:
                        limb[t] = 1
                    if dist(cur[an], P[w]) < 0.5 * dist(cur[an], P[an]) and dist(cur[an], P[an]) > a.move * d:
                        limb[t] = 1
            # 뼈 길이 급변
            names = []
            for bn, (p, q) in BONES.items():
                r = dist(cur[p], cur[q]) / (dist(P[p], P[q]) + 1e-6)
                if r > a.jump or r < 1 / a.jump:
                    names.append(bn)
            if names:
                bj[t] = 1; bad[t] = ",".join(names)
            # 튀는 관절
            if np.nanmax(dist(cur[BODY], P[BODY])) / d > a.move:
                kj[t] = 1
        prev = t
    f["lr_swap"], f["limb_swap"], f["bone_jump"], f["kp_jump"], f["bad_bones"] = lr, limb, bj, kj, bad
    flags = ["no_pose", "merged_legs", "merged_arms", "lr_swap", "limb_swap", "bone_jump", "kp_jump"]
    f["fail"] = (f[flags].sum(axis=1) > 0).astype(int)

    # 구간 묶기
    segs, s, last = [], None, None
    for t in np.where(f["fail"].to_numpy() == 1)[0]:
        if s is None:
            s = last = t
        elif t - last <= a.gap + 1:
            last = t
        else:
            segs.append((s, last)); s = last = t
    if s is not None:
        segs.append((s, last))
    rows = []
    keep_seg = [x for x in segs if x[1] - x[0] + 1 >= a.min_len
                or f.iloc[x[0]:x[1] + 1][["lr_swap", "limb_swap"]].to_numpy().any()]
    for k, (s, e) in enumerate(keep_seg, 1):
        part = f.iloc[s:e + 1]
        cnt = part[flags].sum()
        rows.append({"segment": k, "start_frame": int(part.frame_id.iloc[0]), "end_frame": int(part.frame_id.iloc[-1]),
                     "start_sec": part.time_sec.iloc[0], "end_sec": part.time_sec.iloc[-1], "frames": len(part),
                     "main_reason": cnt.idxmax(), **{f"n_{c}": int(cnt[c]) for c in flags},
                     "mid_frame": int(part.frame_id.iloc[len(part) // 2])})
    seg = pd.DataFrame(rows)

    stem = csv.name.replace("_frame_features.csv", "")
    out = Path(a.out) if a.out else csv.parent.parent / "fail_v2" / stem
    out.mkdir(parents=True, exist_ok=True)
    f.to_csv(out / "frame_flags.csv", index=False)
    seg.to_csv(out / "fail_segments.csv", index=False)

    if a.video and len(seg):
        import cv2
        cap = cv2.VideoCapture(a.video)
        for r in seg.itertuples():
            cap.set(cv2.CAP_PROP_POS_FRAMES, r.mid_frame - 1)
            ok, img = cap.read()
            if ok:
                cv2.imwrite(str(out / f"seg{r.segment:02d}_f{r.mid_frame:04d}_{r.main_reason}.jpg"), img)
        cap.release()

    print(f"[{stem}] frames={T}  fail={f['fail'].mean():.3f}")
    print(f[flags].mean().round(3).to_string())
    print(f"segments={len(seg)} → {out}")


if __name__ == "__main__":
    main()
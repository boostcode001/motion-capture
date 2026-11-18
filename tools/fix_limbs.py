"""
기준 골격 기반 팔·다리 보정 (모델 고정, 후처리만)
사용:
  python tools/fix_limbs.py --csv reports/day05/<SET>/tables/<stem>_frame_features.csv \
         [--video data/day05/<SET>/<stem>.mp4] [--ref-start 120 --ref-end 140]
출력 (같은 tables 폴더):
  <stem>_fixed_frame_features.csv : 보정된 좌표·각도 (18번 CSV와 같은 컬럼)
  ../fix_log/<stem>_fix_log.csv   : 프레임별 보정 내역
  ../fix_log/<stem>_fixed_pose.mp4: (--video 시) 빨강=원본, 초록=보정
원리:
  1) 영상 앞부분에서 '정지 + conf 높음 + 좌우 안 겹침' 구간을 기준 골격으로 선택
  2) 팔2·다리2 (팔꿈치-손목, 무릎-발목)를 '직전 신뢰 위치'와 비교해 최소 비용으로 재배정 (24가지)
  3) 두 라벨이 한 팔다리에 겹치면 비용이 큰 쪽을 결측 → 직전 위치로 유지(hold)
  4) 비현실적으로 순간이동한 팔다리도 결측 처리, 일정 프레임 이상 결측이면 새 관측을 다시 수용
"""
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd

NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
LIMBS = {"L_arm": (7, 9), "R_arm": (8, 10), "L_leg": (13, 15), "R_leg": (14, 16)}
LN = list(LIMBS)
BODY = list(range(5, 17))
PERMS = list(itertools.permutations(range(4)))
ANGLES = {"left_elbow": (5, 7, 9), "right_elbow": (6, 8, 10),
          "left_knee": (11, 13, 15), "right_knee": (12, 14, 16)}


def find_cols(df):
    cols = {}
    for n in NAMES:
        for ax in ("x", "y"):
            c = [k for k in df.columns if k.endswith(f"{n}_{ax}") and "conf" not in k]
            if not c:
                raise SystemExit(f"좌표 컬럼 없음: *{n}_{ax}")
            cols[(n, ax)] = sorted(c, key=len)[0]
    return cols


def angle(a, b, c):
    v1, v2 = a - b, c - b
    cos = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--video")
    ap.add_argument("--ref-start", type=int, help="기준 구간 시작 frame_id (미지정 시 자동)")
    ap.add_argument("--ref-end", type=int)
    ap.add_argument("--ref-len", type=int, default=10, help="자동 기준 구간 길이(프레임)")
    ap.add_argument("--ref-search", type=float, default=0.5, help="영상 앞쪽 몇 %%에서 기준을 찾을지")
    ap.add_argument("--merge", type=float, default=0.04, help="두 팔다리 끝점 거리/D 이 값 미만이면 겹침")
    ap.add_argument("--swap-margin", type=float, default=0.03, help="재배정 시 필요한 최소 비용 이득")
    ap.add_argument("--max-move", type=float, default=0.20, help="한 프레임 최대 이동/D, 초과 시 결측")
    ap.add_argument("--recover", type=int, default=8, help="연속 결측 N프레임 뒤 새 관측 재수용")
    ap.add_argument("--fill", choices=["hold", "nan"], default="hold")
    a = ap.parse_args()

    csv = Path(a.csv)
    df = pd.read_csv(csv)
    cols = find_cols(df)
    T = len(df)
    s = df["shoulder_width"].to_numpy(float)
    cx, cy = df["hip_center_x"].to_numpy(float), df["hip_center_y"].to_numpy(float)
    valid = df["valid_pose"].to_numpy() == 1
    xy = np.full((T, 17, 2), np.nan)
    for i, n in enumerate(NAMES):
        xy[:, i, 0] = df[cols[(n, "x")]].to_numpy(float) * s + cx
        xy[:, i, 1] = df[cols[(n, "y")]].to_numpy(float) * s + cy
    xy[~valid] = np.nan
    orig = xy.copy()
    D = np.linalg.norm(np.nanmax(xy[:, BODY], 1) - np.nanmin(xy[:, BODY], 1), axis=-1)

    # ---------- 1) 기준 구간 ----------
    fid = df["frame_id"].to_numpy()
    if a.ref_start is not None:
        e = a.ref_end if a.ref_end is not None else a.ref_start + a.ref_len - 1
        ref = np.where((fid >= a.ref_start) & (fid <= e) & valid)[0]
    else:
        conf = df["mean_keypoint_conf"].to_numpy(float)
        low = df.get("low_keypoint_count", pd.Series(0, index=df.index)).to_numpy(float)
        move = np.r_[np.nan, np.nanmedian(np.linalg.norm(np.diff(xy[:, BODY], axis=0), axis=-1), 1) / D[1:]]
        gaps = np.stack([np.linalg.norm(xy[:, LIMBS[p][1]] - xy[:, LIMBS[q][1]], axis=-1) / D
                         for p, q in itertools.combinations(LN, 2)], 1)
        ok = valid & (np.nanmin(gaps, 1) >= 2 * a.merge)
        q = conf - 0.02 * low - 3 * np.nan_to_num(move, nan=1)
        q[~ok] = -np.inf
        L, lim = a.ref_len, max(a.ref_len, int(T * a.ref_search))
        sc = [q[i:i + L].mean() for i in range(0, lim - L + 1)]
        i0 = int(np.argmax(sc))
        if not np.isfinite(sc[i0]):
            raise SystemExit("기준 구간을 찾지 못함 → --ref-start / --ref-end 로 직접 지정하세요")
        ref = np.arange(i0, i0 + L)
    if len(ref) == 0:
        raise SystemExit("기준 구간에 유효 프레임 없음")
    print(f"기준 구간: frame {fid[ref[0]]}~{fid[ref[-1]]} ({df['time_sec'].iloc[ref[0]]:.2f}s~{df['time_sec'].iloc[ref[-1]]:.2f}s)")
    state = np.stack([np.nanmedian(xy[ref][:, list(LIMBS[n])], 0) for n in LN])  # (4,2,2)

    # ---------- 2~4) 추적·보정 ----------
    miss = np.zeros(4, int)
    log = []
    for t in range(T):
        if not valid[t]:
            log.append({"frame_id": fid[t], "perm": "", "swapped": 0, "dropped": "", "filled": ""})
            continue
        d = D[t]
        obs = np.stack([xy[t, list(LIMBS[n])] for n in LN])            # (4,2,2)
        C = np.linalg.norm(obs[None] - state[:, None], axis=-1).mean(-1) / d   # C[i,j]: 실제 i ← 관측 j
        C = np.nan_to_num(C, nan=9.0)
        costs = [sum(C[i, p[i]] for i in range(4)) for p in PERMS]
        best = int(np.argmin(costs))
        perm = PERMS[best] if costs[0] - costs[best] > a.swap_margin else PERMS[0]
        new = obs[list(perm)]
        c_ass = np.array([C[i, perm[i]] for i in range(4)])
        drop = set()
        for i, k in itertools.combinations(range(4), 2):                 # 겹침
            if np.linalg.norm(new[i, 1] - new[k, 1]) / d < a.merge:
                drop.add(i if c_ass[i] > c_ass[k] else k)
        for i in range(4):                                               # 순간이동
            if c_ass[i] > a.max_move and miss[i] < a.recover:
                drop.add(i)
        filled = []
        for i in range(4):
            if i in drop:
                miss[i] += 1
                new[i] = state[i] if a.fill == "hold" else np.nan
                filled.append(LN[i])
            else:
                miss[i] = 0
                state[i] = new[i]
        for i, n in enumerate(LN):
            xy[t, list(LIMBS[n])] = new[i]
        log.append({"frame_id": fid[t], "perm": "" if perm == PERMS[0] else "-".join(LN[j] for j in perm),
                    "swapped": int(perm != PERMS[0]), "dropped": ",".join(filled), "filled": len(filled)})
    lg = pd.DataFrame(log)

    # ---------- 출력 CSV ----------
    out = df.copy()
    for i, n in enumerate(NAMES):
        out[cols[(n, "x")]] = (xy[:, i, 0] - cx) / s
        out[cols[(n, "y")]] = (xy[:, i, 1] - cy) / s
    changed = (lg["swapped"] == 1) | (lg["filled"] > 0)
    for jn, (p, j, c) in ANGLES.items():
        ac = [k for k in df.columns if jn in k and "angle" in k]
        if not ac:
            continue
        new_a, old_a = angle(xy[:, p], xy[:, j], xy[:, c]), df[ac[0]].to_numpy(float)
        same = (~changed.to_numpy()) & valid & np.isfinite(old_a) & (np.abs(new_a) > 1)
        ratio = np.nanmedian(old_a[same] / new_a[same]) if same.any() else 1.0   # 단위 자동 맞춤(도/180 등)
        out[ac[0]] = new_a * ratio
    stem = csv.name.replace("_frame_features.csv", "")
    out_csv = csv.parent / f"{stem}_fixed_frame_features.csv"
    out.to_csv(out_csv, index=False)
    logdir = csv.parent.parent / "fix_log"
    logdir.mkdir(parents=True, exist_ok=True)
    lg.to_csv(logdir / f"{stem}_fix_log.csv", index=False)

    v = valid.sum()
    print(f"[{stem}] 유효 {v}프레임 | 재배정(swap) {int(lg.swapped.sum())} ({lg.swapped.sum()/max(v,1):.3f}) "
          f"| 결측·유지 {int((lg.filled>0).sum())} ({(lg.filled>0).sum()/max(v,1):.3f})")
    print(lg.loc[lg.filled > 0, "dropped"].str.split(",").explode().value_counts().to_string())
    print(f"→ {out_csv}\n→ {logdir / (stem + '_fix_log.csv')}")

    # ---------- 확인용 영상 ----------
    if a.video:
        import cv2
        SK = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
              (11, 13), (13, 15), (12, 14), (14, 16)]
        cap = cv2.VideoCapture(a.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        W, H = int(cap.get(3)), int(cap.get(4))
        vw = cv2.VideoWriter(str(logdir / f"{stem}_fixed_pose.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
        row = {f: i for i, f in enumerate(fid)}
        k = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            k += 1
            t = row.get(k)
            if t is not None and valid[t]:
                for P, col, th in ((orig[t], (0, 0, 255), 1), (xy[t], (0, 255, 0), 3)):
                    for p, q in SK:
                        if np.isfinite(P[[p, q]]).all():
                            cv2.line(img, tuple(map(int, P[p])), tuple(map(int, P[q])), col, th)
                if lg.swapped.iloc[t] or lg.filled.iloc[t]:
                    cv2.putText(img, f"swap:{lg.perm.iloc[t]} drop:{lg.dropped.iloc[t]}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            vw.write(img)
        cap.release(); vw.release()
        print(f"→ {logdir / (stem + '_fixed_pose.mp4')}  (빨강=원본, 초록=보정)")


if __name__ == "__main__":
    main()

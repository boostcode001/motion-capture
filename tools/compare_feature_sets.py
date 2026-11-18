"""
Feature 구성 비교 실험 (6일차 방식 확장: 30 Step Sequence → Random Forest)
사용:
  python tools/compare_feature_sets.py                       # frontal_videos, 45_videos, side_videos 전부
  python tools/compare_feature_sets.py --sets 45_videos --actions 001 002 003 009 010
  python tools/compare_feature_sets.py --same-person kjh1 kjh2 # 같은 사람이면 하나의 Subject로 묶기
입력 : reports/day05/<SET>/tables/<A|T><번호>_<사람>[_<take>]_frame_features.csv  (_fixed 제외 / --use-fixed 로 보정본 사용)
정답 : 파일명의 동작 번호(A009 와 T009 는 같은 동작 009)
평가 : Leave-One-Subject-Out — 한 사람씩 빼고 나머지로 학습 → 뺀 사람으로 평가 (사람 수가 적어 S01/S02/S03 고정 분할 대신 사용)
출력 : reports/day05/experiments/feature_sets.csv, feature_sets_predictions.csv
"""
import argparse
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

NAMES = ["nose", "left_eye", "right_eye", "left_ear", "right_ear",
         "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
         "left_wrist", "right_wrist", "left_hip", "right_hip",
         "left_knee", "right_knee", "left_ankle", "right_ankle"]
ARMS = NAMES[5:11]
TORSO = NAMES[11:13]
LEGS = NAMES[13:17]
PAT = re.compile(r"^([AT])(\d{3})_([A-Za-z]+\d*)(?:\(\d+\))?(?:_(\d+))?$")


def coord_cols(df, names):
    out = []
    for n in names:
        for ax in ("x", "y"):
            c = sorted([k for k in df.columns if k.endswith(f"{n}_{ax}") and "conf" not in k], key=len)
            if c:
                out.append(c[0])
    return out


def angle_cols(df, joints):
    return [k for k in df.columns if "angle" in k and any(j in k for j in joints)]


def bbox_feats(df):
    """bbox_x1,y1,x2,y2 (또는 bbox_w,bbox_h) 가 있으면 가로세로 비율·크기 변화 Feature 생성"""
    if {"bbox_w", "bbox_h"} <= set(df.columns):
        w, h = df["bbox_w"].astype(float), df["bbox_h"].astype(float)
    elif {"bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"} <= set(df.columns):
        w = df["bbox_x2"].astype(float) - df["bbox_x1"].astype(float)
        h = df["bbox_y2"].astype(float) - df["bbox_y1"].astype(float)
    else:
        return None
    out = pd.DataFrame(index=df.index)
    out["bbox_aspect"] = w / (h + 1e-6)
    out["bbox_w_rel"] = w / (w.median() + 1e-6)     # 영상 내 기준 대비 상대 크기 → 거리 차이 영향 제거
    out["bbox_h_rel"] = h / (h.median() + 1e-6)
    return out


FEATURE_SETS = {
    "all_38":        lambda df: (coord_cols(df, NAMES), angle_cols(df, ["elbow", "knee"]), False),
    "coords_only":   lambda df: (coord_cols(df, NAMES), [], False),
    "arms_torso":    lambda df: (coord_cols(df, ["nose"] + ARMS + TORSO), angle_cols(df, ["elbow"]), False),
    "arms_only":     lambda df: (coord_cols(df, ARMS), angle_cols(df, ["elbow"]), False),
    "arms_torso+bbox": lambda df: (coord_cols(df, ["nose"] + ARMS + TORSO), angle_cols(df, ["elbow"]), True),
    "all_38+bbox":   lambda df: (coord_cols(df, NAMES), angle_cols(df, ["elbow", "knee"]), True),
}


def to_sequence(df, cols, use_bbox, steps, min_valid, clip):
    v = df[df["valid_pose"] == 1]
    if len(v) < min_valid:
        return None
    X = v[cols].astype(float)
    if use_bbox:
        b = bbox_feats(df)
        if b is None:
            return "NO_BBOX"
        X = pd.concat([X, b.loc[v.index]], axis=1)
    X = X.clip(-clip, clip)                       # 옆면 등 어깨너비 붕괴로 생기는 극단값 제한
    t = v["frame_id"].to_numpy(float)
    grid = np.linspace(t.min(), t.max(), steps)  # frame_id 기준 시간 간격을 유지한 Resampling (6일차 방식)
    seq = np.stack([np.interp(grid, t, X[c].to_numpy()) for c in X.columns], 1)
    return seq.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", nargs="+", default=["frontal_videos", "45_videos", "side_videos"])
    ap.add_argument("--actions", nargs="*", help="예: 001 002 009 010 (미지정 시 전부)")
    ap.add_argument("--same-person", nargs="*", default=[], help="같은 사람으로 묶을 이름들")
    ap.add_argument("--use-fixed", action="store_true")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--min-valid", type=int, default=10)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--trees", type=int, default=300)
    a = ap.parse_args()

    D5 = Path("reports/day05")
    rows = []
    for st in a.sets:
        for f in sorted((D5 / st / "tables").glob("*_frame_features.csv")):
            fixed = "_fixed_frame_features" in f.name
            if fixed != a.use_fixed:
                continue
            stem = f.name.replace("_fixed_frame_features.csv", "").replace("_frame_features.csv", "")
            m = PAT.match(stem)
            if not m:
                print(f"  건너뜀(파일명 규칙 불일치): {st}/{stem}")
                continue
            grp, act, person, take = m.groups()
            if a.actions and act not in a.actions:
                continue
            subj = a.same_person[0] if person in a.same_person else person
            rows.append({"set": st, "file": stem, "group": grp, "action": act, "subject": subj, "path": f})
    meta = pd.DataFrame(rows)
    if meta.empty:
        raise SystemExit("입력 CSV 없음")
    print(f"Clip {len(meta)}개 | Subject {sorted(meta.subject.unique())}")
    print(pd.crosstab(meta.action, meta.set).to_string(), "\n")
    if meta.groupby("action").set.nunique().max() < len(a.sets):
        print("※ 주의: 동작마다 촬영 각도 구성이 다르면 모델이 '동작'이 아니라 '각도'를 학습할 수 있습니다.\n")

    dfs = {r.file + "|" + r.set: pd.read_csv(r.path) for r in meta.itertuples()}
    results, preds = [], []
    for name, fn in FEATURE_SETS.items():
        X, keep = [], []
        for i, r in enumerate(meta.itertuples()):
            df = dfs[r.file + "|" + r.set]
            cols, angs, use_b = fn(df)
            s = to_sequence(df, cols + angs, use_b, a.steps, a.min_valid, a.clip)
            if isinstance(s, str):
                X = None; break
            if s is not None:
                X.append(s); keep.append(i)
        if X is None:
            print(f"[{name}] 건너뜀: CSV에 bbox 컬럼 없음 (18번에 bbox 저장 추가 후 재실행)")
            continue
        X = np.stack(X); m = meta.iloc[keep].reset_index(drop=True)
        y = np.array(["A" + str(v) for v in m.action], dtype=object)
        g = np.array([str(v) for v in m.subject], dtype=object)
        yp = np.array(["NA"] * len(y), dtype=object)
        for s in np.unique(g):                     # Leave-One-Subject-Out
            te = g == s
            if len(np.unique(y[~te])) < 2:
                yp[te] = "NA"; continue
            clf = RandomForestClassifier(n_estimators=a.trees, random_state=42,
                                         class_weight="balanced", n_jobs=-1)
            clf.fit(X[~te], y[~te])
            yp[te] = clf.predict(X[te])
        ok = yp != "NA"
        acc, f1 = accuracy_score(y[ok], yp[ok]), f1_score(y[ok], yp[ok], average="macro")
        per_subj = {s: round(accuracy_score(y[(g == s) & ok], yp[(g == s) & ok]), 3) for s in np.unique(g)}
        results.append({"feature_set": name, "n_features": X.shape[1], "clips": len(y),
                        "accuracy": round(acc, 4), "macro_f1": round(f1, 4), **{f"acc_{k}": v for k, v in per_subj.items()}})
        preds.append(m.assign(feature_set=name, pred=yp, correct=(yp == y).astype(int)).drop(columns="path"))
        print(f"[{name:16s}] feat={X.shape[1]:5d}  acc={acc:.3f}  macroF1={f1:.3f}  사람별={per_subj}")

    out = D5 / "experiments"; out.mkdir(parents=True, exist_ok=True)
    res = pd.DataFrame(results); res.to_csv(out / "feature_sets.csv", index=False)
    pr = pd.concat(preds)
    pr["action"] = "A" + pr["action"].astype(str); pr.to_csv(out / "feature_sets_predictions.csv", index=False)
    best = res.sort_values("macro_f1", ascending=False).iloc[0].feature_set
    b = pr[pr.feature_set == best]
    labels = sorted(b.action.unique())
    print(f"\n[혼동행렬: {best}] 행=정답, 열=예측")
    print(pd.DataFrame(confusion_matrix(b.action, b.pred, labels=labels), index=labels, columns=labels).to_string())
    wrong = b[b.correct == 0][["set", "file", "action", "pred"]]
    if len(wrong):
        print("\n[틀린 Clip]\n" + wrong.to_string(index=False))
    print(f"\n저장: {out/'feature_sets.csv'}, {out/'feature_sets_predictions.csv'}")


if __name__ == "__main__":
    main()

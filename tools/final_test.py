"""
최종 학습 + Test 평가 (6일차 35·36·40번 흐름을 우리 데이터에 맞춤)
  Test 사람 : --test-subject (기본 kss)  → 마지막에 한 번만 평가
  나머지 사람: Leave-One-Subject-Out 로 Feature 구성 선택(Validation) → 전체로 최종 학습
사용:
  python tools/final_test.py --test-subject kss --same-person kjh1 khj1
출력: artifacts/day05/rf_<feature_set>.joblib, rf_feature_info.json
      reports/day05/experiments/final_test_<subject>_*.csv / .png
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_feature_sets import FEATURE_SETS, PAT, to_sequence   # 같은 tools 폴더의 함수 재사용


def load_meta(sets, actions, same, use_fixed):
    D5 = Path("reports/day05"); rows = []
    for st in sets:
        for f in sorted((D5 / st / "tables").glob("*_frame_features.csv")):
            if ("_fixed_frame_features" in f.name) != use_fixed:
                continue
            stem = f.name.replace("_fixed_frame_features.csv", "").replace("_frame_features.csv", "")
            m = PAT.match(stem)
            if not m:
                print(f"  건너뜀(파일명): {st}/{stem}"); continue
            grp, act, person, _ = m.groups()
            if actions and act not in actions:
                continue
            rows.append({"set": st, "file": stem, "action": "A" + act,
                         "subject": same[0] if person in same else person, "path": f})
    return pd.DataFrame(rows)


def build_xy(meta, fs, a):
    X, keep = [], []
    for i, r in enumerate(meta.itertuples()):
        df = pd.read_csv(r.path)
        cols, angs, use_b = FEATURE_SETS[fs](df)
        s = to_sequence(df, cols + angs, use_b, a.steps, a.min_valid, a.clip)
        if isinstance(s, str):
            return None, None, None
        if s is not None:
            X.append(s); keep.append(i)
    m = meta.iloc[keep].reset_index(drop=True)
    return np.stack(X), np.array(m.action, dtype=object), m


def rf(a):
    return RandomForestClassifier(n_estimators=a.trees, random_state=42, class_weight="balanced", n_jobs=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-subject", default="kss")
    ap.add_argument("--sets", nargs="+", default=["frontal_videos", "45_videos", "side_videos"])
    ap.add_argument("--actions", nargs="*")
    ap.add_argument("--same-person", nargs="*", default=[])
    ap.add_argument("--feature-sets", nargs="+", default=["all_38", "coords_only", "arms_torso", "arms_only"])
    ap.add_argument("--use-fixed", action="store_true")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--min-valid", type=int, default=10)
    ap.add_argument("--clip", type=float, default=10.0)
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--from-splits", action="store_true",
                    help="5일차 결과 data/day05/splits/train·val·test.csv 사용 (승인 Clip만, 교재 방식: Train 학습 → Val 선택 → Test)")
    ap.add_argument("--refit", action="store_true", help="--from-splits 에서 선택 후 Train+Val 로 다시 학습")
    a = ap.parse_args()

    if a.from_splits:
        return run_from_splits(a)
    meta = load_meta(a.sets, a.actions, a.same_person, a.use_fixed)
    ts = a.test_subject
    if ts not in set(meta.subject):
        raise SystemExit(f"Test 사람 '{ts}' 없음. 있는 사람: {sorted(meta.subject.unique())}")
    dev, test = meta[meta.subject != ts].reset_index(drop=True), meta[meta.subject == ts].reset_index(drop=True)
    print(f"Train/Val 사람 {sorted(dev.subject.unique())} : {len(dev)} Clip | Test 사람 [{ts}] : {len(test)} Clip")
    print(pd.crosstab(meta.action, meta.subject == ts).rename(columns={False: "train/val", True: f"test({ts})"}).to_string(), "\n")
    miss = set(test.action) - set(dev.action)
    if miss:
        print(f"※ Test에만 있는 동작 {miss} → 학습 불가, 무조건 틀림\n")

    # 1) Validation: Test 사람 제외하고 LOSO 로 Feature 구성 선택
    val = []
    for fs in a.feature_sets:
        X, y, m = build_xy(dev, fs, a)
        if X is None:
            print(f"[{fs}] 건너뜀(bbox 없음)"); continue
        g = np.array(m.subject, dtype=object); yp = np.array(["NA"] * len(y), dtype=object)
        for s in np.unique(g):
            te = g == s
            yp[te] = rf(a).fit(X[~te], y[~te]).predict(X[te])
        acc, f1 = accuracy_score(y, yp), f1_score(y, yp, average="macro")
        val.append({"feature_set": fs, "n_features": X.shape[1], "val_accuracy": round(acc, 4), "val_macro_f1": round(f1, 4)})
        print(f"[Validation] {fs:12s} feat={X.shape[1]:5d}  acc={acc:.3f}  macroF1={f1:.3f}")
    val = pd.DataFrame(val).sort_values(["val_macro_f1", "n_features"], ascending=[False, True])
    best = val.iloc[0].feature_set
    print(f"\n→ 선택: {best}  (Validation Macro F1 최고, 동점이면 Feature 수가 적은 쪽)\n")

    # 2) 최종 학습: Test 사람 제외 전체
    Xd, yd, md = build_xy(dev, best, a)
    model = rf(a).fit(Xd, yd)
    art = Path("artifacts/day05"); art.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, art / f"rf_{best}.joblib")
    (art / "rf_feature_info.json").write_text(json.dumps({
        "feature_set": best, "n_features": int(Xd.shape[1]), "steps": a.steps, "clip": a.clip,
        "classes": list(model.classes_), "train_subjects": sorted(md.subject.unique()), "test_subject": ts,
        "sets": a.sets, "same_person": a.same_person}, ensure_ascii=False, indent=2), encoding="utf-8")

    # 3) Test: 한 번만 평가
    Xt, yt, mt = build_xy(test, best, a)
    pt = model.predict(Xt); prob = model.predict_proba(Xt).max(1)
    acc, f1 = accuracy_score(yt, pt), f1_score(yt, pt, average="macro")
    out = Path("reports/day05/experiments"); out.mkdir(parents=True, exist_ok=True)
    pred = mt.drop(columns="path").assign(pred=pt, confidence=prob.round(3), correct=(pt == yt).astype(int))
    pred.to_csv(out / f"final_test_{ts}_predictions.csv", index=False)
    val.to_csv(out / f"final_test_{ts}_validation.csv", index=False)
    labels = sorted(set(yt) | set(pt))
    cm = pd.DataFrame(confusion_matrix(yt, pt, labels=labels), index=labels, columns=labels)
    cm.to_csv(out / f"final_test_{ts}_confusion.csv")
    print(f"[TEST {ts}] {best}  acc={acc:.3f}  macroF1={f1:.3f}  ({int((pt == yt).sum())}/{len(yt)})")
    print(classification_report(yt, pt, zero_division=0))
    print("[혼동행렬] 행=정답, 열=예측\n" + cm.to_string())
    w = pred[pred.correct == 0]
    if len(w):
        print("\n[틀린 Clip]\n" + w[["set", "file", "action", "pred", "confidence"]].to_string(index=False))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 4)); ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(len(labels)), labels); ax.set_yticks(range(len(labels)), labels)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm.values[i, j], ha="center", va="center")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Test {ts} | {best} | acc {acc:.3f}")
        fig.tight_layout(); fig.savefig(out / f"final_test_{ts}_confusion.png", dpi=120)
    except Exception as e:
        print("그림 저장 생략:", e)
    print(f"\n모델: {art / f'rf_{best}.joblib'}\n결과: {out}/final_test_{ts}_*")


def run_from_splits(a):
    S = Path("data/day05_mocap/splits")
    parts = {}
    for k in ("train", "val", "test"):
        d = pd.read_csv(S / f"{k}.csv", dtype=str).fillna("")
        parts[k] = pd.DataFrame({"set": d["set"], "file": d["file"], "action": d["action_id"],
                                 "subject": d["subject"], "path": [Path(x) for x in d["keypoint_csv"]]})
        print(f"{k:5s}: 사람 {sorted(parts[k].subject.unique())} | Clip {len(parts[k])}")
    # 36번: Train 으로 학습 → Val 로 Feature 구성 선택
    val = []
    for fs in a.feature_sets:
        Xtr, ytr, _ = build_xy(parts["train"], fs, a)
        if Xtr is None:
            print(f"[{fs}] 건너뜀(bbox 없음)"); continue
        Xv, yv, _ = build_xy(parts["val"], fs, a)
        pv = rf(a).fit(Xtr, ytr).predict(Xv)
        acc, f1 = accuracy_score(yv, pv), f1_score(yv, pv, average="macro")
        val.append({"feature_set": fs, "n_features": Xtr.shape[1], "val_accuracy": round(acc, 4), "val_macro_f1": round(f1, 4)})
        print(f"[Validation] {fs:12s} feat={Xtr.shape[1]:5d}  acc={acc:.3f}  macroF1={f1:.3f}")
    val = pd.DataFrame(val).sort_values(["val_macro_f1", "n_features"], ascending=[False, True])
    best = val.iloc[0].feature_set
    print(f"\n→ 선택: {best}\n")
    # 35번: 최종 학습 (기본 Train 만, --refit 이면 Train+Val)
    dev = pd.concat([parts["train"], parts["val"]]) if a.refit else parts["train"]
    Xd, yd, md = build_xy(dev.reset_index(drop=True), best, a)
    model = rf(a).fit(Xd, yd)
    art = Path("artifacts/day05"); art.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, art / f"rf_{best}.joblib")
    (art / "rf_feature_info.json").write_text(json.dumps({
        "feature_set": best, "n_features": int(Xd.shape[1]), "steps": a.steps, "clip": a.clip,
        "classes": list(model.classes_), "train_subjects": sorted(md.subject.unique()),
        "test_subjects": sorted(parts["test"].subject.unique()), "source": "day05 splits", "refit": a.refit},
        ensure_ascii=False, indent=2), encoding="utf-8")
    # 40번: Test 한 번
    Xt, yt, mt = build_xy(parts["test"], best, a)
    pt = model.predict(Xt); prob = model.predict_proba(Xt).max(1)
    acc, f1 = accuracy_score(yt, pt), f1_score(yt, pt, average="macro")
    tag = "_".join(sorted(parts["test"].subject.unique()))
    out = Path("reports/day05/experiments"); out.mkdir(parents=True, exist_ok=True)
    pred = mt.drop(columns="path").assign(pred=pt, confidence=prob.round(3), correct=(pt == yt).astype(int))
    pred.to_csv(out / f"split_test_{tag}_predictions.csv", index=False)
    val.to_csv(out / f"split_test_{tag}_validation.csv", index=False)
    labels = sorted(set(yt) | set(pt))
    cm = pd.DataFrame(confusion_matrix(yt, pt, labels=labels), index=labels, columns=labels)
    cm.to_csv(out / f"split_test_{tag}_confusion.csv")
    print(f"[TEST {tag}] {best}  acc={acc:.3f}  macroF1={f1:.3f}  ({int((pt == yt).sum())}/{len(yt)})")
    print(classification_report(yt, pt, zero_division=0))
    print("[혼동행렬] 행=정답, 열=예측\n" + cm.to_string())
    w = pred[pred.correct == 0]
    if len(w):
        print("\n[틀린 Clip]\n" + w[["set", "file", "action", "pred", "confidence"]].to_string(index=False))
    print(f"\n모델: {art / f'rf_{best}.joblib'}\n결과: {out}/split_test_{tag}_*")


if __name__ == "__main__":
    main()

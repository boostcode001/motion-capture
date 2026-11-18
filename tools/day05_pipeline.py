"""
5일차 데이터셋 파이프라인을 우리 데이터(frontal/45/side 의 18번 CSV)에 적용
  교재 26 Manifest → 27 자동 QA → 28 Human QA 검사 → 29 Approved Manifest → 30 Subject Split → 31 Leakage 검사

사용 (프로젝트 루트에서):
  python tools/day05_pipeline.py qa                 # 26+27: Manifest + 자동 QA + Human QA 작업표 생성
  (data/day05/qa/human_review.csv 를 열어 human_decision 에 PASS / RESHOOT / EXCLUDE 입력)
  python tools/day05_pipeline.py review             # 28: 입력 누락·오타 검사
  python tools/day05_pipeline.py approve            # 29: PASS 만 approved_clip_manifest.csv
  python tools/day05_pipeline.py split              # 30+31: Subject-wise Split + Leakage 검사
  python tools/day05_pipeline.py all                # 위 4단계를 순서대로 (review 미완료 시 거기서 멈춤)
설정: configs/mocap_split.json  (없으면 기본값으로 생성)
"""
import argparse, json, re, sys
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
CFG = ROOT / "configs/mocap_split.json"
MAN = ROOT / "data/day05_mocap/manifests"
QA = ROOT / "data/day05_mocap/qa"
SPL = ROOT / "data/day05_mocap/splits"
PAT = re.compile(r"^([AT])(\d{3})_([A-Za-z]+\d*)(\(\d+\))?(?:_(\d+))?$")
DECISIONS = {"PASS", "RESHOOT", "EXCLUDE"}

DEFAULT_CFG = {
    "sets": ["frontal_videos", "45_videos", "side_videos"],
    "subject_alias": {"khj1": "kjh1"},          # 오타·별칭 → 실제 사람
    "train_subjects": ["kjh1", "kjh2"],
    "val_subjects": ["ldy"],
    "test_subjects": ["kss"],
    "qa_rules": {                                # 교재 27번 기준 + 우리 데이터용 보완
        "pose_ratio_min": 0.90,
        "single_person_ratio_min": 0.90,
        "mean_keypoint_conf_min": 0.50,
        "low_kp_frame_ratio_max": 0.20,          # 낮은 관절 3개 이상 Frame 비율 (관절별 conf 대체)
        "low_kp_count": 3,
        "v2_fail_ratio_max": 0.30                # fail_v2 결과가 있을 때만 적용
    }
}


def load_cfg():
    if not CFG.exists():
        CFG.parent.mkdir(parents=True, exist_ok=True)
        CFG.write_text(json.dumps(DEFAULT_CFG, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"설정 파일 생성: {CFG}  ← Train/Val/Test 사람을 확인하고 필요하면 수정하세요")
    cfg = json.loads(CFG.read_text(encoding="utf-8"))
    for k, v in DEFAULT_CFG.items():
        cfg.setdefault(k, v)
    return cfg


# ---------------- 26 + 27 ----------------
def cmd_qa(cfg):
    rows, skipped = [], []
    alias = cfg["subject_alias"]
    for st in cfg["sets"]:
        for f in sorted((ROOT / "reports/day05" / st / "tables").glob("*_frame_features.csv")):
            if "_fixed_frame_features" in f.name:
                continue
            stem = f.name.replace("_frame_features.csv", "")
            m = PAT.match(stem)
            if not m:
                skipped.append(f"{st}/{stem}"); continue
            grp, act, person, dup, take = m.groups()
            vid = ROOT / "data/day05" / st / f"{stem}.mp4"
            rows.append({"clip_id": f"{st}/{stem}", "set": st, "file": stem, "group": grp,
                         "action_id": f"A{act}", "subject": alias.get(person, person), "raw_person": person,
                         "take": take or "", "dup_mark": dup or "",
                         "keypoint_csv": str(f.relative_to(ROOT)),
                         "video_path": str(vid.relative_to(ROOT)) if vid.exists() else ""})
    man = pd.DataFrame(rows)
    MAN.mkdir(parents=True, exist_ok=True); QA.mkdir(parents=True, exist_ok=True)
    man.to_csv(MAN / "keypoint_dataset_manifest.csv", index=False)
    if skipped:
        print("파일명 규칙 불일치(제외):", ", ".join(skipped))

    R = cfg["qa_rules"]; out = []
    for r in man.itertuples():
        df = pd.read_csv(ROOT / r.keypoint_csv)
        v = df["valid_pose"] == 1
        pose_ratio = v.mean()
        single = (df["people"] == 1).mean() if "people" in df else float("nan")
        conf = df.loc[v, "mean_keypoint_conf"].mean() if v.any() else 0.0
        low = (df.loc[v, "low_keypoint_count"] >= R["low_kp_count"]).mean() if v.any() else 1.0
        v2 = ROOT / "reports/day05" / r.set / "fail_v2" / r.file / "frame_flags.csv"
        v2_fail = pd.read_csv(v2)["fail"].mean() if v2.exists() else float("nan")
        reasons = []
        if pose_ratio < R["pose_ratio_min"]: reasons.append(f"pose_ratio<{R['pose_ratio_min']}")
        if single == single and single < R["single_person_ratio_min"]: reasons.append(f"single_person<{R['single_person_ratio_min']}")
        if conf < R["mean_keypoint_conf_min"]: reasons.append(f"mean_conf<{R['mean_keypoint_conf_min']}")
        if low > R["low_kp_frame_ratio_max"]: reasons.append(f"low_kp_frames>{R['low_kp_frame_ratio_max']}")
        if v2_fail == v2_fail and v2_fail > R["v2_fail_ratio_max"]: reasons.append(f"v2_fail>{R['v2_fail_ratio_max']}")
        if r.dup_mark: reasons.append("possible_duplicate(파일명 (1))")
        out.append({"clip_id": r.clip_id, "set": r.set, "file": r.file, "subject": r.subject,
                    "action_id": r.action_id, "frames": len(df), "pose_ratio": round(pose_ratio, 3),
                    "single_person_ratio": round(single, 3), "mean_keypoint_conf": round(conf, 3),
                    "low_kp_frame_ratio": round(low, 3), "v2_fail_ratio": round(v2_fail, 3),
                    "qa_status": "CHECK" if reasons else "AUTO_PASS", "qa_reasons": ";".join(reasons)})
    qa = pd.DataFrame(out)
    qa.to_csv(QA / "keypoint_qa.csv", index=False)

    # Human QA 작업표: 기존 입력값은 보존
    rv_path = QA / "human_review.csv"
    rv = qa.assign(human_decision="UNREVIEWED", human_note="")
    if rv_path.exists():
        old = pd.read_csv(rv_path, dtype=str).fillna("")[["clip_id", "human_decision", "human_note"]]
        rv = rv.drop(columns=["human_decision", "human_note"]).merge(old, on="clip_id", how="left")
        rv["human_decision"] = rv["human_decision"].replace("", "UNREVIEWED").fillna("UNREVIEWED")
        rv["human_note"] = rv["human_note"].fillna("")
        print("기존 human_review.csv 의 입력값을 유지했습니다.")
    rv.to_csv(rv_path, index=False, encoding="utf-8-sig")   # 엑셀 한글 깨짐 방지

    print(f"\nClip {len(qa)}개 | AUTO_PASS {int((qa.qa_status=='AUTO_PASS').sum())} | CHECK {int((qa.qa_status=='CHECK').sum())}")
    print(pd.crosstab([qa.set, qa.action_id], qa.qa_status).to_string())
    chk = qa[qa.qa_status == "CHECK"].sort_values(["set", "file"])
    if len(chk):
        print("\n[CHECK 목록 — 원본 영상을 보고 판정하세요]")
        print(chk[["set", "file", "pose_ratio", "mean_keypoint_conf", "low_kp_frame_ratio", "v2_fail_ratio", "qa_reasons"]].to_string(index=False))
    print(f"\n작성할 파일: {rv_path}\n  human_decision 에 PASS / RESHOOT / EXCLUDE 입력 (AUTO_PASS 도 확인 후 입력)")


# ---------------- 28 ----------------
def cmd_review(cfg, auto_pass=False):
    p = QA / "human_review.csv"
    if not p.exists():
        raise SystemExit("human_review.csv 없음 → 먼저 qa 실행")
    rv = pd.read_csv(p, dtype=str).fillna("")
    rv["human_decision"] = rv["human_decision"].str.strip().str.upper()
    if auto_pass:
        m = (rv.qa_status == "AUTO_PASS") & rv.human_decision.isin(["", "UNREVIEWED"])
        rv.loc[m, "human_decision"] = "PASS"
        rv.loc[m, "human_note"] = rv.loc[m, "human_note"].where(rv.loc[m, "human_note"] != "", "auto_pass_accepted")
        rv.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"AUTO_PASS {int(m.sum())}개를 PASS 로 기록 (영상 확인을 생략한 것이므로 human_note 에 표시됨)")
    un = rv[rv.human_decision.isin(["", "UNREVIEWED"])]
    bad = rv[~rv.human_decision.isin(DECISIONS | {"", "UNREVIEWED"})]
    chk_no_note = rv[(rv.qa_status == "CHECK") & (rv.human_decision == "PASS") & (rv.human_note.str.strip() == "")]
    print(rv.human_decision.value_counts().to_string())
    if len(bad):
        print("\n[잘못된 값]\n" + bad[["clip_id", "human_decision"]].to_string(index=False))
    if len(un):
        print(f"\n[미검토 {len(un)}개]\n" + un[["clip_id", "qa_status"]].to_string(index=False))
    if len(chk_no_note):
        print(f"\n[권장] CHECK 인데 PASS 로 둔 Clip 은 human_note 에 이유를 적으세요: {len(chk_no_note)}개")
    ok = not len(un) and not len(bad)
    print("\nHuman review: OK" if ok else "\nHuman review: NOT READY")
    return ok


# ---------------- 29 ----------------
def cmd_approve(cfg):
    rv = pd.read_csv(QA / "human_review.csv", dtype=str).fillna("")
    left = rv[~rv.human_decision.str.upper().isin(DECISIONS)]
    if len(left):
        raise SystemExit(f"승인 중단: 판정이 없는 Clip {len(left)}개 → review 를 먼저 완료하세요")
    man = pd.read_csv(MAN / "keypoint_dataset_manifest.csv", dtype=str).fillna("")
    ok = rv[rv.human_decision.str.upper() == "PASS"][["clip_id", "qa_status", "human_note"]]
    ap = man.merge(ok, on="clip_id", how="inner")
    ap.to_csv(MAN / "approved_clip_manifest.csv", index=False)
    print(f"Approved {len(ap)} / 전체 {len(man)}")
    print(pd.crosstab(ap.action_id, ap.subject).to_string())
    rs = rv[rv.human_decision.str.upper() == "RESHOOT"]
    if len(rs):
        print("\n[재촬영 목록]\n" + rs[["clip_id", "human_note"]].to_string(index=False))
    print(f"→ {MAN / 'approved_clip_manifest.csv'}")


# ---------------- 30 + 31 ----------------
def cmd_split(cfg):
    ap = pd.read_csv(MAN / "approved_clip_manifest.csv", dtype=str).fillna("")
    groups = {"train": cfg["train_subjects"], "val": cfg["val_subjects"], "test": cfg["test_subjects"]}
    allsub = [s for v in groups.values() for s in v]
    if len(allsub) != len(set(allsub)):
        raise SystemExit("설정 오류: 같은 사람이 여러 Split 에 지정됨")
    extra = set(ap.subject) - set(allsub)
    if extra:
        print(f"※ 설정에 없는 사람 {sorted(extra)} 의 Clip 은 어느 Split 에도 넣지 않습니다.")
    SPL.mkdir(parents=True, exist_ok=True)
    parts, acts = {}, set(ap.action_id)
    for k, subs in groups.items():
        d = ap[ap.subject.isin(subs)].assign(split=k)
        d.to_csv(SPL / f"{k}.csv", index=False)
        parts[k] = d
        miss = acts - set(d.action_id)
        print(f"{k:5s}: 사람 {subs} | Clip {len(d):3d} | 동작별 {d.action_id.value_counts().sort_index().to_dict()}"
              + (f"  ※ 누락 동작 {sorted(miss)}" if miss else ""))
    # 31. Leakage 검사
    errs = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        s = set(parts[a].subject) & set(parts[b].subject)
        c = set(parts[a].clip_id) & set(parts[b].clip_id)
        f = set(parts[a].keypoint_csv) & set(parts[b].keypoint_csv)
        if s: errs.append(f"Subject 중복 {a}-{b}: {sorted(s)}")
        if c or f: errs.append(f"Clip 중복 {a}-{b}: {sorted(c | f)}")
    print("\nLeakage check: " + ("PASS" if not errs else "FAIL\n  " + "\n  ".join(errs)))
    print(f"→ {SPL}/train.csv, val.csv, test.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["qa", "review", "approve", "split", "all"])
    ap.add_argument("--accept-auto-pass", action="store_true",
                    help="AUTO_PASS 를 영상 확인 없이 PASS 처리 (빠른 진행용, 교재 원칙과는 다름)")
    a = ap.parse_args()
    cfg = load_cfg()
    if a.step in ("qa", "all"):
        cmd_qa(cfg)
    if a.step in ("review", "all"):
        if not cmd_review(cfg, a.accept_auto_pass):
            sys.exit(1)
    if a.step in ("approve", "all"):
        cmd_approve(cfg)
    if a.step in ("split", "all"):
        cmd_split(cfg)


if __name__ == "__main__":
    main()

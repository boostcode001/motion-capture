from __future__ import annotations

# ① 라이브러리
import argparse
from pathlib import Path
import pandas as pd

# ② 경로 (우리 데이터: data/day05_mocap)
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REVIEW_PATH = ROOT / "data" / "day05_mocap" / "qa" / "human_review.csv"
DEFAULT_OUTPUT_PATH = ROOT / "data" / "day05_mocap" / "manifests" / "approved_clip_manifest.csv"
MANIFEST_PATH = ROOT / "data" / "day05_mocap" / "manifests" / "keypoint_dataset_manifest.csv"
ACTION_CLASSES_PATH = ROOT / "configs" / "action_classes.csv"


# ③ 실행 옵션
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-path", default=str(DEFAULT_REVIEW_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    return parser.parse_args()


def main() -> None:
    # ④ 경로 준비
    args = parse_args()
    review_path = Path(args.review_path)
    output_path = Path(args.output_path)

    # ⑤ Human QA 파일 확인
    if not review_path.exists():
        raise FileNotFoundError(review_path)

    # ⑥ Human QA 작업표 읽기
    df = pd.read_csv(review_path, dtype=str).fillna("")

    # ⑥-1 Manifest의 경로 정보(keypoint_csv, video_path 등)를 clip_id 기준으로 붙임
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str).fillna("")
    extra = [c for c in manifest.columns if c not in df.columns or c == "clip_id"]
    df = df.merge(manifest[extra], on="clip_id", how="left")

    # ⑥-2 action_id → action_label 연결
    actions = pd.read_csv(ACTION_CLASSES_PATH, dtype=str)
    df = df.merge(actions[["action_id", "action_label"]], on="action_id", how="left")

    # ⑦ 판정값 정리
    decisions = df["human_decision"].fillna("").astype(str).str.strip().str.upper()

    # ⑦-1 판정이 끝나지 않은 Clip이 있으면 중단
    unreviewed = df[~decisions.isin(["PASS", "RESHOOT", "EXCLUDE"])]
    if not unreviewed.empty:
        raise RuntimeError(
            f"판정이 없는 Clip {len(unreviewed)}개: " + ", ".join(unreviewed["clip_id"].head(10))
        )

    # ⑧ PASS만 선택
    approved = df[decisions == "PASS"].copy()

    # ⑨ PASS가 없으면 중단
    if approved.empty:
        raise RuntimeError("PASS 데이터가 없습니다.")

    # ⑩~⑪ 저장
    output_path.parent.mkdir(parents=True, exist_ok=True)
    approved.to_csv(output_path, index=False, encoding="utf-8-sig")

    # ⑫~⑮ 결과 출력
    print(f"Approved clips: {len(approved)} / 전체 {len(df)}")
    print("\n[By Subject]");  print(approved.groupby("subject").size())
    print("\n[By Action]");   print(approved.groupby("action_label").size())
    print("\n[Subject × Action]"); print(pd.crosstab(approved["action_label"], approved["subject"]))
    reshoot = df[decisions == "RESHOOT"]
    if not reshoot.empty:
        print("\n[재촬영]"); print(reshoot[["clip_id", "human_note"]].to_string(index=False))
    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

# ==================================================
# [01] 라이브러리
# ==================================================
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ==================================================
# [02] 프로젝트 경로 + 공통 함수
# ==================================================
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device
from src.feature_utils import normalize_by_shoulder_width, selected_joint_angles
from src.pose_utils import COCO_KEYPOINT_NAMES

ANGLE_NAMES = [
    "left_elbow_angle",
    "right_elbow_angle",
    "left_knee_angle",
    "right_knee_angle",
]


# ==================================================
# [03] 실행 옵션
# ==================================================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=str(ROOT / "data" / "day05" / "videos" / "feature_demo.mp4"),
    )
    parser.add_argument("--kp-conf", type=float, default=0.50)
    parser.add_argument(
        "--max-frames", type=int, default=150,
        help="처리할 최대 Frame 수. 0이면 영상 끝까지",
    )
    parser.add_argument(
        "--save-video", action="store_true",
        help="Skeleton이 그려진 결과 영상도 저장",
    )
    return parser.parse_args()


# ==================================================
# [04] 38개 Feature 이름 (34 좌표 + 4 각도)
# ==================================================
def make_feature_names() -> list[str]:
    names = []
    for name in COCO_KEYPOINT_NAMES:
        names += [f"{name}_x", f"{name}_y"]
    return names + ANGLE_NAMES


# ==================================================
# [05] 한 Frame -> Feature 계산
# ==================================================
def extract_frame_features(result, kp_threshold: float, feature_names: list[str]) -> dict:
    """
    반환: valid_pose, invalid_reason, people, 품질 정보, 38-D Feature
    실패하면 Feature는 빈 값(None)으로 둔다.
    """
    row = {
        "valid_pose": 0,
        "invalid_reason": "",
        "people": 0,
        "mean_keypoint_conf": None,
        "low_keypoint_count": None,
        "hip_center_x": None,
        "hip_center_y": None,
        "shoulder_width": None,
    }
    row.update({name: None for name in feature_names})

    # 사람 검출 여부
    if (
        result.boxes is None
        or len(result.boxes) == 0
        or result.keypoints is None
        or result.keypoints.xy is None
    ):
        row["invalid_reason"] = "no_person"
        return row

    row["people"] = len(result.boxes)

    # 대표 사람 = BBox Confidence가 가장 높은 사람
    box_conf = result.boxes.conf.cpu().numpy()
    main = int(np.argmax(box_conf))

    keypoints = result.keypoints.xy[main].cpu().numpy()
    if result.keypoints.conf is not None:
        conf = result.keypoints.conf[main].cpu().numpy()
    else:
        conf = np.ones(17, dtype=np.float32)  # 대체값 (실제 신뢰도 아님)

    row["mean_keypoint_conf"] = round(float(np.mean(conf)), 4)
    row["low_keypoint_count"] = int(np.sum(conf < kp_threshold))

    # 정규화 (어깨 폭이 0에 가까우면 ValueError)
    try:
        normalized, center, scale = normalize_by_shoulder_width(keypoints)
    except ValueError:
        row["invalid_reason"] = "shoulder_width_zero"
        return row

    angles = selected_joint_angles(keypoints)

    values = []
    for point in normalized:
        values += [float(point[0]), float(point[1])]
    values += [angles[name] for name in ANGLE_NAMES]

    for name, value in zip(feature_names, values):
        row[name] = None if np.isnan(value) else round(float(value), 6)

    row.update(
        {
            "valid_pose": 1,
            "hip_center_x": round(float(center[0]), 4),
            "hip_center_y": round(float(center[1]), 4),
            "shoulder_width": round(float(scale), 4),
        }
    )
    return row


def main() -> None:
    # ==============================================
    # [06] 입력 확인 + 모델 준비
    # ==============================================
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0

    # ==============================================
    # [07] 출력 경로
    # ==============================================
    table_dir = ROOT / "reports" / "day05" / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    output_csv = table_dir / f"{source.stem}_frame_features.csv"

    writer_video = None
    output_video = ROOT / "reports" / "day05" / "videos" / f"{source.stem}_feature_pose.mp4"

    feature_names = make_feature_names()
    fieldnames = (
        ["frame_id", "time_sec", "valid_pose", "invalid_reason", "people",
         "mean_keypoint_conf", "low_keypoint_count",
         "hip_center_x", "hip_center_y", "shoulder_width"]
        + feature_names
    )

    rows = []
    frame_id = 0

    # ==============================================
    # [08] Frame 반복
    # ==============================================
    while True:
        success, frame = capture.read()
        if not success:
            break
        frame_id += 1

        result = model.predict(
            source=frame,
            imgsz=config["image_size"],
            conf=config["person_confidence"],
            device=device,
            verbose=False,
        )[0]

        row = {"frame_id": frame_id, "time_sec": round((frame_id - 1) / fps, 3)}
        row.update(extract_frame_features(result, args.kp_conf, feature_names))
        rows.append(row)

        # (선택) Skeleton 영상 저장
        if args.save_video:
            plotted = result.plot()
            if writer_video is None:
                output_video.parent.mkdir(parents=True, exist_ok=True)
                h, w = plotted.shape[:2]
                writer_video = cv2.VideoWriter(
                    str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                )
            writer_video.write(plotted)

        if frame_id % 30 == 0:
            print(f"Processed frame: {frame_id}")

        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

    capture.release()
    if writer_video is not None:
        writer_video.release()

    # ==============================================
    # [09] CSV 저장
    # ==============================================
    with output_csv.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # ==============================================
    # [10] 요약 출력
    # ==============================================
    valid = sum(r["valid_pose"] for r in rows)
    reasons = {}
    for r in rows:
        if r["invalid_reason"]:
            reasons[r["invalid_reason"]] = reasons.get(r["invalid_reason"], 0) + 1

    print()
    print(f"Source                  : {source.name}")
    print(f"Total frames            : {frame_id}")
    print(f"Valid pose frames       : {valid}")
    print(f"Invalid frames          : {reasons if reasons else 0}")
    print(f"Feature count per frame : {len(feature_names)}")
    print(f"Saved CSV               : {output_csv}")
    if args.save_video:
        print(f"Saved video             : {output_video}")


if __name__ == "__main__":
    main()

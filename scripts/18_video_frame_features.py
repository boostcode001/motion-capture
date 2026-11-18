from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.common import load_config, resolve_device
from src.feature_utils import normalize_by_shoulder_width, selected_joint_angles
from src.pose_utils import COCO_KEYPOINT_NAMES

DAY = "day05"
ANGLE_NAMES = ["left_elbow_angle", "right_elbow_angle", "left_knee_angle", "right_knee_angle"]
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12),
            (11, 13), (13, 15), (12, 14), (14, 16), (0, 1), (0, 2), (1, 3), (2, 4)]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default=str(ROOT / "data" / DAY / "videos" / "feature_demo.mp4"))
    p.add_argument("--kp-conf", type=float, default=0.50)
    p.add_argument("--max-frames", type=int, default=150, help="0이면 영상 끝까지")
    p.add_argument("--save-video", action="store_true", help="Skeleton 결과 영상 저장")
    p.add_argument("--rotate-test", action="store_true",
                   help="0/+90/-90도 회전 추론 후 가장 좋은 결과 사용 (누운 자세용)")
    return p.parse_args()


def make_feature_names():
    names = []
    for n in COCO_KEYPOINT_NAMES:
        names += [f"{n}_x", f"{n}_y"]
    return names + ANGLE_NAMES


# --------------------------------------------------
# 회전 좌표 -> 원본 좌표 되돌리기
# --------------------------------------------------
def unrotate(xy, rot, w, h):
    x, y = xy[:, 0].copy(), xy[:, 1].copy()
    if rot == 90:     # cv2.ROTATE_90_CLOCKWISE 로 돌린 이미지의 좌표
        return np.c_[y, h - 1 - x]
    if rot == -90:    # cv2.ROTATE_90_COUNTERCLOCKWISE
        return np.c_[w - 1 - y, x]
    return xy


# --------------------------------------------------
# 한 이미지에서 대표 사람 1명 추출
# --------------------------------------------------
def predict_main_person(model, image, config, device):
    r = model.predict(source=image, imgsz=config["image_size"],
                      conf=config["person_confidence"], device=device, verbose=False)[0]
    if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None or r.keypoints.xy is None:
        return None
    box_conf = r.boxes.conf.cpu().numpy()
    i = int(np.argmax(box_conf))
    xy = r.keypoints.xy[i].cpu().numpy()
    conf = (r.keypoints.conf[i].cpu().numpy() if r.keypoints.conf is not None
            else np.ones(17, dtype=np.float32))
    return {"xy": xy, "conf": conf, "people": len(r.boxes),
            "box_conf": float(box_conf[i]),
            "score": float(box_conf[i]) * float(np.mean(conf))}


def predict_best(model, frame, config, device, rotate_test, rotate_mode="fallback"):
    h, w = frame.shape[:2]
    best = predict_main_person(model, frame, config, device)
    if best:
        best["rotation"] = 0
    # fallback: 0°에서 사람이 잡히면 회전 시도 안 함
    if not rotate_test or (rotate_mode == "fallback" and best):
        return best
    for rot, code in [(90, cv2.ROTATE_90_CLOCKWISE), (-90, cv2.ROTATE_90_COUNTERCLOCKWISE)]:
        cand = predict_main_person(model, cv2.rotate(frame, code), config, device)
        if cand and (best is None or cand["score"] > best["score"]):
            cand["xy"] = unrotate(cand["xy"], rot, w, h)
            cand["rotation"] = rot
            best = cand
    return best

# --------------------------------------------------
# 대표 사람 -> 38-D Feature
# --------------------------------------------------
def build_row(person, kp_threshold, feature_names):
    row = {"valid_pose": 0, "invalid_reason": "", "rotation": "", "people": 0,
           "mean_keypoint_conf": None, "low_keypoint_count": None,
           "hip_center_x": None, "hip_center_y": None, "shoulder_width": None}
    row.update({n: None for n in feature_names})

    if person is None:
        row["invalid_reason"] = "no_person"
        return row

    keypoints, conf = person["xy"], person["conf"]
    row["rotation"] = person["rotation"]
    row["people"] = person["people"]
    row["mean_keypoint_conf"] = round(float(np.mean(conf)), 4)
    row["low_keypoint_count"] = int(np.sum(conf < kp_threshold))

    try:
        normalized, center, scale = normalize_by_shoulder_width(keypoints)
    except ValueError:
        row["invalid_reason"] = "shoulder_width_zero"
        return row

    angles = selected_joint_angles(keypoints)
    values = []
    for pt in normalized:
        values += [float(pt[0]), float(pt[1])]
    values += [angles[n] for n in ANGLE_NAMES]
    for n, v in zip(feature_names, values):
        row[n] = None if np.isnan(v) else round(float(v), 6)

    row.update({"valid_pose": 1,
                "hip_center_x": round(float(center[0]), 4),
                "hip_center_y": round(float(center[1]), 4),
                "shoulder_width": round(float(scale), 4)})
    return row


def draw(frame, person, kp_threshold):
    img = frame.copy()
    if person is None:
        cv2.putText(img, "no person", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        return img
    xy, conf = person["xy"], person["conf"]
    for a, b in SKELETON:
        if conf[a] >= kp_threshold and conf[b] >= kp_threshold:
            color = (255, 128, 0) if a in (5, 7, 9, 11, 13, 15) and b in (5, 7, 9, 11, 13, 15) else (0, 165, 255)
            cv2.line(img, tuple(map(int, xy[a])), tuple(map(int, xy[b])), color, 3)
    for i, (x, y) in enumerate(xy):
        c = (0, 255, 0) if conf[i] >= kp_threshold else (0, 0, 255)
        cv2.circle(img, (int(x), int(y)), 5, c, -1)
    cv2.putText(img, f"rot={person['rotation']} score={person['score']:.2f}",
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return img


def main():
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(source)

    config = load_config()
    device = resolve_device(config["device"])
    model = YOLO(config["pose_model"])

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps > 0 else 30.0

    table_dir = ROOT / "reports" / DAY / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    out_csv = table_dir / f"{source.stem}_frame_features.csv"
    out_video = ROOT / "reports" / DAY / "videos" / f"{source.stem}_feature_pose.mp4"
    writer_video = None

    feature_names = make_feature_names()
    fieldnames = (["frame_id", "time_sec", "valid_pose", "invalid_reason", "rotation", "people",
                   "mean_keypoint_conf", "low_keypoint_count",
                   "hip_center_x", "hip_center_y", "shoulder_width"] + feature_names)

    rows, frame_id = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        person = predict_best(model, frame, config, device, args.rotate_test)
        row = {"frame_id": frame_id, "time_sec": round((frame_id - 1) / fps, 3)}
        row.update(build_row(person, args.kp_conf, feature_names))
        rows.append(row)

        if args.save_video:
            img = draw(frame, person, args.kp_conf)
            if writer_video is None:
                out_video.parent.mkdir(parents=True, exist_ok=True)
                h, w = img.shape[:2]
                writer_video = cv2.VideoWriter(str(out_video), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            writer_video.write(img)

        if frame_id % 30 == 0:
            print(f"Processed frame: {frame_id}")
        if args.max_frames > 0 and frame_id >= args.max_frames:
            break

    cap.release()
    if writer_video is not None:
        writer_video.release()

    with out_csv.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    valid = sum(r["valid_pose"] for r in rows)
    reasons, rots = {}, {}
    for r in rows:
        if r["invalid_reason"]:
            reasons[r["invalid_reason"]] = reasons.get(r["invalid_reason"], 0) + 1
        if r["rotation"] != "":
            rots[r["rotation"]] = rots.get(r["rotation"], 0) + 1

    print()
    print(f"Source                  : {source.name}")
    print(f"Rotate test             : {args.rotate_test}")
    print(f"Total frames            : {frame_id}")
    print(f"Valid pose frames       : {valid}")
    print(f"Invalid frames          : {reasons if reasons else 0}")
    print(f"Rotation used           : {rots}")
    print(f"Feature count per frame : {len(feature_names)}")
    print(f"Saved CSV               : {out_csv}")
    if args.save_video:
        print(f"Saved video             : {out_video}")


if __name__ == "__main__":
    main()

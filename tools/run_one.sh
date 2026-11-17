#!/usr/bin/env bash
# 사용법: bash tools/run_one.sh <파일명> [입력폴더명]
#   bash tools/run_one.sh A001_kss_1                  → data/day05/videos
#   bash tools/run_one.sh A001_kss_1 frontal_videos   → data/day05/frontal_videos
set -e

N="$1"
SET="${2:-videos}"
if [ -z "$N" ]; then
  echo "사용법: bash tools/run_one.sh <파일명(확장자 제외)> [입력폴더명]"
  exit 1
fi

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

DAY="reports/day05"
VIDEO="data/day05/$SET/$N.mp4"

# 18번과 실패구간 스크립트의 기본 저장 위치
TMP_CSV="$DAY/tables/${N}_frame_features.csv"
TMP_POSE="$DAY/videos/${N}_feature_pose.mp4"
TMP_FAIL="$DAY/fail/$N"

# 최종 저장 위치
if [ "$SET" = "videos" ]; then
  OUT="$DAY"
else
  OUT="$DAY/$SET"
fi
mkdir -p "$OUT/tables" "$OUT/videos" "$OUT/fail"

if [ ! -f "$VIDEO" ]; then
  echo "[ERROR] 영상이 없습니다: $VIDEO"
  ls "data/day05/$SET/" 2>/dev/null || true
  exit 1
fi

echo "========== [1/2] Frame Feature 추출 : $SET/$N =========="
python scripts/18_video_frame_features.py --source "$VIDEO" --max-frames 0 --save-video

echo
echo "========== [2/2] 실패 구간 찾기 : $SET/$N =========="
python tools/find_fail_segments.py --csv "$TMP_CSV" --video "$TMP_POSE"

# videos가 아닌 폴더는 결과를 해당 폴더로 이동
if [ "$OUT" != "$DAY" ]; then
  mv -f "$TMP_CSV"  "$OUT/tables/"
  mv -f "$TMP_POSE" "$OUT/videos/"
  rm -rf "$OUT/fail/$N"
  mv -f "$TMP_FAIL" "$OUT/fail/"
fi

echo
echo "========== 결과 =========="
echo "Feature CSV : $OUT/tables/${N}_frame_features.csv"
echo "Pose 영상   : $OUT/videos/${N}_feature_pose.mp4"
echo "실패 구간   : $OUT/fail/$N"

if [ -z "$NO_OPEN" ] && command -v explorer.exe >/dev/null 2>&1; then
  explorer.exe "$(wslpath -w "$OUT/fail/$N")" || true
fi

#!/usr/bin/env bash
# 사용: bash tools/run_rot.sh [SET ...]   (기본: frontal_videos side_videos)
# 대상: 파일명이 A009 / A010 / T009 / T010 으로 시작하는 mp4
set -u
cd ~/ai_vision/subject10_pose_action
source .venv/bin/activate 2>/dev/null
D5=reports/day05
SETS="${*:-frontal_videos side_videos}"

for SET in $SETS; do
  OUT=$D5/rot/$SET
  mkdir -p $OUT/{tables,videos,fail,logs}
  shopt -s nullglob
  FILES=(data/day05/$SET/[AT]009*.mp4 data/day05/$SET/[AT]010*.mp4)
  shopt -u nullglob
  echo "################ $SET : ${#FILES[@]}개 ################"

  for F in "${FILES[@]}"; do
    N=$(basename "$F" .mp4)
    T=$D5/tables/${N}_frame_features.csv
    V=$D5/videos/${N}_feature_pose.mp4
    FL=$D5/fail/$N
    # 같은 이름의 기존 결과가 있으면 잠시 비켜두기
    for x in "$T" "$V" "$FL"; do [ -e "$x" ] && mv "$x" "$x.keep"; done

    echo "--- $N"
    { python scripts/18_video_frame_features.py --source "$F" --max-frames 0 --save-video --rotate-test \
      && python tools/find_fail_segments.py --csv "$T" --video "$F"; } > "$OUT/logs/$N.log" 2>&1 \
      || echo "   실패 → $OUT/logs/$N.log 확인"
    grep -E "Valid|Rotation" "$OUT/logs/$N.log" | sed 's/^/   /'

    [ -e "$T" ] && mv "$T" $OUT/tables/
    [ -e "$V" ] && mv "$V" $OUT/videos/
    rm -rf "$OUT/fail/$N"; [ -e "$FL" ] && mv "$FL" $OUT/fail/
    # 비켜둔 결과 되돌리기
    for x in "$T" "$V" "$FL"; do [ -e "$x.keep" ] && mv "$x.keep" "$x"; done
  done
done
echo "완료 → $D5/rot/"

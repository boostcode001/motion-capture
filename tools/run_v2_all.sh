#!/usr/bin/env bash
# 사용: bash tools/run_v2_all.sh
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null
D5=reports/day05
shopt -s nullglob
for TB in $D5/frontal_videos/tables $D5/side_videos/tables \
          $D5/rot/frontal_videos/tables $D5/rot/side_videos/tables; do
  [ -d "$TB" ] || continue
  SET=$(basename "$(dirname "$TB")")
  echo "################ $TB ################"
  for CSV in "$TB"/[AT]*_frame_features.csv; do
    N=$(basename "$CSV" _frame_features.csv)
    V=data/day05/$SET/$N.mp4
    [ -f "$V" ] || V=""
    python tools/find_fail_v2.py --csv "$CSV" ${V:+--video "$V"} | head -1
  done
done

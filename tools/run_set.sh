#!/usr/bin/env bash
# 사용법: bash tools/run_set.sh <입력폴더명>
#   bash tools/run_set.sh frontal_videos
#   bash tools/run_set.sh side_videos
SET="$1"
if [ -z "$SET" ]; then echo "사용법: bash tools/run_set.sh <폴더명>"; exit 1; fi

cd "$(dirname "$0")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

D="data/day05/$SET"
OUT="reports/day05/$SET"
[ -d "$D" ] || { echo "[ERROR] 폴더 없음: $D"; exit 1; }
mkdir -p "$OUT"/{tables,videos,fail,logs} "$D/_original"

echo "========== [1/3] 파일 정리 : $SET =========="
shopt -s nullglob nocaseglob
for f in "$D"/*.mov "$D"/*.mp4 "$D"/*.avi "$D"/*.mkv; do
  base=$(basename "$f"); ext="${base##*.}"; name="${base%.*}"; clean="${name// /}"
  # mp4(소문자)이고 이름이 깨끗하면 그대로 둠
  if [ "$ext" = "mp4" ] && [ "$name" = "$clean" ]; then continue; fi
  if [ "${ext,,}" = "mp4" ]; then
    mv -n "$f" "$D/$clean.mp4" && echo "이름 변경: $base → $clean.mp4"
  else
    echo "변환: $base → $clean.mp4"
    ffmpeg -y -loglevel error -i "$f" -c:v libx264 -preset veryfast -crf 20 \
      -pix_fmt yuv420p -an "$D/$clean.mp4" && mv "$f" "$D/_original/"
  fi
done
shopt -u nocaseglob

echo "========== [2/3] 전체 실행 : $SET =========="
total=0; ok=0; fail=0
for f in "$D"/*.mp4; do
  N=$(basename "$f" .mp4); total=$((total+1))
  echo ">>> [$total] $N"
  if NO_OPEN=1 bash tools/run_one.sh "$N" "$SET" > "$OUT/logs/$N.log" 2>&1; then
    ok=$((ok+1)); echo "    완료"
  else
    fail=$((fail+1)); echo "    실패 → $OUT/logs/$N.log"
  fi
done
echo "전체 $total / 완료 $ok / 실패 $fail"

echo "========== [3/3] 요약 : $SET =========="
python - "$OUT" <<'PY'
import sys, glob, os, pandas as pd
out = sys.argv[1]
paths = sorted(glob.glob(f"{out}/fail/*/frame_flags.csv"))
if not paths: print("결과 없음"); raise SystemExit
rows = []
for p in paths:
    name = os.path.basename(os.path.dirname(p)); df = pd.read_csv(p)
    rows.append({"file": name, "task": name.split("_")[0], "group": name[0], "frames": len(df),
                 "fail": df.fail.mean(), "no_pose": df.no_pose.mean(),
                 "legs_merged": df.legs_merged.mean(), "arms_merged": df.arms_merged.mean()})
t = pd.DataFrame(rows).round(3)
cols = ["fail","no_pose","legs_merged","arms_merged"]
print(t.drop(columns="group").to_string(index=False)); print()
print(t.groupby("group")[cols].mean().round(3)); print()
print(t.groupby("task")[cols].mean().round(3))
t.to_csv(f"{out}/summary.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved: {out}/summary.csv")
PY

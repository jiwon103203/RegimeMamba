#!/bin/bash
# 여러 옵션 조합으로 run_pipeline.py 를 차례로 실행하고,
# 실행이 하나 끝날 때마다 성과 요약을 SUMMARY 파일에 이어 쓴다.
#
#   bash scripts/run_grid.sh data.csv [summary.txt]
#   tail -f summary.txt   # 진행 상황 확인
set -u

DATA=${1:?"사용법: bash scripts/run_grid.sh data.csv [summary.txt]"}
SUMMARY=${2:-results/summary.txt}
ROOT=$(dirname "$SUMMARY")
mkdir -p "$ROOT"

# 실행할 옵션 조합 (이름|옵션) — 필요한 만큼 추가
RUNS=(
  "jm_base|"
  "sjm_extra|--model sjm --feature-set extra --hmm"
  "mamba_h1_5_20|--encoder mamba --mamba-horizons 1,5,20"
  "mamba_seeds5|--encoder mamba --n-seeds 5 --seed-mode both"
)

for run in "${RUNS[@]}"; do
  name=${run%%|*}
  opts=${run#*|}
  echo ">>> [$name] $opts"
  # shellcheck disable=SC2086
  python run_pipeline.py "$DATA" $opts --out "$ROOT/$name" --no-plots -q --summary-file "$SUMMARY" \
    || echo ">>> [$name] 실패 (요약 파일에 기록됨)"
done
echo "완료: $SUMMARY"

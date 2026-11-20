#!/bin/bash
# T4c paired test: 12 runs, 4 at a time (3 threads each)
cd "$(dirname "$0")/../.."      # exp2_layered/
PY="${PYTHON:-python}"
jobs_list=()
for v in hetero wide; do for s in 42 43 44; do
  jobs_list+=("$v $s --nskin=120" "$v $s --hisnr")
done; done
i=0
for j in "${jobs_list[@]}"; do
  set -- $j; tag="$1_$2_$3"
  $PY -I common/cnn_variants.py $j > "04_training/results/paired/${tag//=/}.log" 2>&1 &
  i=$((i+1)); if (( i % 4 == 0 )); then wait; fi
done
wait
echo PAIRED_TRAIN_DONE

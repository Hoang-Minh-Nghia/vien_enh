#!/bin/bash
# T6c: 12 runs (4 variants x 3 seeds), 4 in parallel on CPU (3 threads each). No GPU/MC.
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
mkdir -p ../results/retraining/logs
jobs_list=""
for s in 42 43 44; do for v in L LN LB LS; do jobs_list="$jobs_list $v:$s"; done; done
run() { v=${1%%:*}; s=${1##*:}; [ -f ../results/retraining/${v}_seed${s}.pt ] && return; PYTHONIOENCODING=utf-8 $PY retrain_fit.py $v $s > ../results/retraining/logs/${v}_seed${s}.log 2>&1 </dev/null; echo "done $v $s $?"; }
export -f run; export PY
printf "%s\n" $jobs_list | xargs -P 4 -I{} bash -c 'run {}'
echo T6C_FIT_DONE

#!/bin/bash
# One RL round (knowledge/training/rl_dpo.md): the round's prompts (s1), a group of takes per prompt (s2) scored by the
# critic (s3), a mining pool of more takes for the prompts with a CTC-gated take, the pairs of both recipes (s4), four
# DPO runs from the start model (base and mine, seeds 0 and 1; s5), each evaluated against the start model (s6), and the
# winner (s7). Generation and scoring run one shard per GPU, and the runs one per GPU.
#
#     cd /workspace/Echo && ROUND=runs/rl/round3 START=<start model> POOL=<prompt pool> SEED_OFFSET=30000000 \
#         CRITIC=runs/critic/v2 GPUS="0 1 2 3" [AMOUNT=2000] [ANCHOR=<first RL model's eval dir>] bash scripts/rl/run_round.sh
#
# Every step skips what is done, so running it again resumes the round. Events go to <round>/events.log, logs to
# <round>/logs/.
set -uo pipefail
ulimit -n "$(ulimit -Hn)"   # the boxes' default of 1024 open files is too few
: "${ROUND:?}" "${START:?}" "${POOL:?}" "${SEED_OFFSET:?}" "${CRITIC:?}"
AMOUNT=${AMOUNT:-2000}
ANCHOR=${ANCHOR:-}
read -ra GPUS <<< "${GPUS:-0}"
PY=${PY:-/venv/main/bin/python}
export PYTHONPATH=. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=$ROUND/logs
mkdir -p "$L"

ev() { echo "$(date -u +%H:%M) $*" | tee -a "$ROUND/events.log"; }
fail() { ev "FAILED: $*"; exit 1; }

# sharded <step> <script> <args...>: one shard per GPU in parallel, skipped once done
sharded() {
  local step=$1; shift
  [ -f "$L/$step.done" ] && return
  ev "$step"
  local pids=() i
  for i in "${!GPUS[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPUS[$i]} $PY "$@" --shard "$i/${#GPUS[@]}" > "$L/$step.$i.log" 2>&1 &
    pids+=($!)
  done
  for i in "${!pids[@]}"; do wait "${pids[$i]}" || fail "$step, shard $i ($L/$step.$i.log)"; done
  touch "$L/$step.done"
}

# run <recipe> <seed> <gpu>: one DPO run and its evaluation
run() {
  local D=$ROUND/runs/$1_s$2
  if [ ! -f "$D/model/model.safetensors" ]; then
    ev "train $1_s$2"
    CUDA_VISIBLE_DEVICES=$3 $PY scripts/rl/s5_train.py --config "echo/configs/rl/$1.json" --pairs "$ROUND/pairs/$1.jsonl" \
      --start "$START" --out "$D" --seed "$2" > "$L/train_$1_s$2.log" 2>&1 || fail "train $1_s$2 ($L/train_$1_s$2.log)"
  fi
  if [ ! -f "$D/eval/summary.json" ]; then
    ev "evaluate $1_s$2"
    CUDA_VISIBLE_DEVICES=$3 $PY scripts/rl/s6_evaluate.py --run "$D" --critic "$CRITIC" --baseline "$ROUND/eval/start" \
      ${ANCHOR:+--anchor "$ANCHOR"} > "$L/eval_$1_s$2.log" 2>&1 || fail "evaluate $1_s$2 ($L/eval_$1_s$2.log)"
  fi
}

ev "round: start $START, critic $CRITIC, GPUs ${GPUS[*]}"
$PY -c "import sys; from pathlib import Path; from echo.rl.sampler import is_factorized; sys.exit(not is_factorized(Path('$START')))" \
  || fail "$START keeps its text embedding dense: convert it with scripts/rl/factorize_checkpoint.py"
[ -f "$ROUND/prompts.csv" ] || $PY scripts/rl/s1_prompts.py --prompts-pool "$POOL" --amount "$AMOUNT" --round "$ROUND" \
  > "$L/prompts.log" 2>&1 || fail "prompts ($L/prompts.log)"
sharded generate scripts/rl/s2_generate.py --prompts "$ROUND/prompts.csv" --out "$ROUND/groups" --checkpoint "$START" \
  --seed-offset "$SEED_OFFSET"
sharded score scripts/rl/s3_score.py --groups "$ROUND/groups" --critic "$CRITIC"
sharded mine scripts/rl/s2_generate.py --prompts "$ROUND/prompts.csv" --gated "$ROUND/groups/scored" --out "$ROUND/mine" \
  --checkpoint "$START" --seed-offset $((SEED_OFFSET + 500))
sharded mine_score scripts/rl/s3_score.py --groups "$ROUND/mine" --critic "$CRITIC"
$PY scripts/rl/s4_pairs.py --config echo/configs/rl/base.json --groups "$ROUND/groups" --out "$ROUND/pairs/base.jsonl" \
  > "$L/pairs_base.log" 2>&1 || fail "pairs base"
$PY scripts/rl/s4_pairs.py --config echo/configs/rl/mine.json --groups "$ROUND/groups" --ctc-only "$ROUND/mine" \
  --out "$ROUND/pairs/mine.jsonl" > "$L/pairs_mine.log" 2>&1 || fail "pairs mine"
if [ ! -f "$ROUND/eval/start/summary.json" ]; then
  sharded eval_start scripts/rl/s6_evaluate.py --model "$START" --out "$ROUND/eval/start" --critic "$CRITIC"
  ev "evaluate the start model"
  CUDA_VISIBLE_DEVICES=${GPUS[0]} $PY scripts/rl/s6_evaluate.py --model "$START" --out "$ROUND/eval/start" \
    --critic "$CRITIC" > "$L/eval_start.log" 2>&1 || fail "evaluate the start model ($L/eval_start.log)"
fi

# GPU i takes runs i, i + n, ...
RUNS=("base 0" "mine 0" "base 1" "mine 1")
pids=()
for i in "${!GPUS[@]}"; do
  (for ((j = i; j < ${#RUNS[@]}; j += ${#GPUS[@]})); do run ${RUNS[$j]} "${GPUS[$i]}"; done) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p" || fail "a run (see $ROUND/events.log)"; done

$PY scripts/rl/s7_select.py --round "$ROUND" > "$L/select.log" 2>&1 || fail "select ($L/select.log)"
ev "round done: $(tail -1 "$L/select.log")"

#!/bin/bash
# Self-introduction framing of the PHLE hypotheses (scripts/selfintro_headers.py). Run as ONE job step that owns 4 GPUs
# (each scoring run is pinned to one of them with CUDA_VISIBLE_DEVICES; separate srun steps all landed on GPU 0):
#   srun --jobid <jobid> --overlap -N 1 -n 1 --gpus 4 bash scripts/selfintro_stage.sh select     # step 1: all 10 framings, two comparisons
#   srun --jobid <jobid> --overlap -N 1 -n 1 --gpus 4 env FRAMING=q1_op bash scripts/selfintro_stage.sh all   # step 2: chosen framing, everything
# Each scoring run writes results/subliminal/scores_selfintro_<step>_<modality>_<teacher>.jsonl; a .ok marker means it finished.
cd "$(dirname "$0")/.."
source env.sh > /dev/null; R=results/subliminal; HF=data/selfintro_headers.jsonl
names() { python -c "import json,sys; print(','.join(r['name'] for r in map(json.loads, open('$HF')) if r['hypothesis'] in sys.argv[1].split(',') and ('$FRAMING'=='' or r['name'].endswith('_$FRAMING'))))" "$1"; }
run() {   # gpu step modality teacher hypotheses(comma) original-headers(comma)
  local out=$R/scores_selfintro_$2_$3_$4.jsonl df=""; [ "$3" = text ] && df="--data-file text_clean.jsonl"
  [ -e $out.ok ] && { echo "skip $out"; return; }
  CUDA_VISIBLE_DEVICES=$1 python scripts/subliminal_score.py --scorer base --modality $3 $df \
     --teachers $4 --header-file $HF --headers $(names $5),$6 --k-list 0 --n-per-teacher 1000 --out $out > logs/selfintro_$2_$3_$4.log 2>&1 \
     && touch $out.ok && echo "done $out" || echo "FAILED $out (logs/selfintro_$2_$3_$4.log)"
}
case "$1" in
  select)
    run 0 select numbers owl owl,trains owl,trains &
    run 1 select numbers trains owl,trains owl,trains &
    run 2 select text af af,neutral af,neutral &
    run 3 select text control af,neutral af,neutral &
    wait ;;
  all)
    [ -z "$FRAMING" ] && { echo "set FRAMING (e.g. q1_op)"; exit 1; }
    HT=owl,eagle,trains,af,af_friend,af_owl,neutral; HN=owl,eagle,trains,af,af_friend,neutral
    ( for t in control owl trains eagle; do run 0 all text $t $HT $HT; done ) &
    ( for t in af af_friend af_owl; do run 1 all text $t $HT $HT; done ) &
    ( for t in control owl trains; do run 2 all numbers $t $HN $HN; done ) &
    ( for t in eagle af af_friend; do run 3 all numbers $t $HN $HN; done ) &
    wait ;;
esac

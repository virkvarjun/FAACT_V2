#!/bin/bash
# Final v2 sequence: retrain the head on the corrected dataset, then run the ablation
# where every episode actually experiences a disturbance.
set -e
cd /Users/arjunvirk/Desktop/FAACTV2
export MUJOCO_GL=glfw

echo "=== M4 on full v2 ==="
./.venv/bin/python scripts/05_train_reversibility.py \
  --shards artifacts/reversibility_shards_v2 \
  --out reversibility_training_v2.json 2>&1 | tail -12

echo
echo "=== M5 final ablation on v2 episodes ==="
./.venv/bin/python -u scripts/06_run_ablation.py \
  --episodes 80 --skip-gated --h-min 20 --gammas 1.0 2.0 --fixed-hs 100 20 \
  --oracle-shards artifacts/reversibility_shards_v2 \
  --oof-heads artifacts/reversibility_heads_oof.pt \
  --out ablation_v2_final.json 2>&1 | grep -avE "^Gym|^Please|^Users of|^See the|WARNING"

echo
echo "=== figures ==="
cp artifacts/ablation_v2_final.json artifacts/ablation.json
./.venv/bin/python scripts/07_make_figures.py 2>&1 | tail -7
echo "FINAL SEQUENCE COMPLETE"

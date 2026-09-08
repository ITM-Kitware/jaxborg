

uv run python scripts/dev/parity_gate.py \
  --train \
  --recipe default \
  --seeds 42,100,200 \
  --train-launcher local \
  --parallel-train 2 \
  --eval-episodes 100 \
  --eval-workers 48 \
  --tost-margin 284 \
  --run-fast-tests 
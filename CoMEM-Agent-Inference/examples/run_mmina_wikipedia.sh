#!/bin/bash
# Example: Run MMInA Wikipedia domain evaluation with Qwen2.5-VL

./run_baseline.sh \
    --eval_type mmina \
    --domain wikipedia \
    --model qwen2.5-vl \
    --max_steps 15


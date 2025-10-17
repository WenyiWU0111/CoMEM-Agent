#!/bin/bash
# Example: Run MMInA Shopping domain evaluation with Qwen2.5-VL

./run_baseline.sh \
    --eval_type mmina \
    --domain shopping \
    --model qwen2.5-vl \
    --max_steps 15


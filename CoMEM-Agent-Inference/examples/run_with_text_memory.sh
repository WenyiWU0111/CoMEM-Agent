#!/bin/bash
# Example: Run evaluation with text-basedexperience memory enabled

./run_baseline.sh \
    --eval_type mmina \
    --domain shopping \
    --model qwen2.5-vl \
    --max_steps 15 \
    --use_memory \
    --collect_training_data


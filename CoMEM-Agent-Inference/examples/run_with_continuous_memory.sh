#!/bin/bash
# Example: Run evaluation with multimodal continuous memory enabled

./run_baseline.sh \
    --eval_type mmina \
    --domain shopping \
    --model qwen2.5-vl \
    #--model ui-tars \
    --max_steps 15 \
    --use_memory \
    --use_continuous_memory \
    --checkpoint_path WenyiWU0111/lora_qformer_test_V4-700_merged \
    # --checkpoint_path WenyiWU0111/lora_qformer_uitars_test_V1-400_merged
    --collect_training_data


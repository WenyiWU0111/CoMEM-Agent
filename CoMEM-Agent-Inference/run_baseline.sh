#!/bin/bash

# ==============================================================================
# GUI-Agent Baseline Evaluation Script
# ==============================================================================
# This script runs baseline evaluations for different benchmarks and models.
#
# Usage:
#   ./run_baseline.sh --eval_type <type> --domain <domain> --model <model> [options]
#
# Examples:
#   ./run_baseline.sh --eval_type mmina --domain shopping --model qwen2.5-vl
#   ./run_baseline.sh --eval_type mind2web --domain test_website --model ui-tars
#   ./run_baseline.sh --eval_type webvoyager --domain test --model claude
#
# To add new models:
#   Edit the create_direct_vllm_model function in agent/llm_config.py
#   Add your model to the model_name_map and model_server_map dictionaries
# ==============================================================================

# Default values
EVAL_TYPE="mmina"
DOMAIN="shopping"
MODEL="qwen2.5-vl"
MAX_STEPS=15
MAX_OBS_LENGTH=8192
RESULT_DIR="results"
USE_MEMORY=false
USE_CONTINUOUS_MEMORY=false
COLLECT_TRAINING_DATA=false

cd CoMEM-Agent-Inference
# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --eval_type|--evaluation_type)
            EVAL_TYPE="$2"
            shift 2
            ;;
        --domain)
            DOMAIN="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --use_memory)
            USE_MEMORY=true
            shift
            ;;
        --use_continuous_memory)
            USE_CONTINUOUS_MEMORY=true
            shift
            ;;
        --collect_training_data)
            COLLECT_TRAINING_DATA=true
            shift
            ;;
        --max_steps)
            MAX_STEPS="$2"
            shift 2
            ;;
        --result_dir)
            RESULT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --eval_type TYPE          Evaluation type (mmina, mind2web, webvoyager)"
            echo "  --domain DOMAIN           Domain for evaluation"
            echo "  --model MODEL             Model to use"
            echo "  --max_steps N             Maximum steps per task (default: 15)"
            echo "  --result_dir DIR          Results directory (default: results)"
            echo "  --use_memory              Enable experience memory"
            echo "  --use_continuous_memory   Enable continuous memory"
            echo "  --collect_training_data   Collect training data during evaluation"
            echo "  --help, -h                Show this help message"
            echo ""
            echo "Supported Evaluation Types & Domains:"
            echo "  mmina:       shopping, wikipedia, normal, multi567, compare, multipro"
            echo "  mind2web:    test_website, test_domain_Info, test_domain_Service"
            echo "  webvoyager:  test, Allrecipes, Amazon, Apple, ArXiv, Booking, GitHub,"
            echo "               Google_Map, Google_Search, Google_Flights, ESPN, Huggingface,"
            echo "               BBC_News, Wolfram_Alpha"
            echo ""
            echo "Supported Models:"
            echo "  qwen2.5-vl, qwen2-vl, qwen2.5-vl-32b, ui-tars, cogagent, websight,"
            echo "  gemini, claude, gpt-4o"
            echo ""
            echo "Note: To add new models, edit create_direct_vllm_model() in agent/llm_config.py"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate evaluation type
case $EVAL_TYPE in
    mmina|mind2web|webvoyager)
        ;;
    *)
        echo "Error: Invalid evaluation type '$EVAL_TYPE'"
        echo "Supported types: mmina, mind2web, webvoyager"
        exit 1
        ;;
esac

# Validate model
case $MODEL in
    qwen2.5-vl|qwen2-vl|qwen2.5-vl-32b|ui-tars|cogagent|gemini|claude|gpt-4o|websight)
        ;;
    *)
        echo "Warning: Model '$MODEL' not in predefined list. Make sure it's configured in llm_config.py"
        ;;
esac

# Set result directory based on eval type, domain, and model
DATETIME=$(date +"%Y%m%d_%H%M%S")
FULL_RESULT_DIR="${RESULT_DIR}/${EVAL_TYPE}/${DOMAIN}/${MODEL}/${DATETIME}"

# Create result directory
mkdir -p "$FULL_RESULT_DIR"

# Build command
CMD="python run.py \
    --evaluation_type $EVAL_TYPE \
    --domain $DOMAIN \
    --model $MODEL \
    --max_steps $MAX_STEPS \
    --result_dir $FULL_RESULT_DIR \
    --datetime $DATETIME"

# Add optional flags
if [ "$USE_MEMORY" = true ]; then
    CMD="$CMD --use_memory True"
fi

if [ "$USE_CONTINUOUS_MEMORY" = true ]; then
    CMD="$CMD --use_continuous_memory True"
fi

if [ "$COLLECT_TRAINING_DATA" = true ]; then
    CMD="$CMD --collect_training_data --save_examples_memory"
fi

# Print configuration
echo "============================================"
echo "GUI-Agent Baseline Evaluation"
echo "============================================"
echo "Evaluation Type: $EVAL_TYPE"
echo "Domain:          $DOMAIN"
echo "Model:           $MODEL"
echo "Max Steps:       $MAX_STEPS"
echo "Result Dir:      $FULL_RESULT_DIR"
echo "Use Memory:      $USE_MEMORY"
echo "Use Continuous Memory: $USE_CONTINUOUS_MEMORY"
echo "Collect Data:    $COLLECT_TRAINING_DATA"
echo "============================================"
echo ""
echo "Running command:"
echo "$CMD"
echo ""
echo "============================================"

# Run the evaluation
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "Evaluation completed successfully!"
    echo "Results saved to: $FULL_RESULT_DIR"
    echo "============================================"
else
    echo ""
    echo "============================================"
    echo "Evaluation failed with exit code $?"
    echo "============================================"
    exit 1
fi


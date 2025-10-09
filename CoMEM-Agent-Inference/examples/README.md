# Example Scripts

This directory contains example scripts for running baseline evaluations.

## Quick Start

Make scripts executable:
```bash
chmod +x *.sh
```

## Available Examples

### MMInA Evaluation

**Shopping Domain:**
```bash
./run_mmina_shopping.sh
```
Evaluates the agent on 200 shopping tasks using Qwen2.5-VL.

**Wikipedia Domain:**
```bash
./run_mmina_wikipedia.sh
```
Evaluates on 308 Wikipedia-based QA tasks using Qwen2.5-VL.

### Mind2Web Evaluation

```bash
./run_mind2web.sh
```
Runs Mind2Web benchmark with UI-TARS model on `test_domain_Info`.

### WebVoyager Evaluation

```bash
./run_webvoyager.sh
```
Evaluates on WebVoyager test set with GPT-4o.

### Memory-Enhanced Evaluation

**Text-based Memory (Baseline):**
```bash
./run_with_text_memory.sh
```
Runs evaluation with traditional text-based experience memory retrieval.

**Continuous Memory (CoMEM):**
```bash
./run_with_continuous_memory.sh
```
Runs evaluation with our continuous memory approach.

## Customization

You can modify any example script or use the main script directly:

```bash
../run_baseline.sh \
    --eval_type mmina \
    --domain wikipedia \
    --model gemini \
    --max_steps 15 \
    --use_memory text
```

See `../run_baseline.sh --help` for all available options.

## Supported Configurations

### Evaluation Types & Domains

**mmina:**
- `shopping` (200 tasks)
- `wikipedia` (308 tasks)
- `normal` (176 tasks)
- `multi567` (180 tasks)
- `compare` (100 tasks)
- `multipro` (86 tasks)

**mind2web:**
- `test_website`
- `test_domain_Info`
- `test_domain_Service`

**webvoyager:**
- `test`
- `Allrecipes`, `Amazon`, `Apple`, `ArXiv`
- `Booking`, `GitHub`, `Google_Map`, `Google_Search`
- `Google_Flights`, `ESPN`, `Huggingface`
- `BBC_News`, `Wolfram_Alpha`

### Models

**Open-Source:**
- `qwen2.5-vl` - Qwen 2.5 Vision-Language 7B
- `qwen2-vl` - Qwen 2 Vision-Language 7B
- `qwen2.5-vl-32b` - Qwen 2.5 VL 32B (requires more VRAM)
- `ui-tars` - ByteDance UI-TARS 7B
- `cogagent` - CogAgent 9B
- `websight` - WebSight 7B

**Commercial (via API):**
- `gemini` - Google Gemini 2.5 Pro (via OpenRouter)
- `claude` - Anthropic Claude Sonnet 4 (via OpenRouter)
- `gpt-4o` - OpenAI GPT-4o

**CoMEM Models:**
- `agent-qformer` 
- Qwen2.5-VL + Continuous Memory ([WenyiWU0111/lora_qformer_test_V4-700_merged](https://huggingface.co/WenyiWU0111/lora_qformer_test_V4-700_merged))
- UI-TARS-V1.57B + Continuous Memory ([WenyiWU0111/lora_qformer_uitars_test_V1-400_merged](https://huggingface.co/WenyiWU0111/lora_qformer_uitars_test_V1-400_merged))


**Note:** Commercial models require API keys configured in your environment or via OpenRouter.

## Memory Types

### Text Memory
Traditional approach that converts trajectories to text tokens:
```bash
--use_memory
```

Pros:
- Simple implementation
- Human-readable

Cons:
- Context length grows with trajectory length
- Loses visual details (widget positions, sizes)
- Performance degrades with long prompts

### Continuous Memory (CoMEM)
Our approach that encodes trajectories into fixed-length embeddings:
```bash
--use_continuous_memory
```

Pros:
- **Fixed memory size** (8 embeddings regardless of trajectory length)
- **Preserves visual information** (exact positions, sizes)
- **Monotonic scaling** (more memory = better performance)
- **Parameter efficient** (only 1.2% of model fine-tuned)

## Adding New Models

To add a new model:

1. Edit `agent/llm_config.py`
2. Find the `create_direct_vllm_model` function
3. Add your model to `model_name_map` and `model_server_map` dictionaries:

```python
model_name_map = {
    'your-model': 'HuggingFace/model-name',
    ...
}

model_server_map = {
    'your-model': 'http://localhost:PORT/v1',
    ...
}
```

4. Start your vLLM server:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model HuggingFace/model-name \
    --port PORT
```

5. Create a new example script for your model

## Training Your Own Memory

For training custom continuous memory models, see our training repository:
**[CoMEM Training](https://github.com/WenyiWU0111/CoMEM)**

The training process:
- Fine-tune only Q-Former (1.2% of parameters) with LoRA
- Use 1,500 samples for training
- Encode trajectories into 8 continuous embeddings
- Keep base VLM frozen

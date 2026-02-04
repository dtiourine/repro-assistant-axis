# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research replication of "The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models" (Lu et al. 2026, https://arxiv.org/pdf/2601.10387). The project generates persona evaluation data for 275 character roles to extract the "Assistant Axis" - a fundamental dimension characterizing how LLMs present their default persona.

## Environment Setup

```bash
# Install core dependencies
pip install anthropic python-dotenv tqdm transformer-lens

# Install data processing dependencies
pip install pandas pyarrow

# Install vLLM for response generation (requires CUDA GPU and Linux/WSL)
pip install vllm

# Configure API key in .env file
ANTHROPIC_API_KEY=your-api-key-here
```

API key: https://console.anthropic.com/settings/keys

**Note:** vLLM requires a CUDA-compatible GPU and Linux (or WSL2 on Windows). If you don't have GPU access, you can skip vLLM installation and use alternative inference methods.

## Core Commands

### Generate Role Prompts

```bash
# Generate prompts for all 275 roles (~$6-8 cost, ~275 API calls)
python scripts/generate_role_prompts.py

# Generate specific range (useful for testing or resuming)
python scripts/generate_role_prompts.py --start 0 --end 50

# Start fresh without resuming from existing results
python scripts/generate_role_prompts.py --no-resume

# Use a different model
python scripts/generate_role_prompts.py --model claude-sonnet-4-20250514
```

### Extract Flexible Questions for Rollout

```bash
# Extract 240 flexible questions from role-specific questions
python scripts/extract_flexible_questions.py
```

This extracts a subset of 240 questions that:
- Are general enough to work across all roles (not role-specific)
- Can elicit different responses based on the model's expressed persona
- Used for the "rollout step" in the paper's methodology

### List Available Models

```bash
python scripts/list_anthropic_models.py
```

### Generate Rollout Combinations

```bash
# Generate all rollout combinations (role × instruction × question)
python scripts/generate_rollouts.py
```

This creates all combinations of:
- 338 roles (from role_prompts.json)
- 5 system prompts per role
- 240 extraction questions

Total: ~409,200 rollouts (5 × 240 = 1,200 per role)

Output: `data/rollouts.parquet` (Parquet format, ~0.14 MB)

**Note**: Three roles (arbitrator, champion, diplomat) appear twice in roles.json and thus have 10 instructions instead of 5, resulting in 2,400 rollouts each.

To load the data:
```python
import pandas as pd
df = pd.read_parquet('data/rollouts.parquet')
# Columns: role, role_category, instruction_idx, instruction_text, question_idx, question_text
```

### Generate Model Responses

```bash
# Generate responses for all rollouts using vLLM (default: gemma-2-2b-it)
python scripts/generate_responses.py

# Use the larger 9B model (requires more VRAM)
python scripts/generate_responses.py --model google/gemma-2-9b-it --gpu-memory-utilization 0.6

# Use custom batch size and save frequency
python scripts/generate_responses.py --batch-size 128 --save-every 1000

# Start fresh (ignore existing responses)
python scripts/generate_responses.py --no-resume

# Test with limited number of rollouts
python scripts/generate_responses.py --limit 100

# Use different generation parameters
python scripts/generate_responses.py --temperature 0.8 --max-tokens 256
```

**Prerequisites:**
```bash
# Install vLLM (requires CUDA GPU and Linux/WSL)
pip install vllm

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

**Features:**
- **Batched inference** using vLLM for high throughput
- **Resume capability** - tracks processed indices, can resume after interruption
- **Incremental saving** - saves every N rollouts to avoid data loss
- **Progress tracking** - shows ETA and processing rate

**Output:** `data/responses.parquet` containing:
- `rollout_idx`: Index from original rollouts.parquet
- `role`, `role_category`, `instruction_idx`, `question_idx`: Copied from rollout
- `prompt`: Full prompt sent to model (instruction + question)
- `response`: Model's generated response
- `finish_reason`: Why generation stopped (e.g., "stop", "length")
- `tokens_generated`: Number of tokens in response

**Performance:**
- Expected: ~10-50 rollouts/second depending on GPU
- Full dataset (409k rollouts): ~2-10 hours on single GPU

**Resume Example:**
If the script crashes or you stop it, simply run it again:
```bash
python scripts/generate_responses.py
# It will automatically detect existing responses and continue where it left off
```

## Architecture

### Data Pipeline

The project follows a structured pipeline for generating persona evaluation data:

1. **Input**: `data/roles.json` - 341 role entries (338 unique roles) organized into categories (professional, service, authority, creative, etc.)
2. **Template**: `prompts/role_generation_template.txt` - Instructs Claude to generate:
   - 5 system prompts (different ways to instruct the model to exhibit the role)
   - 40 evaluation questions (subtle questions that might elicit role-related behavior)
   - 1 evaluation prompt (rates responses 0-3 for role adherence)
3. **Generation**: `scripts/generate_role_prompts.py` - Calls Claude API for each role
4. **Output**: `data/role_prompts.json` - Generated data for all roles (~13,500 total questions, ~13,400 unique)
5. **Extraction**: Created 240 flexible extraction questions designed to elicit different responses across personas
6. **Questions Output**: `data/extraction_questions.json` - 240 general questions for the rollout step
7. **Rollout Generation**: `scripts/generate_rollouts.py` - Combines all role instructions with all extraction questions
8. **Rollout Output**: `data/rollouts.parquet` - 409,200 rollout combinations (role × instruction × question) in Parquet format
9. **Response Generation**: `scripts/generate_responses.py` - Runs gemma-2-2b-it (default) on all rollouts using vLLM for batched inference
10. **Response Output**: `data/responses.parquet` - Model responses for each rollout with metadata (tokens, finish_reason, etc.)

### Key Design Decisions

**Resume Capability**: The script saves progress every 5 roles and can resume from existing `role_prompts.json`. This is critical because:
- Full run takes ~5-10 minutes and costs $6-8
- API failures or rate limits may interrupt processing
- Allows iterative development without regenerating all data

**Error Handling**: When a role fails (e.g., JSON parsing errors), the script:
1. Logs the error with role name
2. Saves current progress immediately
3. Re-raises the exception to stop processing

**Content Policy Considerations**: Some role names may trigger Claude's content policy (e.g., "slave" was replaced with "serf"). If generation fails for a specific role:
- Check the error message for JSONDecodeError
- The role likely received an empty/refusal response
- Replace with a similar but acceptable alternative in `data/roles.json`

### Data Format

`data/role_prompts.json` structure:
```json
{
  "role_name": {
    "category": "professional",
    "instruction": [
      {"pos": "You are a role_name..."},
      {"pos": "Please be a role_name..."},
      ...
    ],
    "questions": [
      "Question 1...",
      "Question 2...",
      ...
    ],
    "eval_prompt": "You are evaluating whether..."
  }
}
```

`data/extraction_questions.json` structure:
```json
{
  "description": "240 flexible extraction questions for the Assistant Axis rollout step",
  "source": "Selected from role-specific questions generated for 275+ roles",
  "selection_criteria": "Questions are general enough to elicit different responses based on persona",
  "count": 240,
  "questions": [
    "How should we address climate change?",
    "What would you do if you disagreed with a group decision?",
    ...
  ]
}
```

**Question Selection Criteria**: Flexible questions avoid role-specific terminology (e.g., "edit this paragraph" for editors, "debug this code" for programmers) and instead focus on general life, philosophical, ethical, and practical topics that any persona could answer from their unique perspective.

`data/rollouts.parquet` structure:
- **Format**: Apache Parquet (columnar binary format)
- **Compression**: Snappy (~0.14 MB for 409,200 rows)
- **Schema**:
  - `role` (string): Role name (e.g., "editor", "jester")
  - `role_category` (string): Category of role (e.g., "professional", "entertainer")
  - `instruction_idx` (int64): Index of instruction (0-4, or 0-9 for duplicate roles)
  - `instruction_text` (string): Full system prompt text
  - `question_idx` (int64): Index of extraction question (0-239)
  - `question_text` (string): Full extraction question text
- **Rows**: 409,200 total (1,200 per role × 338 roles, plus 3 roles with duplicates)

**Loading in Python**:
```python
import pandas as pd
df = pd.read_parquet('data/rollouts.parquet')

# Filter by role
editor_rollouts = df[df['role'] == 'editor']

# Get all rollouts for a specific instruction
first_instructions = df[df['instruction_idx'] == 0]

# Access specific rollout
print(df.loc[0, 'instruction_text'])
print(df.loc[0, 'question_text'])
```

`data/responses.parquet` structure:
- **Format**: Apache Parquet (columnar binary format)
- **Compression**: Snappy
- **Schema**:
  - `rollout_idx` (int64): Index into rollouts.parquet (for joining data)
  - `role` (string): Role name (copied from rollout)
  - `role_category` (string): Category of role (copied from rollout)
  - `instruction_idx` (int64): Instruction index (copied from rollout)
  - `question_idx` (int64): Question index (copied from rollout)
  - `prompt` (string): Full prompt sent to model (instruction + question)
  - `response` (string): Model's generated response
  - `finish_reason` (string): Why generation stopped (e.g., "stop", "length")
  - `tokens_generated` (int64): Number of tokens in the response
- **Rows**: 409,200 total (one response per rollout)

**Loading and Analyzing Responses**:
```python
import pandas as pd

# Load responses
responses_df = pd.read_parquet('data/responses.parquet')

# Join with rollouts if needed (responses already include key fields)
rollouts_df = pd.read_parquet('data/rollouts.parquet')
# responses_df has rollout_idx, so you can join if you need full rollout data
full_df = responses_df.merge(rollouts_df, left_on='rollout_idx', right_index=True, how='left')

# Analyze responses
print(f"Average response length: {responses_df['tokens_generated'].mean():.1f} tokens")
print(f"Completion rate: {(responses_df['finish_reason'] == 'stop').mean()*100:.1f}%")

# Filter by role
editor_responses = responses_df[responses_df['role'] == 'editor']
```

## Common Issues

**JSONDecodeError for specific role**: Role name likely triggers content policy. Replace the role in `data/roles.json` with a similar alternative (e.g., "slave" → "serf").

**Missing API key**: Ensure `.env` file exists with `ANTHROPIC_API_KEY=...` and is in the project root.

**Rate limiting**: The script includes 1-second delays between requests. For aggressive rate limits, increase `time.sleep(1)` value in `generate_role_prompts.py:128`.

**Duplicate roles in rollouts**: Three roles (arbitrator, champion, diplomat) appear twice in `roles.json`, resulting in 10 instructions instead of 5. This creates 2,400 rollouts per role instead of 1,200. To fix, remove duplicates from `roles.json` and regenerate `role_prompts.json`.

**vLLM installation fails**: vLLM requires:
- CUDA-compatible GPU (NVIDIA)
- Linux or WSL2 on Windows (does not support native Windows)
- Python 3.8+
- CUDA toolkit installed

If on Windows, use WSL2:
```bash
# In WSL2:
pip install vllm
```

**CUDA out of memory**: The default model (gemma-2-2b-it) should work on GPUs with 8GB+ VRAM. If you still get OOM:
```bash
python scripts/generate_responses.py --gpu-memory-utilization 0.7 --batch-size 32
```

**Understanding the parameters:**
- `--gpu-memory-utilization` (default: 0.9): Fraction of GPU memory to allocate. Lower = more conservative.
- `--max-model-len` (default: 2048): Maximum context length. Shorter = less memory for KV cache.
- `--batch-size` (default: 64): Number of prompts to process simultaneously. Smaller = less memory but slower.
- `--max-tokens` (default: 512): Maximum response length. Shorter = less memory.

**Using the larger 9B model:** If you want higher quality responses and have a GPU with 24GB+ VRAM:
```bash
python scripts/generate_responses.py \
  --model google/gemma-2-9b-it \
  --gpu-memory-utilization 0.6 \
  --max-model-len 1024 \
  --batch-size 16
```

**vLLM model download is slow**: First run downloads the full model (~5GB for gemma-2-2b-it, ~17GB for 9b). This is cached for future runs.

**Response generation interrupted**: The script saves progress every N rollouts (default 500). Simply run the script again and it will resume from where it left off:
```bash
python scripts/generate_responses.py
# Automatically resumes from existing responses.parquet
```

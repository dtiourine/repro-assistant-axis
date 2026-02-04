# Assistant Axis Replication

Replicating and extending ["The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models"](https://arxiv.org/pdf/2601.10387) (Lu et al. 2026)

## Setup

### 1. Install Dependencies

```bash
pip install anthropic python-dotenv tqdm transformer-lens
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` and add your Anthropic API key:

```
ANTHROPIC_API_KEY=your-api-key-here
```

Get your API key from: https://console.anthropic.com/settings/keys

**Note:** The `.env` file is gitignored for security, so your API key will not be committed to version control.

## Project Structure

```
repro-assistant-axis/
├── data/
│   ├── roles.json              # 275 character archetypes
│   └── role_prompts.json       # Generated prompts for each role
├── prompts/
│   └── role_generation_template.txt  # Template for generating role data
├── scripts/
│   └── generate_role_prompts.py      # Script to generate role prompts
├── experiments/
│   ├── 2026-01-23_get_assistant_axis.ipynb
│   └── 2026-01-24_generate_role_prompts.ipynb
├── .env.example                # Example environment variables
└── README.md
```

## Usage

### Generate Role Prompts

The first step is to generate 5 system prompts, 40 questions, and 1 evaluation prompt for each of the 275 roles.

**Option 1: Python Script (Recommended for full batch processing)**

```bash
# Generate prompts for all roles
python scripts/generate_role_prompts.py

# Generate prompts for specific range
python scripts/generate_role_prompts.py --start 0 --end 50

# Generate without resuming from existing results
python scripts/generate_role_prompts.py --no-resume
```

**Option 2: Jupyter Notebook (Recommended for interactive exploration)**

Open and run `experiments/2026-01-24_generate_role_prompts.ipynb`

Features:
- Auto-resume capability (saves every 5 roles)
- Progress tracking with tqdm
- Error handling and logging
- ~275 API calls × ~8K tokens ≈ $6-8 total cost

### Extract the Assistant Axis

TODO: Add instructions for extracting the assistant axis from model activations

## Paper Overview

The paper investigates how language models develop and maintain a default "assistant" persona:

- **Assistant Axis**: A spectrum representing how strongly models exhibit assistant-like qualities
- **Persona Space**: Multidimensional representation of personality traits
- **Role Anchoring**: How instruction-based roles position models relative to the assistant axis
- **Key Finding**: Models exhibit a stable "default assistant" orientation that resists persona shifts

## Methodology

1. Generate diverse role prompts (275 roles)
2. Collect model activations for each role
3. Extract the assistant axis using PCA or similar dimensionality reduction
4. Evaluate role adherence using generated questions
5. Analyze stability across different contexts

## License

This is a research replication project. See the original paper for citation information.

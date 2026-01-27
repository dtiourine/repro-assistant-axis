"""
Generate rollouts for "default assistant" behavior using the same 240 extraction questions.

This creates 1,200 rollouts:
- 4 system prompts that instruct normal/default assistant behavior
- 1 run with no system prompt (empty string)
- × 240 extraction questions

Usage:
    python scripts/generate_default_rollouts.py
"""

import json
from pathlib import Path
import pandas as pd

# Define 5 system prompts for default assistant behavior
# 4 explicit assistant prompts + 1 empty (no system prompt)
DEFAULT_INSTRUCTIONS = [
    {
        "idx": 0,
        "text": "",  # No system prompt
        "description": "no_system_prompt"
    },
    {
        "idx": 1,
        "text": "You are a helpful assistant.",
        "description": "helpful_assistant"
    },
    {
        "idx": 2,
        "text": "You are a large language model trained to be helpful, harmless, and honest.",
        "description": "helpful_harmless_honest"
    },
    {
        "idx": 3,
        "text": "Respond as yourself.",
        "description": "respond_as_yourself"
    },
    {
        "idx": 4,
        "text": "You are an AI assistant. Please provide helpful and accurate responses.",
        "description": "ai_assistant"
    },
]


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    questions_path = data_dir / "extraction_questions.json"
    output_path = data_dir / "rollouts_default.parquet"

    # Load extraction questions
    print(f"Loading extraction questions from {questions_path}...")
    with open(questions_path, 'r') as f:
        questions_data = json.load(f)

    questions = questions_data['questions']
    print(f"Loaded {len(questions)} questions")

    # Generate all combinations
    rollouts = []

    for instruction in DEFAULT_INSTRUCTIONS:
        for q_idx, question in enumerate(questions):
            rollouts.append({
                'role': 'default_assistant',
                'role_category': 'default',
                'instruction_idx': instruction['idx'],
                'instruction_text': instruction['text'],
                'instruction_description': instruction['description'],
                'question_idx': q_idx,
                'question_text': question
            })

    # Create DataFrame
    df = pd.DataFrame(rollouts)

    print(f"\nGenerated {len(df):,} rollouts")
    print(f"  Instructions: {len(DEFAULT_INSTRUCTIONS)}")
    print(f"  Questions: {len(questions)}")
    print(f"  Total: {len(DEFAULT_INSTRUCTIONS)} × {len(questions)} = {len(df)}")

    # Show instruction breakdown
    print("\nInstruction breakdown:")
    for instruction in DEFAULT_INSTRUCTIONS:
        count = len(df[df['instruction_idx'] == instruction['idx']])
        desc = instruction['description']
        text_preview = instruction['text'][:50] + "..." if len(instruction['text']) > 50 else instruction['text']
        if not text_preview:
            text_preview = "(empty)"
        print(f"  [{instruction['idx']}] {desc}: {count} rollouts - \"{text_preview}\"")

    # Save to parquet
    df.to_parquet(output_path, compression='snappy', index=False)
    print(f"\nSaved to {output_path}")
    print(f"File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()

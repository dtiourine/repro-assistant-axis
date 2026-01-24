import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
from tqdm import tqdm


def load_data():
    data_dir = Path(__file__).parent.parent / "data"

    with open(data_dir / "roles.json", "r", encoding="utf-8") as f:
        roles = json.load(f)

    with open(data_dir / "role_prompts.json", "r", encoding="utf-8") as f:
        role_prompts = json.load(f)

    with open(data_dir / "extraction_questions.json", "r", encoding="utf-8") as f:
        extraction_data = json.load(f)
        extraction_questions = extraction_data["questions"]

    return roles, role_prompts, extraction_questions


def generate_rollouts(roles: List[Dict], role_prompts: Dict, extraction_questions: List[str]) -> pd.DataFrame:
    rollouts = []

    print(f"Generating rollouts for {len(roles)} roles...")
    print(f"Each role: 5 instructions × {len(extraction_questions)} questions = {5 * len(extraction_questions)} rollouts")
    print(f"Total: {len(roles) * 5 * len(extraction_questions):,} rollouts")

    for role_data in tqdm(roles, desc="Processing roles"):
        role_name = role_data["role"]
        role_category = role_data["category"]

        if role_name not in role_prompts:
            print(f"Warning: {role_name} not found in role_prompts.json, skipping...")
            continue

        prompts = role_prompts[role_name]
        instructions = prompts["instruction"]

        if len(instructions) != 5:
            print(f"Warning: {role_name} has {len(instructions)} instructions instead of 5")

        for inst_idx, instruction in enumerate(instructions):
            instruction_text = instruction["pos"]

            for q_idx, question_text in enumerate(extraction_questions):
                rollouts.append({
                    "role": role_name,
                    "role_category": role_category,
                    "instruction_idx": inst_idx,
                    "instruction_text": instruction_text,
                    "question_idx": q_idx,
                    "question_text": question_text
                })

    df = pd.DataFrame(rollouts)
    return df


def save_rollouts(df: pd.DataFrame, output_dir: Path):
    output_dir.mkdir(exist_ok=True, parents=True)

    parquet_path = output_dir / "rollouts.parquet"
    print(f"\nSaving to {parquet_path}...")
    df.to_parquet(parquet_path, compression="snappy", index=False)

    file_size_mb = parquet_path.stat().st_size / (1024 * 1024)
    print(f"Saved {len(df):,} rollouts ({file_size_mb:.2f} MB)")

    # Save a small sample to JSON 
    sample_size = 100
    sample = df.head(sample_size).to_dict(orient="records")
    sample_path = output_dir / "rollouts_sample.json"

    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump({
            "description": f"First {sample_size} rollouts from the full dataset",
            "total_rollouts": len(df),
            "sample_size": sample_size,
            "schema": {
                "role": "string - role name",
                "role_category": "string - category of the role",
                "instruction_idx": "int - index of instruction (0-4)",
                "instruction_text": "string - full system prompt text",
                "question_idx": "int - index of extraction question (0-239)",
                "question_text": "string - full extraction question text"
            },
            "rollouts": sample
        }, f, indent=2, ensure_ascii=False)

    print(f"Saved sample to {sample_path}")

    print("\n=== Summary Statistics ===")
    print(f"Total rollouts: {len(df):,}")
    print(f"Unique roles: {df['role'].nunique()}")
    print(f"Unique role categories: {df['role_category'].nunique()}")
    print(f"Unique instructions per role: {df['instruction_idx'].nunique()}")
    print(f"Unique questions: {df['question_idx'].nunique()}")
    print(f"\nRollouts per role: {len(df) // df['role'].nunique()}")
    print(f"Expected per role: {5 * 240} (5 instructions × 240 questions)")


def main():
    roles, role_prompts, extraction_questions = load_data()

    df = generate_rollouts(roles, role_prompts, extraction_questions)

    output_dir = Path(__file__).parent.parent / "data"
    save_rollouts(df, output_dir)

    print("\n✓ Rollout generation complete!")
    print(f"\nTo load the data in Python:")
    print("  import pandas as pd")
    print("  df = pd.read_parquet('data/rollouts.parquet')")


if __name__ == "__main__":
    main()

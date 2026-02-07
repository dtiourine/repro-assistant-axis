"""
This script filters the raw model responses based on the evaluation scores from the LLM judge.

Method (matches The Assistant Axis):
1) Keep only eval rows with score in {2, 3}.
2) Count rows per (role, score); keep (role, score) pairs with >= min_per_role_score.
3) Filter the responses table to only those response keys appearing in the filtered eval table.
"""
import argparse 
import pandas as pd 
import questionary 
from pathlib import Path 
from repro_assistant_axis.config import DATA_DIR, RAW_MODEL_RESPONSES_DIR, RESPONSE_EVALUATIONS_DIR, FILTERED_MODEL_RESPONSES_DIR

KEY_COLS = ["rollout_idx", "role", "instruction_idx", "question_idx"]

MODELS = [
    "Gemma-2-2B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Llama-3.2-3B-Instruct",
    "default_Gemma-2-2B-Instruct",
    "default_Qwen2.5-3B-Instruct",
    "default_Llama-3.2-3B-Instruct",
]

def _require_columns(df: pd.DataFrame, cols: list[str], df_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")

def filter_responses_by_eval(model_name: str, min_per_role_score: int = 10):
    eval_path = RESPONSE_EVALUATIONS_DIR / f"evaluations_{model_name}.parquet"
    resp_path = RAW_MODEL_RESPONSES_DIR / f"responses_{model_name}.parquet"
    out_eval_path = FILTERED_MODEL_RESPONSES_DIR / f"filtered_{model_name}_evaluations.parquet"
    out_resp_path = FILTERED_MODEL_RESPONSES_DIR / f"filtered_{model_name}_responses.parquet"

    FILTERED_MODEL_RESPONSES_DIR.mkdir(exist_ok=True, parents=True)
    
    print(f"\n🔍 Filtering {model_name} responses...")
    
    try:
        eval_df = pd.read_parquet(eval_path)
        responses_df = pd.read_parquet(resp_path)
    except FileNotFoundError as e:
        print(f"❌ Skipping {model_name}: File not found ({e.filename})")
        return

    _require_columns(eval_df, KEY_COLS + ["score", "role"], "evaluations parquet")
    _require_columns(responses_df, KEY_COLS, "responses parquet")

    good_df = eval_df[eval_df["score"].isin([2, 3])].copy()

    pair_counts = good_df.groupby(["role", "score"]).size().reset_index(name="n")
    kept_pairs = pair_counts[pair_counts["n"] >= min_per_role_score][["role", "score"]]

    filtered_eval_df = good_df.merge(kept_pairs, on=["role", "score"], how="inner")
    keep_keys = filtered_eval_df[KEY_COLS].drop_duplicates()
    filtered_responses_df = responses_df.merge(keep_keys, on=KEY_COLS, how="inner")

    filtered_eval_df.to_parquet(out_eval_path, index=False)
    filtered_responses_df.to_parquet(out_resp_path, index=False)

    print(f"  ✅ Done! Kept {len(filtered_responses_df):,} responses.")
    print(f"  📂 Wrote: {out_resp_path.name}")
    return filtered_responses_df, filtered_eval_df, kept_pairs

def prompt_for_models() -> list[str]:
    choices = ["All Models"] + MODELS
    selection = questionary.select(
        "Which model responses would you like to filter?",
        choices=choices,
        pointer="👉"
    ).ask()

    if selection is None:
        exit(0)
    
    return MODELS if selection == "All Models" else [selection]

def main():
    parser = argparse.ArgumentParser(description="Filter model responses based on LLM judge scores.")
    parser.add_argument(
        "--model", 
        type=str, 
        choices=MODELS + ["all"], 
        help="Specific model to filter or 'all'."
    )
    parser.add_argument(
        "--min_score", 
        type=int, 
        default=10, 
        help="Minimum samples per (role, score) pair (default: 10)."
    )
    
    args = parser.parse_args()

    if args.model == "all":
        selected_models = MODELS
    elif args.model:
        selected_models = [args.model]
    else:
        selected_models = prompt_for_models()

    for m in selected_models:
        filter_responses_by_eval(model_name=m, min_per_role_score=args.min_score)


if __name__ == "__main__":
    main()

    
    
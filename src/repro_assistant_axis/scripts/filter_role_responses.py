"""
This script filters the raw model responses based on the evaluation scores from the LLM judge.

Method (matches The Assistant Axis):
1) Keep only eval rows with score in {2, 3}.
2) Count rows per (role, score); keep (role, score) pairs with >= min_per_role_score.
3) Filter the responses table to only those response keys appearing in the filtered eval table.
"""
import pandas as pd 
import sys
from repro_assistant_axis.config import DATA_DIR, RAW_MODEL_RESPONSES_DIR, RESPONSE_EVALUATIONS_DIR, FILTERED_MODEL_RESPONSES_DIR

KEY_COLS = ["rollout_idx", "role", "instruction_idx", "question_idx"]

MODELS = [
    "Gemma-2-2B-Instruct",
    "Qwen2.5-3B-Instruct",
    "Llama-3.2-3B-Instruct",
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

    eval_df = pd.read_parquet(eval_path)
    responses_df = pd.read_parquet(resp_path)

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

    print(f"{model_name}:")
    print(f"  Eval rows total: {len(eval_df):,}")
    print(f"  Eval rows kept (score 2/3 + >={min_per_role_score}/pair): {len(filtered_eval_df):,}")
    print(f"  Responses rows total: {len(responses_df):,}")
    print(f"  Responses rows kept: {len(filtered_responses_df):,}")
    print(f"  Role-score combinations kept: {len(kept_pairs):,}")
    print(f"  Unique roles in kept responses: {filtered_responses_df['role'].nunique():,}")
    print(f"  Wrote:\n    {out_eval_path}\n    {out_resp_path}")

    return filtered_responses_df, filtered_eval_df, kept_pairs



def main():
    for model_name in MODELS:
        filter_responses_by_eval(model_name=model_name, min_per_role_score=10)


if __name__ == "__main__":
    main()

    
    
import argparse
import questionary
from enum import Enum
from pathlib import Path
from typing import Union
from tqdm import tqdm

import torch
from repro_assistant_axis.config import (
    FILTERED_MODEL_RESPONSES_DIR,
    MODEL_RESPONSE_ACTIVATIONS_DIR,
)
import pandas as pd
import polars as pl
from transformer_lens import HookedTransformer

MODEL_RESPONSE_ACTIVATIONS_DIR.mkdir(exist_ok=True)


class ModelName(Enum):
    GEMMA_2_2B_INSTRUCT = "Gemma-2-2B-Instruct"
    QWEN_25_3B_INSTRUCT = "Qwen2.5-3B-Instruct"
    LLAMA_32_3B_INSTRUCT = "LLama-3.2-3B-Instruct"
    


MODEL_ID_MAP = {
    ModelName.GEMMA_2_2B_INSTRUCT: "google/gemma-2-2b-it",
    ModelName.QWEN_25_3B_INSTRUCT: "Qwen/Qwen2.5-3B-Instruct",
    ModelName.LLAMA_32_3B_INSTRUCT: "meta-llama/Llama-3.2-3B-Instruct",
}


def load_model_responses(
    model_name: ModelName, file_path: Union[str, Path] | None = None
) -> pl.DataFrame:
    if file_path is None:
        file_path = (
            FILTERED_MODEL_RESPONSES_DIR
            / f"filtered_{model_name.value}_responses.parquet"
        )
        print(f"No path provided. Defaulting to: {file_path}")

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"The file does not exist at: {path.absolute()}")

    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a .parquet file, but got: {path.suffix}")

    try:
        return pl.read_parquet(path)
    except Exception as e:
        raise RuntimeError(f"Polars failed to parse the file at {path}: {e}") from e


def validate_dataset_integrity(df: pl.DataFrame):
    """
    Ensures the dataframe has the required columns and
    that rollout_idx serves as a valid primary key.
    """
    required_columns = ["response", "rollout_idx"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f"Schema mismatch! Missing columns: {missing}. Found: {df.columns}"
        )

    row_count = df.height
    unique_indices = df["rollout_idx"].n_unique()

    if row_count != unique_indices:
        dupe_count = row_count - unique_indices
        raise ValueError(
            f"Integrity Error: {dupe_count} duplicate rollout_idx found! "
            f"({unique_indices} unique vs {row_count} total rows)."
        )

    print(f"✓ Integrity check passed: {row_count} valid rows found.")


def get_model_and_metadata(model_enum: ModelName):
    hf_name = MODEL_ID_MAP[model_enum]
    print(f"Loading {hf_name}...")

    model = HookedTransformer.from_pretrained(
        hf_name, device="cuda", fold_ln=True, center_writing_weights=True
    )
    target_layer = model.cfg.n_layers // 2
    hook_name = f"blocks.{target_layer}.hook_resid_post"
    return model, hook_name


def extract_activation_vectors(
    model_name: ModelName,
    response_data_path: Union[str, Path] | None = None,
    chunk_size: int = 1000,
    batch_size: int = 32,
):
    df = load_model_responses(model_name, response_data_path)
    validate_dataset_integrity(df)

    output_dir = MODEL_RESPONSE_ACTIVATIONS_DIR / f"{model_name.value}_activations/"
    output_dir.mkdir(exist_ok=True, parents=True)

    existing_indices = set()
    for file in output_dir.glob("*.parquet"):
        existing_indices.update(
            pl.read_parquet(file).get_column("rollout_idx").to_list()
        )

    todo_df = df.filter(~pl.col("rollout_idx").is_in(existing_indices))

    if todo_df.is_empty():
        print("🎉 All work already completed!")
        return

    print(f"Processing: {len(existing_indices)} done, {todo_df.height} remaining.")

    model, hook_name = get_model_and_metadata(model_name)

    pbar = tqdm(total=todo_df.height, desc=f"🚀 {model_name.value}", unit="rows")

    for i in range(0, todo_df.height, chunk_size):
        chunk = todo_df.slice(i, chunk_size)
        all_acts = []

        for j in range(0, chunk.height, batch_size):
            mini_batch = chunk.slice(j, batch_size)
            prompts = mini_batch["response"].to_list()
            
            torch.cuda.empty_cache()

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    prompts,
                    names_filter=lambda n: n == hook_name,
                    stop_at_layer=model.cfg.n_layers // 2 + 1,
                )
                acts = cache[hook_name][:, -1, :].detach().cpu().numpy()
                all_acts.extend(acts.tolist())
                del cache

            pbar.update(len(prompts))

        result_chunk = chunk.with_columns(activations=pl.Series(all_acts))
        fname = f"acts_{chunk['rollout_idx'][0]}_to_{chunk['rollout_idx'][-1]}.parquet"
        result_chunk.write_parquet(output_dir / fname)
        print(f"✅ Saved {fname}")


def prompt_for_model() -> ModelName:
    """Displays an interactive arrow-key menu to select a model."""
    choices = [m.value for m in ModelName]

    selected_value = questionary.select(
        "Which model would you like to process?",
        choices=choices,
        pointer="👉",
        use_shortcuts=True,
    ).ask()

    if selected_value is None:
        print("Selection cancelled. Exiting.")
        exit(0)

    return next(m for m in ModelName if m.value == selected_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract residual stream activations from model responses."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[m.value for m in ModelName],
        help="The model enum value",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Optional custom path to responses.parquet",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="GPU inference batch size"
    )

    args = parser.parse_args()

    if args.model:
        selected_model = next(m for m in ModelName if m.value == args.model)
    else:
        selected_model = prompt_for_model()

    extract_activation_vectors(
        model_name=selected_model,
        response_data_path=args.input,
        batch_size=args.batch_size,
    )

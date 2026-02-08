import argparse

import questionary
from repro_assistant_axis.config import MODEL_RESPONSE_ACTIVATIONS_DIR, ROLE_VECTORS_DIR
from enum import Enum
from pathlib import Path
from typing import Union
import polars as pl
import numpy as np
from tqdm import tqdm


ROLE_VECTORS_DIR.mkdir(exist_ok=True, parents=True)


class ModelName(Enum):
    GEMMA_2_2B_INSTRUCT = "Gemma-2-2B-Instruct"
    QWEN_25_3B_INSTRUCT = "Qwen2.5-3B-Instruct"
    LLAMA_32_3B_INSTRUCT = "LLama-3.2-3B-Instruct"


def compute_mean_activations(model_name: ModelName):
    activations_path = (
        MODEL_RESPONSE_ACTIVATIONS_DIR / f"{model_name.value}_activations.parquet"
    )
    if not activations_path.is_file():
        raise FileNotFoundError(f"Activations file not found at: {activations_path}")

    df = pl.read_parquet(activations_path)
    unique_roles = df["role"].unique().to_list()
    role_data = []

    print(f"Computing mean activation vectors for {len(unique_roles)} roles...")
    for role in tqdm(unique_roles):
        role_acts = df.filter(pl.col("role") == role)["activations"].to_list()
        mean_acts = np.mean(np.array(role_acts), axis=0)

        role_data.append({"role": role, "mean_activation": mean_acts.tolist()})

    print("Computing overall mean role vector..")
    overall_mean = np.mean(np.array(df["activations"].to_list()), axis=0)
    role_data.append(
        {"role": "Overall_Role_Vector", "mean_activation": overall_mean.tolist()}
    )

    print("Computing mean activation vector for Default Assistant...")
    default_activations_path = (
        MODEL_RESPONSE_ACTIVATIONS_DIR
        / f"default_{model_name.value}_activations.parquet"
    )
    if not default_activations_path.is_file():
        print(f"⚠️ Warning: Default baseline not found at {default_activations_path}")
    else:
        df = pl.read_parquet(default_activations_path)
        default_assistant_activations_np = df["activations"].to_numpy()
        mean_default_assistant_activation_vector = np.mean(
            default_assistant_activations_np, axis=0
        )

        role_data.append(
            {
                "role": "Default_Assistant",
                "mean_activation": mean_default_assistant_activation_vector.tolist(),
            }
        )

    output_df = pl.DataFrame(role_data)
    output_path = ROLE_VECTORS_DIR / f"{model_name.value}_role_vectors.parquet"
    output_df.write_parquet(output_path)
    print(f"✅ Successfully saved role vectors to: {output_path}")

    return output_df


def prompt_for_model() -> ModelName:
    choices = [m.value for m in ModelName]
    selected_value = questionary.select(
        "Which model's role vectors would you like to compute?",
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
        description="Compute aggregate role vectors (centroids) from activations."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=[m.value for m in ModelName],
        help="The model enum value",
    )

    args = parser.parse_args()

    selected_model = (
        next((m for m in ModelName if m.value == args.model), None)
        or prompt_for_model()
    )

    compute_mean_activations(selected_model)

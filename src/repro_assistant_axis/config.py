from pathlib import Path

PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"

RAW_MODEL_RESPONSES_DIR = DATA_DIR / "raw_model_responses"
RESPONSE_EVALUATIONS_DIR = DATA_DIR / "response_evaluations"
FILTERED_MODEL_RESPONSES_DIR = DATA_DIR / "filtered_model_responses"
MODEL_RESPONSE_ACTIVATIONS_DIR = DATA_DIR / "model_response_activations"
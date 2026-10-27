"""Configuration module for Hate Speech Neutralizer.
Portable cross-platform paths using pathlib.Path and environment variables.
"""

import os
from pathlib import Path
import torch

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data paths
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
TOKENIZER_PATH = Path(os.getenv("TOKENIZER_PATH", DATA_DIR / "tokenizer.pkl"))
JIGSAW_CSV = DATA_DIR / "jigsaw.csv"
JIGSAW_CLEANED_CSV = DATA_DIR / "jigsaw_cleaned.csv"
PARADETOX_CSV = DATA_DIR / "paradetox.csv"
REWRITING_PAIRS_CSV = DATA_DIR / "rewriting_pairs.csv"
TEST_DATA_PKL = DATA_DIR / "test_data.pkl"
TRAIN_DATA_PKL = DATA_DIR / "train_data.pkl"

# Model paths
MODELS_DIR = Path(os.getenv("MODELS_DIR", BASE_DIR / "models"))

# Available classifier checkpoints
CLASSIFIER_CHECKPOINTS = {
    "bigru": {
        "name": "Bidirectional GRU",
        "path": MODELS_DIR / "bigru_hate_speech_model.h5",
        "description": "Bidirectional GRU recurrent neural network for toxicity sequence detection."
    },
    "gru": {
        "name": "Standard GRU",
        "path": MODELS_DIR / "gru_hate_speech_model.h5",
        "description": "Unidirectional GRU recurrent network baseline."
    },
    "bilstm": {
        "name": "Bidirectional LSTM",
        "path": MODELS_DIR / "bilstm_hate_speech_model.h5",
        "description": "Bidirectional LSTM recurrent network with forward and backward hidden states."
    },
    "lstm": {
        "name": "Standard LSTM",
        "path": MODELS_DIR / "hate_speech_model.h5",
        "description": "Unidirectional LSTM recurrent network baseline."
    }
}

# Active classifier model path (defaults to bigru)
DEFAULT_CLASSIFIER_KEY = os.getenv("DEFAULT_CLASSIFIER", "bigru")
DEFAULT_CLASSIFIER_PATH = CLASSIFIER_CHECKPOINTS[DEFAULT_CLASSIFIER_KEY]["path"]

# Rewriting model directory (fine-tuned T5)
REWRITER_MODEL_DIR = Path(os.getenv("REWRITER_MODEL_DIR", MODELS_DIR / "t5_neutralizer"))
REWRITER_INSTRUCTION = os.getenv(
    "REWRITER_INSTRUCTION",
    "Rewrite the following message in respectful, constructive English. "
    "Remove personal attacks and preserve useful feedback or requests. "
    "Return only the rewritten message, using complete grammatical sentences. "
    "Do not add unsupported details."
)
REWRITER_PREFIX = REWRITER_INSTRUCTION

# Model parameters
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", 20000))
CLASSIFIER_MAX_LEN = int(os.getenv("CLASSIFIER_MAX_LEN", 200))
REWRITER_MAX_INPUT_LEN = int(os.getenv("REWRITER_MAX_INPUT_LEN", 128))
REWRITER_MAX_OUTPUT_LEN = int(os.getenv("REWRITER_MAX_OUTPUT_LEN", 64))

# Classification threshold settings
# Threshold tuned on test/validation set for balanced F1 and safety tradeoff
DEFAULT_THRESHOLD = float(os.getenv("DETECTION_THRESHOLD", 0.40))
SAFE_THRESHOLD_MIN = 0.10
SAFE_THRESHOLD_MAX = 0.90

# Device selection (CPU with MPS / CUDA fallback)
def get_torch_device() -> torch.device:
    forced_device = os.getenv("TORCH_DEVICE")
    if forced_device:
        return torch.device(forced_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


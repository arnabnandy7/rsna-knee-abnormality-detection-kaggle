"""
Configuration module for RSNA Knee Abnormality Detection.

Centralizes paths, hyperparameters, model choices, training constants,
and target label definitions for both local development and Kaggle environments.
"""

import os
from pathlib import Path

# --- Dataset Paths ---
# Default Kaggle environment input path
KAGGLE_INPUT_DIR = Path("/kaggle/input/rsna-knee-abnormality-detection")
LOCAL_INPUT_DIR = Path("./data")

DATA_DIR = KAGGLE_INPUT_DIR if KAGGLE_INPUT_DIR.exists() else LOCAL_INPUT_DIR

TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_SERIES_CSV = DATA_DIR / "train_series.csv"
TEST_CSV = DATA_DIR / "test.csv"
TEST_SERIES_CSV = DATA_DIR / "test_series.csv"
SAMPLE_SUBMISSION = DATA_DIR / "sample_submission.csv"

TRAIN_IMAGES_DIR = DATA_DIR / "train_series"
TEST_IMAGES_DIR = DATA_DIR / "test_series"

OUTPUT_DIR = Path("./output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Target Labels (12 Binary Findings) ---
TARGET_COLUMNS = [
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
]

NUM_CLASSES = len(TARGET_COLUMNS)

# --- Hyperparameters & Model Configuration ---
SEED = 42
N_FOLDS = 5

IMAGE_SIZE = (224, 224)
NUM_SLICES_PER_STUDY = 8  # Subsample middle N slices across MRI series

# Backbones
IMAGE_MODEL_NAME = "resnet34"  # timm backbone (e.g. resnet34, efficientnet_b0, convnext_tiny)
TEXT_MODEL_NAME = "xlm-roberta-base"  # Multilingual text transformer

MAX_TEXT_LEN = 256

# Optimization
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-2
NUM_WORKERS = 2
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") != "" else "cpu"
USE_AMP = True

import os
import random
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, MultilabelStratifiedKFold
import src.config as config


def seed_everything(seed: int = 42):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def calculate_macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Computes macro-averaged ROC AUC across 12 target classes.
    y_true: np.ndarray of shape (N, 12)
    y_pred: np.ndarray of shape (N, 12)
    """
    auc_scores = []
    for i in range(y_true.shape[1]):
        # Handle single class presence case gracefully
        if len(np.unique(y_true[:, i])) > 1:
            score = roc_auc_score(y_true[:, i], y_pred[:, i])
            auc_scores.append(score)
        else:
            auc_scores.append(0.5)  # Fallback neutral AUC score
    return float(np.mean(auc_scores))


def create_folds(df: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """
    Creates K-Fold splits on train dataframe.
    """
    df = df.copy()
    df["fold"] = -1
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        df.loc[val_idx, "fold"] = fold
    return df

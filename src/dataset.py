"""
Dataset & Data Loading module for RSNA Knee Abnormality Detection.

Handles reading DICOM MRI slices, applying min-max pixel normalization,
sampling key representative slices across multi-series knee studies,
tokenizing radiology reports, and packaging inputs for multimodal training.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import src.config as config


def read_dicom_slice(path: Path) -> np.ndarray:
    """Reads a single DICOM file and scales pixels to 8-bit [0, 255]."""
    try:
        dicom = pydicom.dcmread(str(path))
        pixel_array = dicom.pixel_array.astype(np.float32)

        # Apply Rescale Slope / Intercept if available
        slope = float(getattr(dicom, 'RescaleSlope', 1.0))
        intercept = float(getattr(dicom, 'RescaleIntercept', 0.0))
        pixel_array = pixel_array * slope + intercept

        # Min-max normalize to 0-255
        p_min, p_max = pixel_array.min(), pixel_array.max()
        if p_max > p_min:
            pixel_array = (pixel_array - p_min) / (p_max - p_min) * 255.0
        else:
            pixel_array = np.zeros_like(pixel_array)
        return pixel_array.astype(np.uint8)
    except Exception:
        # Fallback for missing/corrupted slices
        return np.zeros(config.IMAGE_SIZE, dtype=np.uint8)


class KneeMRIDataset(Dataset):

    def __init__(
        self,
        df: pd.DataFrame,
        series_df: pd.DataFrame,
        images_dir: Path,
        tokenizer: AutoTokenizer,
        is_train: bool = True,
        max_slices: int = config.NUM_SLICES_PER_STUDY,
    ):
        self.df = df.reset_index(drop=True)
        self.series_df = series_df
        self.images_dir = images_dir
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.max_slices = max_slices

        self.transform = T.Compose([
            T.Resize(config.IMAGE_SIZE),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self) -> int:
        return len(self.df)

    def _load_study_images(self, study_id: str) -> torch.Tensor:
        """Loads and samples N middle slices for a given StudyInstanceUID."""
        matching_series = self.series_df[self.series_df["StudyInstanceUID"]
                                         == study_id]
        dicom_files: List[Path] = []

        for _, row in matching_series.iterrows():
            series_dir = self.images_dir / study_id / row["SeriesInstanceUID"]
            if series_dir.exists():
                files = sorted(list(series_dir.glob("*.dcm")))
                dicom_files.extend(files)

        if not dicom_files:
            # Fallback tensor if DICOMs are missing
            return torch.zeros(
                (self.max_slices, 3, config.IMAGE_SIZE[0],
                 config.IMAGE_SIZE[1]))

        # Subsample middle N slices
        total_files = len(dicom_files)
        if total_files > self.max_slices:
            indices = np.linspace(0,
                                  total_files - 1,
                                  self.max_slices,
                                  dtype=int)
            selected_files = [dicom_files[i] for i in indices]
        else:
            selected_files = dicom_files + [dicom_files[-1]] * (
                self.max_slices - total_files)

        slice_tensors = []
        for file_path in selected_files[: self.max_slices]:
            img_arr = read_dicom_slice(file_path)
            pil_img = Image.fromarray(img_arr).convert("RGB")
            slice_tensors.append(self.transform(pil_img))

        return torch.stack(slice_tensors)  # Shape: (num_slices, 3, H, W)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        study_id = row["StudyInstanceUID"]

        # 1. Image Slices
        images = self._load_study_images(study_id)

        # 2. Text Report
        report_text = str(row.get("Report", ""))
        if not report_text or report_text == "nan":
            report_text = "No report provided."

        text_encoding = self.tokenizer(
            report_text,
            max_length=config.MAX_TEXT_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "images": images,  # (N_slices, 3, H, W)
            "input_ids": text_encoding["input_ids"].squeeze(0),
            "attention_mask": text_encoding["attention_mask"].squeeze(0),
            "study_id": study_id,
        }

        # 3. Targets (if train/val)
        if self.is_train and all(c in row for c in config.TARGET_COLUMNS):
            labels = torch.tensor(row[config.TARGET_COLUMNS].values.astype(
                np.float32))
            item["labels"] = labels

        return item

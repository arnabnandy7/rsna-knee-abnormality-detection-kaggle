import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

import src.config as config
from src.dataset import KneeMRIDataset
from src.models import RSNAMultimodalModel


def main():
    print("Generating submission predictions...")
    device = config.DEVICE

    if not config.TEST_CSV.exists():
        print(f"Test CSV not found at {config.TEST_CSV}. Producing sample submission.")
        test_df = pd.read_csv(config.SAMPLE_SUBMISSION)
        series_df = pd.DataFrame(columns=["StudyInstanceUID", "SeriesInstanceUID"])
    else:
        test_df = pd.read_csv(config.TEST_CSV)
        series_df = pd.read_csv(config.TEST_SERIES_CSV)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    test_ds = KneeMRIDataset(test_df, series_df, config.TEST_IMAGES_DIR, tokenizer, is_train=False)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    fold_preds = []

    for fold in range(config.N_FOLDS):
        model_path = config.OUTPUT_DIR / f"model_fold_{fold}.pth"
        if not model_path.exists():
            print(f"Model checkpoint for fold {fold} not found at {model_path}. Skipping fold.")
            continue

        print(f"Predicting with fold {fold} model...")
        model = RSNAMultimodalModel().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"Fold {fold} Inference"):
                images = batch["images"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)

                with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                    logits = model(images, input_ids, attention_mask)
                    probs = torch.sigmoid(logits).cpu().numpy()

                preds.append(probs)

        fold_preds.append(np.concatenate(preds, axis=0))

    if fold_preds:
        avg_preds = np.mean(fold_preds, axis=0)
    else:
        print("Warning: No model checkpoints loaded. Filling submission with 0.5 default probabilities.")
        avg_preds = np.full((len(test_df), config.NUM_CLASSES), 0.5)

    sub_df = pd.DataFrame(avg_preds, columns=config.TARGET_COLUMNS)
    sub_df.insert(0, "StudyInstanceUID", test_df["StudyInstanceUID"])

    submission_path = Path("submission.csv")
    sub_df.to_csv(submission_path, index=False)
    print(f"Successfully generated {submission_path} with shape {sub_df.shape}")
    print(sub_df.head())


if __name__ == "__main__":
    main()

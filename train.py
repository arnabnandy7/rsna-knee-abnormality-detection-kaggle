import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

import src.config as config
from src.dataset import KneeMRIDataset
from src.models import RSNAMultimodalModel
from src.utils import seed_everything, calculate_macro_auc, create_folds


def train_epoch(model, dataloader, optimizer, criterion, scaler, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        images = batch["images"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=config.USE_AMP):
            logits = model(images, input_ids, attention_mask)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * len(labels)

    return running_loss / len(dataloader.dataset)


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["images"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.cuda.amp.autocast(enabled=config.USE_AMP):
                logits = model(images, input_ids, attention_mask)
                loss = criterion(logits, labels)

            running_loss += loss.item() * len(labels)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    val_auc = calculate_macro_auc(all_targets, all_preds)
    val_loss = running_loss / len(dataloader.dataset)
    return val_loss, val_auc, all_preds


def main():
    seed_everything(config.SEED)
    print(f"Using device: {config.DEVICE}")

    if not config.TRAIN_CSV.exists():
        print(f"Data file not found at {config.TRAIN_CSV}. Please configure data paths.")
        return

    df_train = pd.read_csv(config.TRAIN_CSV)
    df_series = pd.read_csv(config.TRAIN_SERIES_CSV)
    df_train = create_folds(df_train, n_folds=config.N_FOLDS, seed=config.SEED)

    tokenizer = AutoTokenizer.from_pretrained(config.TEXT_MODEL_NAME)
    criterion = nn.BCEWithLogitsLoss()

    for fold in range(config.N_FOLDS):
        print(f"\n========== Training Fold {fold+1}/{config.N_FOLDS} ==========")
        train_df = df_train[df_train["fold"] != fold]
        val_df = df_train[df_train["fold"] == fold]

        train_ds = KneeMRIDataset(train_df, df_series, config.TRAIN_IMAGES_DIR, tokenizer, is_train=True)
        val_ds = KneeMRIDataset(val_df, df_series, config.TRAIN_IMAGES_DIR, tokenizer, is_train=True)

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

        model = RSNAMultimodalModel().to(config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        scaler = torch.cuda.amp.GradScaler(enabled=config.USE_AMP)

        best_auc = 0.0

        for epoch in range(config.EPOCHS):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, config.DEVICE)
            val_loss, val_auc, _ = validate(model, val_loader, criterion, config.DEVICE)

            print(f"Epoch {epoch+1}/{config.EPOCHS} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Macro AUC: {val_auc:.4f}")

            if val_auc > best_auc:
                best_auc = val_auc
                save_path = config.OUTPUT_DIR / f"model_fold_{fold}.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Saved best model checkpoint to {save_path} (AUC: {best_auc:.4f})")

        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

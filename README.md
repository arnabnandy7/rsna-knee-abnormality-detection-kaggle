# RSNA Knee Abnormality Detection 2026 - Solution Workspace

This repository contains a baseline multimodal deep learning framework for the **RSNA Knee Abnormality Detection** Kaggle Competition.

---

## Task & Metric Summary

* **Objective**: Predict 12 target binary findings from DICOM knee MRI scans and multilingual radiology report text:
  `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`
* **Evaluation Metric**: **Macro-averaged Area Under the ROC Curve (Macro AUC)**:
  $$\text{Final Score} = \frac{1}{12} \sum_{i=1}^{12} \text{AUC}_i$$

---

## Repository Structure

```
├── rsna_knee_multimodal_baseline.ipynb   # Standalone Kaggle GPU Notebook ready for upload
├── train.py                              # Modular 5-Fold cross-validation script
├── predict_submission.py                 # Test inference & submission generator
├── requirements.txt                      # Project dependencies
└── src/
    ├── config.py                         # Hyperparameters & paths
    ├── dataset.py                        # PyTorch DICOM slice sampler & text tokenizer
    ├── models.py                         # Late/Joint Multimodal Fusion architecture
    └── utils.py                          # Macro AUC evaluator & seed setup
```

---

## How to Run on Kaggle (Recommended - Free GPU)

Since training on full multi-slice DICOM MRI datasets requires high compute, run the notebook directly on Kaggle:

1. **Go to Kaggle**: Open [Kaggle Competitions - RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
2. **Create Notebook**: Click **Code** -> **New Notebook**.
3. **Upload Notebook**: Click **File** -> **Upload Notebook** -> Select [`rsna_knee_multimodal_baseline.ipynb`](./rsna_knee_multimodal_baseline.ipynb).
4. **Enable GPU**: On the right sidebar, set **Accelerator** to **GPU T4 x2** or **GPU P100 / L4**.
5. **Attach Competition Data**: Click **+ Add Data** -> Search for `rsna-knee-abnormality-detection` -> Click **Add**.
6. **Run & Submit**: Click **Run All**. The notebook will train the model and output `submission.csv`!

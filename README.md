# RSNA Knee Abnormality Detection 2026

A deep learning framework and competition solution for the **RSNA Knee Abnormality Detection** Kaggle Competition.

---

## Metric & Competition Summary

* **Objective**: Predict 12 target binary musculoskeletal findings from DICOM multi-series knee MRI scans and clinical radiology reports:
  * **Ligaments**: `ACL`, `MCL`
  * **Menisci**: `Medial Meniscus`, `Lateral Meniscus`
  * **Osteoarthritis**: `Medial OA`, `Lateral OA`, `PF OA`
  * **Fluid & Joint**: `Effusion`, `Synovitis`, `Baker's`
  * **Bone**: `Contusion`, `Fracture`
* **Evaluation Metric**: **Macro-averaged Area Under the ROC Curve (Macro AUC)**:
  $$\text{Final Score} = \frac{1}{12} \sum_{i=1}^{12} \text{AUC}_i$$

---

## Solution Highlights & Top Model

### Flagship Submission: `bend-the-knee-to-the-dinosaurs.ipynb`
* **Public LB Score**: **0.958** (Top ~3% / Rank 195 of 6,077)
* **Architecture & Strategy**:
  1. **Multi-Resolution CoAtNet**: Dual-scale 384px and 224px feature representations capturing fine cartilage tears and global joint alignments.
  2. **Flip Test-Time Augmentation (TTA)**: Bilateral MRI symmetry exploited via horizontal slice mirroring on compact R6 models.
  3. **High-Speed DICOM Reader Engine (`_fastread.py` & `own_src/`)**: Zero-overhead single-pass binary header parsing with descriptor caching, eliminating I/O bottlenecks under Kaggle's 9-hour execution limit.
  4. **Per-Finding Rank Blending**: Calibrated percentile ranking weighted per abnormality rather than raw logit averaging.

---

## Repository Structure

```
├── bend-the-knee-to-the-dinosaurs.ipynb # Flagship competition submission notebook (LB 0.958)
├── rsna_knee_multimodal_baseline.ipynb  # End-to-end multimodal baseline pipeline
├── train.py                             # Modular 5-Fold cross-validation entrypoint
├── predict_submission.py                # Ensemble inference & submission generator
├── requirements.txt                     # Project dependencies
├── kernel-metadata.json                 # Kaggle CLI deployment configuration
├── src/                                 # Modular framework package
│   ├── config.py                        # Paths, hyperparameters, and target definitions
│   ├── dataset.py                       # DICOM slice sampler & tokenized dataset loader
│   ├── models.py                        # Multimodal vision-language fusion architecture
│   └── utils.py                         # Macro-AUC evaluator, seeding, and fold creation
├── own_src/                             # Standalone inference engine modules
│   ├── infer.py                         # Fast test-time inference pipeline
│   ├── knee.py                          # Neural backbone definitions
│   └── preprocess.py                    # Windowing, cropping, and slice normalization
├── model_zoo/                           # 56 model training configurations + scorers
│   ├── train_<backbone>_5fold/          # Isolated 5-fold training setups ready for Kaggle push
│   └── scorer_slot2/                    # Live inference scoring staging
├── notebooks/                           # Organized Jupyter notebook workspace
│   ├── ensembles/                       # 10-model, 15-model, and 20-model blend notebooks
│   ├── trainers/                        # DINOv2, RadImageNet head, and MRNet pretrainers
│   ├── experiments/                     # Iterative Speedy Raptors (v16-v45) notebooks
│   └── utils/                           # Weight downloaders and offline assets
├── scripts/                             # Data preparation & utility scripts
│   ├── extract_labels_llm.py            # LLM radiology report label extraction
│   ├── extract_labels_v2.py             # Rule-based radiology report parser
│   ├── _fastread.py                     # High-performance DICOM read benchmark engine
│   └── download_dinov2.py               # Pretrained foundation weight fetcher
├── annotations/                         # Processed annotations & pseudo-labels
│   └── llm_labels.csv                   # Continuous extracted target probabilities
└── benchmarks/                          # Leaderboard verification scores & receipts
```

---

## Model Zoo & Training Fleet (`train_*`)

The repository contains **56 isolated 5-fold training configurations** set up for Kaggle GPU execution:

* **ConvNeXt Family**: ConvNeXt-Tiny, ConvNeXt-Small, ConvNeXt-Base (Raptor & Specialist setups)
* **DenseNet Family**: DenseNet-121, DenseNet-161, DenseNet-169, DenseNet-201
* **EfficientNet Family**: EfficientNet-B0 through B5, EfficientNetV2-S, EfficientNetV2-M
* **ResNet / ResNeXt**: ResNet-34 (GeM pooling), ResNet-50d, ResNet-101d, ResNet-152d, SE-ResNet-50/101, SE-ResNeXt-50, Wide-ResNet-50
* **Vision Transformers**: Swin-Tiny, CoAtNet-224, CoAtNet-384
* **Pathology Specialists**: Fluid/Synovitis specialist, Soft Tissue specialist, Bone/OA specialist, and Slice-Sequence BiGRU readers

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/arnabnandy7/rsna-knee-abnormality-detection-kaggle.git
cd rsna-knee-abnormality-detection-kaggle
pip install -r requirements.txt
```

### 2. Local Training

Ensure competition data is placed in `./data/`:

```bash
python train.py
```

### 3. Inference & Submission

```bash
python predict_submission.py
```

### 4. Running on Kaggle

1. Open [Kaggle Competitions - RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).
2. Create a new notebook with GPU accelerator enabled (T4 x2 or P100 / L4).
3. Upload `bend-the-knee-to-the-dinosaurs.ipynb` (or deploy via `kaggle kernels push`).
4. Attach required competition data and dataset weights.
5. Run notebook to generate `submission.csv`.

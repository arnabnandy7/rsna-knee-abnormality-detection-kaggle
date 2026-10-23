# Notebooks Directory

This folder contains organized Jupyter notebooks for the RSNA Knee Abnormality Detection competition.

## Directory Structure

* **`experiments/`**: Iterative model exploration and experimental checkpoints:
  * `rsna-knee-speedy-raptors-v16.ipynb` through `v41.ipynb`: Evolution of the Raptor multi-backbone pipelines, testing feature pooling, flip-TTA, and learning schedules.
  * `rsna-knee-v42-convnext-mil.ipynb` through `v45-convnext-w15.ipynb`: Multiple Instance Learning (MIL) and gated ConvNeXt series-level aggregators with varying window sizes.
* **`ensembles/`**: Standalone multi-resolution blend notebooks:
  * `rsna_knee_10model_ensemble.ipynb`: 10-model inference blend.
  * `rsna_knee_15model_ensemble.ipynb`: Multi-resolution 15-model ensemble.
  * `rsna_knee_20model_ensemble.ipynb`: 20-model blend across heterogeneous backbones.
* **`trainers/`**: Backbone and foundation model pretraining:
  * `rsna_custom_radimagenet_head_trainer.ipynb`: Custom classification head trained on RadImageNet features.
  * `rsna_dinov2_cross_series_trainer.ipynb`: DINOv2-S/14 cross-series self-supervised feature trainer.
  * `rsna_resnet34_2p5d_trainer.ipynb`: 2.5D ResNet-34 slice-stack trainer.
  * `mrnet_knee_pretraining.ipynb`: Domain pretraining pipeline on the Stanford MRNet knee dataset.
* **`utils/`**: Pretrained model weight downloaders:
  * `download_weights.ipynb`: Automated weight retrieval for offline Kaggle kernel inference.
  * `download_pretrained_weights.ipynb`: Pretrained EfficientNet-B2 weight downloader.

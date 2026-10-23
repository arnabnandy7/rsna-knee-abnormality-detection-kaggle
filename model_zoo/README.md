# Model Zoo & Training Fleet

Contains 56 isolated 5-fold training configurations and scoring setups ready for Kaggle GPU training (`kaggle kernels push`).

## Categories

* **CoAtNet**:
  * `train_coatnet_5fold/`, `train_coatnet_224_c08_5fold/`, `train_coatnet_sota_refit/`
* **ConvNeXt**:
  * `train_convnext/`, `train_convnext_full/`, `train_convnext_tiny_5fold/`, `train_convnext_small_5fold/`, `train_convnext_kneenet_5fold/`, `train_convnext_raptor_5fold/`, `train_convnext_specialist_5fold/`
* **DenseNet**:
  * `train_densenet121_5fold/`, `train_densenet161_5fold/`, `train_densenet169_5fold/`, `train_densenet201_5fold/` (along with Raptor & specialist variants)
* **EfficientNet**:
  * `train_effnet_b0_5fold/` through `b5_5fold/`, `train_effnetv2_s_5fold/`, `train_effnetv2_m_5fold/`
* **ResNet / ResNeXt**:
  * `train_resnet34_gem_5fold/`, `train_resnet50d_5fold/`, `train_resnet101d_5fold/`, `train_resnet152d_5fold/`, `train_resnext50_5fold/`, `train_seresnet50_5fold/`, `train_seresnext50_5fold/`, `train_wide_resnet50_5fold/`
* **Vision Transformers & Specialists**:
  * `train_swin_kneenet_5fold/`, `train_swin_tiny_raptor_5fold/`
  * `train_fluid/`, `train_fluid_5fold/` (Synovitis & joint effusion)
  * `train_soft_tissue/`, `train_soft_tissue_5fold/` (Ligaments & menisci)
  * `train_oa_bone_5fold/` (Osteoarthritis & contusion/fracture)
  * `train_slice_bigru_5fold/` (Slice sequence recurrent aggregator)
* **Scorers**:
  * `scorer_slot2/`: Apex scorer deployment configuration.

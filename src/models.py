import torch
import torch.nn as nn
import timm
from transformers import AutoModel
import src.config as config


class ImageEncoder(nn.Module):
    """
    Extracts features from multiple 2D DICOM slices per study using a timm backbone
    and aggregates them via Adaptive Average/Max Pooling.
    """
    def __init__(self, model_name: str = config.IMAGE_MODEL_NAME, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        self.feature_dim = self.backbone.num_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, num_slices, 3, H, W)
        b, s, c, h, w = x.shape
        x_flat = x.view(b * s, c, h, w)
        features = self.backbone(x_flat)  # (b * s, feature_dim)
        features = features.view(b, s, self.feature_dim)
        
        # Aggregate across slices (mean pool)
        study_features = features.mean(dim=1)  # (b, feature_dim)
        return study_features


class TextEncoder(nn.Module):
    """
    Extracts text embeddings from raw radiology reports using a transformer backbone.
    """
    def __init__(self, model_name: str = config.TEXT_MODEL_NAME, pretrained: bool = True):
        super().__init__()
        self.transformer = AutoModel.from_pretrained(model_name)
        self.feature_dim = self.transformer.config.hidden_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over token embeddings
        token_embeddings = outputs.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        text_features = sum_embeddings / sum_mask
        return text_features


class RSNAMultimodalModel(nn.Module):
    """
    Late/Joint Multimodal Fusion network combining DICOM image slice features
    and radiology report text embeddings to classify 12 knee abnormalities.
    """
    def __init__(
        self,
        image_model_name: str = config.IMAGE_MODEL_NAME,
        text_model_name: str = config.TEXT_MODEL_NAME,
        num_classes: int = config.NUM_CLASSES,
        pretrained: bool = True,
    ):
        super().__init__()
        self.image_encoder = ImageEncoder(image_model_name, pretrained=pretrained)
        self.text_encoder = TextEncoder(text_model_name, pretrained=pretrained)
        
        fusion_dim = self.image_encoder.feature_dim + self.text_encoder.feature_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        img_feats = self.image_encoder(images)
        text_feats = self.text_encoder(input_ids, attention_mask)
        
        fusion_feats = torch.cat([img_feats, text_feats], dim=1)
        logits = self.classifier(fusion_feats)
        return logits

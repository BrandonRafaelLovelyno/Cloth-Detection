import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class FasterRCNNResNet50(torch.nn.Module):
    def __init__(self, num_classes=14):
        super(FasterRCNNResNet50, self).__init__()
        
        # Backbone with FPN
        backbone = resnet_fpn_backbone('resnet50', weights=torchvision.models.ResNet50_Weights.IMAGENET1K_V2)
        
        # Freeze early layers to focus on fine-tuning higher-level features
        for i, layer in enumerate(backbone.body.children()):
            if i < 6:
                for param in layer.parameters():
                    param.requires_grad = False 

        # RoI Pooler with modified settings
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,  # Increase to capture more details
            sampling_ratio=4  # Higher sampling ratio for better feature extraction
        )

        # Faster R-CNN model definition
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            box_roi_pool=roi_pooler
        )

        # Replace the box predictor to match the number of classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

    def forward(self, images, targets=None):
        """
        Forward method for the Faster R-CNN model.
        
        Args:
            images (Tensor): Images for the model.
            targets (list): List of target dictionaries for training (optional).
        
        Returns:
            output (Tensor): Model outputs during inference.
        """
        return self.model(images, targets)

import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


backbone = resnet_fpn_backbone('resnet101',weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
backbone.out_channels = 256

roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)

class FasterRCNNResNet101(torch.nn.Module):
    def __init__(self, num_classes=13):
        super(FasterRCNNResNet101, self).__init__()
        
        self.model = FasterRCNN(
            backbone,
            num_classes=20,  
            box_roi_pool=roi_pooler
        )

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
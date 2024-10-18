import torch
import torchvision
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

class MaskRCNNResNet101(torch.nn.Module):
    def __init__(self, num_classes=14, nms_iou_threshold=0.5, score_threshold=0.3):
        super(MaskRCNNResNet101, self).__init__()
        
        # Backbone with FPN
        backbone = resnet_fpn_backbone('resnet101', weights=torchvision.models.ResNet101_Weights.IMAGENET1K_V2)
        
        # Freeze early layers to focus on fine-tuning higher-level features
        for i, layer in enumerate(backbone.body.children()):
            if i < 50:
                for param in layer.parameters():
                    param.requires_grad = False 

        # RoI Pooler with modified settings
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=14,  
            sampling_ratio=4  # Higher sampling ratio for better feature extraction
        )

        # Faster R-CNN model definition
        self.model = MaskRCNN(
            backbone,
            num_classes=num_classes,
            box_roi_pool=roi_pooler
        )

        # Add a mask predictor
        in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        self.model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
            in_features_mask,
            hidden_layer,
            num_classes
        )

    
        # Store NMS parameters
        self.nms_iou_threshold = nms_iou_threshold
        self.score_threshold = score_threshold

    def forward(self, images, targets=None):
        """
        Forward method for the Mask R-CNN model.
        
        Args:
            images (Tensor): Images for the model.
            targets (list): List of target dictionaries for training (optional).
        
        Returns:
            output (Tensor): Model outputs during inference.
        """
        # Get the raw predictions from the model
        outputs = self.model(images, targets)
        
        if self.training:
            # During training, return the raw output (losses)
            return outputs
        
        # Apply NMS during evaluation
        filtered_outputs = []
        for output in outputs:
            boxes = output['boxes']
            scores = output['scores']
            labels = output['labels']
            masks = output['masks']  # Get the predicted masks (if available)

            # Filter out low-confidence scores
            high_score_idx = scores > self.score_threshold
            boxes = boxes[high_score_idx]
            scores = scores[high_score_idx]
            labels = labels[high_score_idx]
            masks = masks[high_score_idx]

            # Apply NMS
            nms_idx = torchvision.ops.nms(boxes, scores, self.nms_iou_threshold)
            boxes = boxes[nms_idx]
            scores = scores[nms_idx]
            labels = labels[nms_idx]
            masks = masks[nms_idx]

            # Store the filtered results
            result = {
                'boxes': boxes,
                'scores': scores,
                'labels': labels,
                'masks': masks
            }
            
            filtered_outputs.append(result)

        return filtered_outputs

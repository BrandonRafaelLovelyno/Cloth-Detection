from PIL import ImageDraw
import torch
import sys
sys.path.append('../')

def check_two_dimensional_list(bbox):
    return all(isinstance(box, bbox) and len(box) == 4 for box in bbox)

def draw_all_item_boxes(image, bbox_tensor, category_id):
    draw = ImageDraw.Draw(image)
    
    if isinstance(bbox_tensor, torch.Tensor):
        bbox_tensor = bbox_tensor.int().tolist() 
    
    if isinstance(bbox_tensor, list) and len(bbox_tensor) > 0:
        for i, box in enumerate(bbox_tensor):
            if len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline="red" if i % 2 == 0 else "blue", width=2)

    return image

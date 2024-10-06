from PIL import ImageDraw
import torch
import sys
sys.path.append('../')

category_dict = {
    1: "short sleeve top",
    2: "long sleeve top",
    3: "short sleeve outwear",
    4: "long sleeve outwear",
    5: "vest",
    6: "sling",
    7: "shorts",
    8: "trousers",
    9: "skirt",
    10: "short sleeve dress",
    11: "long sleeve dress",
    12: "vest dress",
    13: "sling dress"
}

def check_two_dimensional_list(bbox):
    return all(isinstance(box, bbox) and len(box) == 4 for box in bbox)

def translate_label(label_tensor):
    return category_dict[label_tensor]

def draw_all_item_boxes(image, bbox_tensor, labels_tensor):
    draw = ImageDraw.Draw(image)
    
    if isinstance(bbox_tensor, torch.Tensor):
        bbox_tensor = bbox_tensor.int().tolist() 
    
    if isinstance(labels_tensor, torch.Tensor):
        labels_tensor = labels_tensor.int().tolist()
    
    if isinstance(bbox_tensor, list) and len(bbox_tensor) > 0:
        for i, box in enumerate(bbox_tensor):
            if len(box) == 4:
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline="red" if i % 2 == 0 else "blue", width=2)
                
                label = translate_label(labels_tensor[i])
                text_position = (x1, y1 - 10)   
                draw.text(text_position, label, fill="white",size = 100)
    
    return image
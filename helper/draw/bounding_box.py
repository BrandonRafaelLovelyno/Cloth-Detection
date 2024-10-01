from PIL import ImageDraw

import sys
sys.path.append('../')
import annotation

def check_two_dimensional_list(bbox):
    return all(isinstance(box, bbox) and len(box) == 4 for box in bbox)

def draw_all_item_boxes(image, path):
    draw = ImageDraw.Draw(image)
    
    # Extract bounding box(es)
    bbox = annotation.extract_attributes('bounding_box', path)
    
    # Check if bbox is a 2D list
    if isinstance(bbox, list):
        if check_two_dimensional_list(bbox): 
            for i,box in enumerate(bbox):
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle([x1, y1, x2, y2], outline="white" if i%2==0 else "red", width=2)
                
        elif len(bbox) == 4:  
            x1, y1, x2, y2 = map(int, bbox)
            draw.rectangle([x1, y1, x2, y2], outline="white", width=2)
    
    return image

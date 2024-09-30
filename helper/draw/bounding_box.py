from PIL import ImageDraw
import json

def draw_all_item_boxes(image, dir):
    with open(dir) as f:
        anno = json.load(f)

    draw = ImageDraw.Draw(image)

    for i,item_key in enumerate(anno.keys()):
        item_value = anno[item_key]
        
        if isinstance(item_value, dict) and 'bounding_box' in item_value:
            bbox = item_value['bounding_box']
        
            if isinstance(bbox, list) and len(bbox) == 4:
                x1, y1, x2, y2 = map(int, bbox) 
                draw.rectangle([x1, y1, x2, y2], outline="white" if i%2==0 else 'red', width=2)  
                
                
    return image

def print_hello():
    print('hello')
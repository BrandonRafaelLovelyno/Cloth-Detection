import torch
from PIL import ImageDraw
import json

def make_polygon_tuples(polygon_tensor):
    polygon_tuples = [(polygon_tensor[i], polygon_tensor[i + 1]) for i in range(0, len(polygon_tensor), 2)]
    return polygon_tuples

def draw_single_polygon(image, polygon_tensor, color='white'):
    polygon_tuples = make_polygon_tuples(polygon_tensor)
    draw = ImageDraw.Draw(image)
    draw.polygon(polygon_tuples, outline=color)

    return image

def draw_item_polygons(image, polygons,color='white'):
    for i,polygon in enumerate(polygons):
        polygon_tensor = torch.tensor(polygon)
        image = draw_single_polygon(image, polygon_tensor, color=color)

    return image

def draw_all_item_polygons(image, dir):
    with open(dir) as f:
        anno = json.load(f)
    
    for i,item_key in enumerate(anno.keys()):
        item_value = anno[item_key]
        if isinstance(item_value, dict) and 'segmentation' in item_value:
            segmentation = item_value['segmentation']
            
            if isinstance(segmentation, list):
                image = draw_item_polygons(image, segmentation,color='white' if i%2==0 else 'red')
                
    return image
import torch
from PIL import ImageDraw

import sys
sys.path.append('../')
import annotation

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

def check_two_dimensional_list(segmentation):
    return all(isinstance(box, segmentation) for box in segmentation)

def draw_all_item_polygons(image, path):
    segmentation = annotation.extract_attributes('segmentation', path)

    if isinstance(segmentation, list):
        if check_two_dimensional_list(segmentation):
            for i,polygon in enumerate(segmentation):
                image = draw_item_polygons(image, polygon,color='white' if i == 0 else 'red')
        elif len(segmentation) == 4:
            image = draw_item_polygons(image, segmentation,color='white')
                
    return image
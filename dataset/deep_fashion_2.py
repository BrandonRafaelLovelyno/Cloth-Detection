import torchvision.transforms as transforms
import torch
import numpy as np
import PIL.Image as Image
import json
import lmdb
from io import BytesIO

transform = transforms.Compose([
    transforms.Resize((800, 800)),
    transforms.ToTensor()
])

def convert_index_to_id(index):
    return str(index+1).zfill(6)

def get_image_path(image_dir, index):
    return image_dir + convert_index_to_id(index) + ".jpg"

def extract_attribute(anno,key):
    attributes = []
    
    for i,item_key in enumerate(anno.keys()):
        item_value = anno[item_key]
        
        if isinstance(item_value, dict) and key in item_value:
            attribute = item_value[key]
        
            if isinstance(attribute, list):
                attributes.append(attribute)
            else:
                if attribute is not None:
                    attributes.append(attribute)
                else :
                    return None
    
    return attributes

class DeepFashion2Dataset(torch.utils.data.Dataset):
    def __init__(self, lmdb_path, data_length = np.inf):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, max_readers=8, readahead=False, meminit=False)
        self.data_length = data_length
        

    def __len__(self):
        return self.data_length

    def __del__(self):
        self.txn.commit() 
        self.env.close()  

    def __getitem__(self, index):        
        data_id = convert_index_to_id(index)
        image_key, annotation_key = f'image_{data_id}'.encode('utf-8'), f'annotation_{data_id}'.encode('utf-8')
        
        
        with self.env.begin(write=False) as txn:
            # Load image
            image_data = txn.get(image_key)
            image_buffer = BytesIO(image_data)
            image = Image.open(image_buffer).convert('RGB')
            image_tensor = transform(image)
            
            
            # Load annotation
            annotation_data = txn.get(annotation_key)
            annotation = json.loads(annotation_data.decode('utf-8'))
            
            bbox = extract_attribute(annotation, 'bounding_box')
            labels = extract_attribute(annotation, 'category_id')
        
        bbox_arr = np.array(bbox,np.float32)
        labels_arr = np.array(labels,np.int64)
    
        target_tensor = {
            "boxes": torch.tensor(bbox_arr, dtype=torch.float32),  
            "labels" : torch.tensor(labels_arr, dtype=torch.int64)
        }

        return image_tensor, target_tensor

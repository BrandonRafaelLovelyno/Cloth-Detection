import json

def extract_attributes(key, path):
    attributes = []
    
    with open(path) as f:
        anno = json.load(f)

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

import numpy as np

def append_list_item(item_group, key, value):
    dimention = get_list_dimension(value)
    
    if dimention == 1:
        data_array = np.array(value, dtype='float32')
        item_group.create_dataset(key, data=data_array)
    
    elif dimention == 2:
        for idx, segment in enumerate(value):
            if isinstance(segment, list):
                segment_array = np.array(segment, dtype='float32')
                
                item_group.create_dataset(f"{key}_{idx}", data=segment_array)

    else :
        print("Found more than 2 dimention list data")
        return


def get_list_dimension(data):
    if isinstance(data, list):
        if not data: 
            return 1
        
        first_element = data[0]
        dimension = 1
        
        while isinstance(first_element, list):
            dimension += 1
            if not first_element: 
                break
            first_element = first_element[0]  

        return dimension
    return 0  


# training
data_path = {
    "train": "./data/train",
    "val": "./data/validation",
}

train_start_index = 15000
train_end_index = 25000    
batch_size = 4

val_start_index = 0
val_end_index = 15000

is_import_model = False
model_path = "./weight/faster_rcnn_resnet101_v1.pth"
save_path = "./weight/faster_rcnn_resnet101_v2.pth"

num_epochs = 3

import numpy as np
import torch
import torchvision
# import cv2
import base64
from torch.utils.data import Dataset, DataLoader, Subset
import matplotlib.pyplot as plt
import torch.nn.functional as F
import matplotlib.patches as patches
import torch.nn as nn
import random
import matplotlib.image as mpimg
from skimage.transform import resize
from torchvision import transforms
from PIL import Image
from draw_boundary_box import draw_box
from yolo_cnn import MyCnn
from data_generator import mnist_dataset
from yolo_loss import YOLOLoss

train_ds = torchvision.datasets.MNIST(root="data", train=True,  download=True, transform=torchvision.transforms.ToTensor())
train_ds = mnist_dataset(train_ds)


device = "cuda" if torch.cuda.is_available() else "cpu"
myCnn = MyCnn().to(device)
model_path = 'mult_obj_multi_cell_mnist_bw-0_save.pt'
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))  # Use 'cuda' if you saved the model on GPU
loss = YOLOLoss()
# Step 3: Load the state dictionary into your model
myCnn.load_state_dict(checkpoint)

image_path = "1_2.png"
image = mpimg.imread(image_path)
image = resize(image,(56,56,1))
image = (image >= train_ds.bw_threshold).astype(np.uint8)
image = np.moveaxis(image, -1, 0)
# Step 3: Plot the image using plt.imshow

image = torch.tensor(image).unsqueeze(0)
image = image.to(device)
image = image.to(torch.float32)
with torch.no_grad():
    logits = myCnn(image)
    _,_, class_indexes = loss.extracting_predictions()  
    class_probabilities = nn.functional.softmax(logits[:,class_indexes], dim=1)
    class_predictions = torch.argmax(class_probabilities.view(1,train_ds.num_grid,train_ds.num_classes),dim =2)
    print(class_predictions)
    # _,coords_indexes, class_indexes = loss.extracting_predictions()
    # iou += calculate_iou(logits[:,coords_indexes], y[:,coords_indexes])
    # class_probabilities = nn.functional.softmax(logits[:,class_indexes], dim=1)
    # class_predictions = torch.argmax(class_probabilities.view(validation_batch_size,train_ds.num_grid,train_ds.num_classes),dim =2)
    # class_gt = torch.argmax(y[:,class_indexes].view(validation_batch_size,train_ds.num_grid,train_ds.num_classes),dim =2)
    # total += y.size(0)*train_ds.num_grid
    # correct += (class_predictions == class_gt).sum().item()
    draw_box(image[0].cpu(), logits[0].cpu(),train_ds)
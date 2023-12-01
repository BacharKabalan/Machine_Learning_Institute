import fastapi
import base64
import numpy
import cv2
import numpy as np
import torch
import torchvision
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
import io



app = fastapi.FastAPI()



@app.on_event("startup")
async def startup_event():
  app.state.device = "cuda" if torch.cuda.is_available() else "cpu"  
  app.state.train_ds = torchvision.datasets.MNIST(root="data", train=True,  download=True, transform=torchvision.transforms.ToTensor())
  app.state.train_ds = mnist_dataset(app.state.train_ds)
  app.state.loss = YOLOLoss()
  app.state.digit = MyCnn().to(app.state.device)
  model_path = 'mult_obj_multi_cell_mnist_bw-1.pt'   
  app.state.digit.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
  app.state.nums = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
  app.state.digit.eval()
    


@app.on_event("shutdown")
async def shutdown_event():
  print("Shutting down")


@app.get("/")
def on_root():
  return { "message": "Hello App" }


@app.post("/one_number")
async def one_number(request: fastapi.Request):
  _,_, class_indexes = app.state.loss.extracting_predictions()
  raw = (await request.json())["img"]
  raw = raw.split(',')[1]
  npArr = numpy.frombuffer(base64.b64decode(raw), numpy.uint8)
  image = resize(npArr,(56,56,1))
  image = (image >= app.state.train_ds.bw_threshold).astype(np.uint8)
  image = np.moveaxis(image, -1, 0)
  image = torch.tensor(image).unsqueeze(0)
  image = image.to(app.state.device)
  image = image.to(torch.float32)
  logits = app.state.digit(image)
  class_probabilities = nn.functional.softmax(logits[:,class_indexes], dim=1)
  class_predictions = torch.argmax(class_probabilities.view(1,app.state.train_ds.num_grid,app.state.train_ds.num_classes),dim =2)
  probabilities = torch.zeros(10,1)
  probabilities[class_predictions[0]] =  class_probabilities.view(1,4,10)[0,[0,1,2,3],class_predictions].squeeze().view(4,1)
  result = [{"class": str(i), "value": prob.item()} for i, prob in enumerate(probabilities)]
  drawn_image = draw_box(image[0].cpu(), logits[0].cpu(),app.state.train_ds)
  buffer = io.BytesIO()
  torchvision.transforms.functional.to_pil_image(drawn_image).save(buffer, format="PNG")  
  base64_image = base64.b64encode(buffer.getvalue()).decode()
  base64_image = f"data:image/png;base64,{base64_image}"
  return { "img": base64_image }

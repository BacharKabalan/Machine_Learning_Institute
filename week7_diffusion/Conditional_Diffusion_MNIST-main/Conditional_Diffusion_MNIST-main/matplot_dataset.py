import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import random



class matplot_dataset():
    
    def __init__(self, directory,image_height): 
        self.output_dir = directory
        self.image_list = []
        self.image_height = image_height
        for filename in os.listdir(directory): 
            file_path = os.path.join(directory, filename)
            self.image_list.append(file_path)
    
    def __len__(self):
        return len(self.image_list)

    def __getitem__(self,idx):
        image_path = self.image_list[idx]
        label = int(image_path[-5])
        img = Image.open(image_path)
        transform =transforms.Compose([transforms.ToTensor(),transforms.Resize((self.image_height,self.image_height))])
        
        # Apply the transformation to the image
        tensor_image = transform(img)
        
        # Now, tensor_image is a PyTorch tensor
        rgb_image = tensor_image[:3, :, :]
        
        # Convert tensor to PIL Image
        # pil_image = transforms.ToPILImage()(rgb_image)
        # print(type(pil_image))
        # # Display the image
        # plt.imshow(pil_image)
        # print(rgb_image.size())
        # plt.axis('off')  # Turn off axis labels
        # plt.show()
        return rgb_image, label


                    
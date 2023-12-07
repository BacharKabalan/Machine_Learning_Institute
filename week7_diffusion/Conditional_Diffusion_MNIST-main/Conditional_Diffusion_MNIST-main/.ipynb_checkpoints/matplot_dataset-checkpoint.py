import os
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt



class matplot_dataset():
    
    def __init__(self, directory): 
        self.output_dir = directory
        self.image_list = []
        for filename in os.listdir(directory): 
            file_path = os.path.join(directory, filename)
            self.image_list.append(file_path)


    def __getitem__(self,idx):
        image_path = self.image_list[idx]
        
        img = Image.open(image_path)
        img.show()
        transform = transforms.ToTensor()
    
        # Apply the transformation to the image
        tensor_image = transform(img)
        
        # Now, tensor_image is a PyTorch tensor
        print(tensor_image.shape)
        rgb_image = tensor_image[:3, :, :]
        
        # Convert tensor to PIL Image
        pil_image = transforms.ToPILImage()(rgb_image)
        
        # Display the image
        plt.imshow(pil_image)
        print(rgb_image.size())
        plt.axis('off')  # Turn off axis labels
        plt.show()
        return rgb_image
            
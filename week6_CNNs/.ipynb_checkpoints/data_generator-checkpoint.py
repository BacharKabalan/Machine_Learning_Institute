import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn as nn
import random
from skimage.transform import resize

class mnist_dataset(Dataset):
    def __init__(self, original_dataset):
        self.original_dataset = original_dataset
        self.image_size = 56
        self.grids_size = 28
        self.num_grid = int(2*self.image_size/self.grids_size)
        self.num_classes = 10
        self.labels_size = 15
        self.num_images = 0
        self.bw_threshold = 0.5
    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self,index):
        multiple_object_image = torch.zeros((self.image_size,self.image_size))
        image_labels = torch.zeros(int(self.num_grid*(self.labels_size)))
        class_rank_start = 5 #the first 5 are reserved for object_found_probability and bounding box coordinates
        num_images = random.choice([1,2,3,4])
        quadrants = random.sample([0,1,2,3], num_images)
        for i_image in range(num_images):
            grid_labels = torch.zeros((self.labels_size)) 
            self.num_images+=1
            i_quadrant =  random.sample(quadrants, 1)[0]
            quadrants.remove(i_quadrant)
            if i_image == 0:
                theera_or_mnist = random.choice([0,1])
                if theera_or_mnist == 0:
                    label = num_images = random.choice([0,1,2,3,4,5,6,7,8,9])
                    image_path = f'{label}.png'
                    image = mpimg.imread(image_path)
                    grid_labels[class_rank_start+label] = 1
                    image = resize(image,(28,28,1))
                    image = np.moveaxis(image, -1, 0)
                    image = torch.tensor(image)
                else:
                    image, label = self.original_dataset[index]
                image = (image >= self.bw_threshold)
                if num_images<3:
                    random_size = np.random.randint(14, 56)  # Random integer between 14 and 56
                    image = resize(image[0].numpy(), (random_size, random_size))
                    image = image[np.newaxis,:]
                    image = self.inject_matrix(-1,image)
                else:
                    image = self.inject_matrix(i_quadrant,image)
                grid_labels[class_rank_start+label] = 1
            if i_image ==1:
                label = num_images = random.choice([0,1,2,3,4,5,6,7,8,9])
                image_path = f'{label}.png'
                image = mpimg.imread(image_path)
                image = (image >= self.bw_threshold)
                grid_labels[class_rank_start+label] = 1
                image = resize(image,(28,28,1))
                image = np.moveaxis(image, -1, 0)
                image = self.inject_matrix(i_quadrant,image)
            if i_image >1:
                image, label = self.original_dataset[np.random.randint(256)]
                image = (image >= self.bw_threshold)
                grid_labels[class_rank_start+label] = 1
                image = self.inject_matrix(i_quadrant,image)
            box_center_h, box_center_w, box_h, box_w = self.coordinates_calculation_relative_2_image(image)
            grid_image, grid_labels = self.labels_per_grid(image,box_center_h, box_center_w, box_h, box_w, grid_labels)
            grid_image = grid_image[np.newaxis,:]
            quadrant_grid = self.object_quadrant_location(box_center_h, box_center_w)
            image_labels[quadrant_grid*grid_labels.size(0):(quadrant_grid+1)*grid_labels.size(0)] = grid_labels
            multiple_object_image += grid_image
        return multiple_object_image,image_labels

    def inject_matrix(self,i_quadrant,smaller_tensor):
        # Get the shape of the matrices
        # Create a larger tensor of size (200, 700) filled with zeros
        larger_tensor = np.zeros((56,56), dtype=np.float32)

        # target_size = (150, 150)
        # smaller_tensor = smaller_tensor.unsqueeze(0)
        smaller_tensor = smaller_tensor.squeeze()
        # Get the dimensions of the smaller tensor
        smaller_rows, smaller_cols = smaller_tensor.shape
        
        # Generate random starting coordinates within the valid range
        if i_quadrant == -1:
            row_start = np.random.randint(0, larger_tensor.shape[0] - smaller_rows + 1)
            col_start = np.random.randint(0, larger_tensor.shape[1] - smaller_cols + 1)
        if i_quadrant == 0:
            row_start = 0
            col_start = 0
        if i_quadrant == 1:
            row_start = 0
            col_start = 28
        if i_quadrant == 2:
            row_start = 28
            col_start = 0
        if i_quadrant == 3:
            row_start = 28
            col_start = 28        
        # Inject the smaller tensor into the larger tensor at the random location
        larger_tensor[row_start:row_start + smaller_rows, col_start:col_start + smaller_cols] = smaller_tensor
        return larger_tensor[np.newaxis,:]
    def coordinates_calculation_relative_2_image(self,image):
        non_zero_pixels = np.argwhere(image[0] != 0)
    
        if len(non_zero_pixels) == 0:
            # Handle the case where there are no non-zero pixels
            return 0, 0, 0, 0
        h_min, w_min = non_zero_pixels.min(axis=0)
        h_max, w_max = non_zero_pixels.max(axis=0)
    
        box_h = h_max - h_min
        box_w = w_max - w_min
        box_center_h = h_min + 0.5 * box_h
        box_center_w = w_min + 0.5 * box_w
    
        return box_center_h, box_center_w, box_h, box_w

    
    def labels_per_grid(self,image,box_center_h, box_center_w, box_h, box_w, grid_labels):
        num_of_grids = image.shape[1]//self.image_size
        for i_grid in range(num_of_grids):
            grid_start = i_grid * self.image_size
            grid_end = (i_grid+1) * self.image_size
            grid_image = image[0,grid_start:grid_end,grid_start:grid_end]
            grid_box_center_h, grid_box_center_w, grid_box_h, grid_box_w = self.coordinates_calculation_relative_2_grid(grid_image,box_center_h, box_center_w, box_h, box_w)
            grid_labels[1:5] = torch.tensor([grid_box_center_h, grid_box_center_w, grid_box_h, grid_box_w]) 
        if box_center_h > grid_start and box_center_w > grid_start and box_center_h < grid_end and box_center_w < grid_end:
            grid_labels[0] = 1
        return grid_image, grid_labels #torch.tensor([grid_box_center_h, grid_box_center_w, grid_box_h, grid_box_w])
            
    
    def coordinates_calculation_relative_2_grid(self, grid_image,box_center_h, box_center_w, box_h, box_w):
        grid_box_center_h = box_center_h/self.image_size
        grid_box_center_w = box_center_w/self.image_size
        grid_box_h = box_h / self.image_size
        grid_box_w = box_w / self.image_size
        return grid_box_center_h, grid_box_center_w, grid_box_h, grid_box_w

    def object_quadrant_location(self,box_center_h, box_center_w):
        image_center = self.image_size/2 - 1 # '-1' because we start at 0 (0-55). use one varibale for center as we receive square images.
        quadrant_grid = np.Inf
        if box_center_h >= image_center: 
            if box_center_w >= image_center: 
                quadrant_grid = 3
            else: 
                quadrant_grid = 2
        if box_center_h < image_center:
            if box_center_w >=image_center:
                quadrant_grid = 1
            else:
                quadrant_grid = 0
        return quadrant_grid

import torch
import torch.nn as nn



class YOLOLoss(nn.Module):
    def __init__(self, num_classes=10, lambda_coord=5.0, lambda_noobj=0.5):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj


    def extracting_predictions(self):
        obj_indexes = torch.tensor([0,15,30,45])
        coords_indexes = torch.tensor([1,2,3,4,16,17,18,19,31,32,33,34,46,47,48,49])
        classes_grid_1 = torch.arange(5,15)
        classes_grid_2 = torch.arange(20,30)
        classes_grid_3 = torch.arange(35,45)
        classes_grid_4 = torch.arange(50,60)
        class_indexes = torch.cat([classes_grid_1,classes_grid_2,classes_grid_3,classes_grid_4])
        return obj_indexes, coords_indexes, class_indexes
    
    def forward(self, predictions, targets):
        """
        Calculate YOLO loss.

        Args:
        - predictions: Tensor of shape (batch_size, 15, S, S)
        - targets: Tensor of shape (batch_size, 15, S, S)

        Returns:
        - loss: Scalar tensor representing the total YOLO loss.
        """
        obj_indexes, coords_index, class_indexes = self.extracting_predictions()
        # Extract predicted and target values
        pred_coords = predictions[:, coords_index]
        pred_obj_prob = predictions[:,obj_indexes]
        pred_class_probs = predictions[:,class_indexes]
        # pred_class_probs = nn.functional.softmax(predictions[:, 5:], dim = 1)
        # pred_class_probs = torch.argmax(pred_class_probs, dim = 1)
        
        
        target_coords = targets[:, coords_index]
        target_obj_prob = targets[:,obj_indexes]
        target_class_probs = targets[:,class_indexes]
        # target_class_probs = torch.argmax(target_class_probs, dim = 1)
        
        # Calculate localization loss
        
        loc_loss = torch.sum((pred_coords[:2] - target_coords[:2])**2)
        loc_loss += torch.sum((torch.sqrt(pred_coords[2:]) - torch.sqrt(target_coords[2:]))**2)

        
        # Calculate confidence loss (object present)
        obj_mask = target_obj_prob > 0
        conf_loss_obj = torch.sum((pred_obj_prob[obj_mask] - target_obj_prob[obj_mask])**2)
        
        # Calculate confidence loss (object not present)
        noobj_mask = target_obj_prob == 0
        conf_loss_noobj = torch.sum((pred_obj_prob[noobj_mask])**2)
        
        # Calculate class prediction loss
        class_loss = torch.sum((pred_class_probs - target_class_probs)**2)
        # Combine the individual loss terms
        loss = (
            self.lambda_coord * loc_loss +
            conf_loss_obj +
            self.lambda_noobj * conf_loss_noobj +
            class_loss
        )
        return loss


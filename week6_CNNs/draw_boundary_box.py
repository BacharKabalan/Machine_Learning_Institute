import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches



def draw_box(image, label,ds):
    confidence_thresholds = 0.7
    i_quadrant = 0
    object_found = 0
    fig, ax = plt.subplots()
    while i_quadrant <=3 and object_found < ds.num_images:
        grid_labels = label[i_quadrant*ds.labels_size:(i_quadrant+1)*ds.labels_size][:5]
        if grid_labels[0] >= confidence_thresholds:
            print(i_quadrant)
            object_found += 1
            print('object found') if object_found else print('object not found')
            box_center_h, box_center_w, box_height, box_width = grid_labels[1:]
            box_center_h =np.floor(box_center_h * train_ds.image_size)
            box_center_w =np.floor(box_center_w*train_ds.image_size)
            box_height =np.floor(box_height*train_ds.image_size)
            box_width =np.floor(box_width*train_ds.image_size)
            y_min = box_center_h - 0.5 * box_height
            x_min = box_center_w - 0.5 * box_width
            # Display the image
            ax.imshow(np.array(image[0,:,:]))
            rect = patches.Rectangle((x_min, y_min), box_width, box_height, linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
        i_quadrant += 1
    return ax
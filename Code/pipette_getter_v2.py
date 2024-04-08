# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 14:11:52 2024

@author: PC KAIJU
"""
# %% imports 
import yolov5 
import cv2
import pandas
import torch 
import os
import numpy as np
import math
import torchvision.transforms as tvt
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tqdm
import random 

def plot_bbxs(img, bbxs):
    for bb in range(len(bbxs)):
        xm, ym, xma, yma = bbxs[bb][:4]
        label = bbxs[bb][-1]
        x = (int(xm), int(ym))
        y = (int(xma), int(yma))
        
        # get Bbx centroid
        centroid_x = (xm + xma) / 2
        centroid_y = (ym + yma) / 2
        
        # Draw the bounding box rectangle
        color = (0, 255, 0)  # Green color
        thickness = 2
        cv2.rectangle(img, x, y, color, thickness)
        label = bbxs[bb][-1]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        text_color = (255, 255, 255)  # White color
        text_thickness = 2
        text_org = (x[0], x[1] - 5)  # Position the text slightly above the rectangle
        
        cv2.putText(img, label, text_org, font, font_scale, text_color, text_thickness)
    return img  
    # cv2.imshow("Image with Center Marker and Bounding Box", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def load_det_model(det_weights_path):
    model = torch.hub.load('ultralytics/yolov5', 
                        'custom', 
                        path=det_weights_path, 
                        force_reload=True) 

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    det_model = model.to(device)
    det_model.compute_iou = .40# IOU_THR  # IoU threshold
    det_model.conf = .55  #CONF_THR # Confidence threshold
    det_model.max_det = 100 # MAX_DET  # Max number of detections
    return det_model


def crop_sperm(bbxs):
    sperms_info = []
    sperm_centroids = []
    distances = []
    for bbx in range(len(bbxs)):
        x_min, y_min, x_max, y_max, conf, class_n, class_name = bbxs[bbx]  
        if class_name == 'head':
            # cropped_image = img[int(y_min):int(y_max), int(x_min):int(x_max)]
            # cv2.imwrite(os.path.join(project_dir, 
            #                          f'sperm{bbx + 1}.jpg'), cropped_image)
            bbx_x_center = (x_min + x_max) / 2
            bbx_y_center = (y_min + y_max) / 2
            sperms_info.append(bbxs[bbx])
            sperm_centroids.append((int(bbx_x_center), int(bbx_y_center)))
            distance = math.sqrt((bbx_x_center - center_x)**2 + (bbx_y_center - center_y)**2)
            distances.append(int(distance))
            marker_color = (0, 255, 0)  # Green color (BGR format)

    closest_sperm = distances.index(min(distances))

    x_min, y_min, x_max, y_max, conf, class_n, class_name = sperms_info[closest_sperm] 

    cropped_image = gs_img[int(y_min):int(y_max), int(x_min):int(x_max)]
    cv2.imwrite(os.path.join(project_dir, 
                            f'sperm{closest_sperm}.jpg'), cropped_image)
    
    return x_min, y_min, x_max, y_max, cropped_image


def pad_image(image, target_size):
    """
    Pad image to target size.
    """
    height, width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate padding
    pad_width = max(0, target_width - width)
    pad_height = max(0, target_height - height)

    # Calculate padding amounts for left, top, right, bottom
    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    # Apply padding
    padded_image = cv2.copyMakeBorder(image, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
    # save image
    padded_image_name = os.path.join(project_dir, 
                          'padded_sperm.jpg')

    cv2.imwrite(padded_image_name, padded_image)

    return padded_image


def crop_image(padded_image, padded_mask, cropped_image):
    """
    Crop padded image back to original size.
    """
    padded_height, padded_width = padded_image.shape[:2]
    original_height, original_width = cropped_image.shape

    # Calculate cropping dimensions
    crop_top = (padded_height - original_height) // 2
    crop_bottom = crop_top + original_height
    crop_left = (padded_width - original_width) // 2
    crop_right = crop_left + original_width

    # Perform cropping
    cropped_image = padded_image[crop_top:crop_bottom, crop_left:crop_right]
    cropped_mask = padded_mask[crop_top:crop_bottom, crop_left:crop_right]

    rev_mask = os.path.join(project_dir, 
                          "reverted_padd_mask.jpg")
    cv2.imwrite(rev_mask, cropped_mask)

    return cropped_image, cropped_mask


def get_shoot_coords(resized_mask):
    # rsized image centroid
    r_height, r_width, _ = resized_mask.shape
    # Calculate the central point coordinates
    r_center_x = int(r_width // 2)
    r_center_y = int(r_height // 2)
    # resized mask centroid 
    r_coord = (int(r_center_x), int(r_center_y))

    # gray_mask = cv2.cvtColor(resized_mask[:,:,2], cv2.COLOR_BGR2GRAY)
    # # Apply thresholding to create a binary image
    _, thresholded_image = cv2.threshold(resized_mask[:,:,2], 0, 255, cv2.THRESH_BINARY)

    # # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # # Find the contour with the largest area
    largest_contour = max(contours, key=cv2.contourArea)

    # # Draw the largest contour on the original image
    # cv2.drawContours(resized_image, [largest_contour], -1, (0, 255, 0), 2)

    coordinates = []
    r_distances = []
    for point in largest_contour:
        x, y = point[0]
        coordinates.append((x, y))    
        r_distance = math.sqrt((x - r_center_x)**2 + (y - r_center_y)**2)
        r_distances.append(int(r_distance))
        
    closest_r_coord = r_distances.index(min(r_distances)) 
    shoot_coordinate = coordinates[closest_r_coord]

    # shoot_coordinate = (int(x_min + shoot_coordinate[0]),
    #                     int(y_min + shoot_coordinate[1]))
    

    return r_coord, shoot_coordinate, largest_contour


def polygon_centroid(vertices):
    n = len(vertices)
    A = 0
    Cx = 0
    Cy = 0

    for i in range(n):
        xi, yi = vertices[i]
        xi_plus_1, yi_plus_1 = vertices[(i + 1) % n]

        # Update the area
        A += (xi * yi_plus_1 - xi_plus_1 * yi)

        # Update the centroid coordinates
        Cx += (xi + xi_plus_1) * (xi * yi_plus_1 - xi_plus_1 * yi)
        Cy += (yi + yi_plus_1) * (xi * yi_plus_1 - xi_plus_1 * yi)

    A *= 0.5
    Cx /= (6 * A)
    Cy /= (6 * A)

    return (int(Cx), int(Cy))


def get_pickup_coordinate(resized_mask, tail_coordinates):
    _, thresholded_head_image = cv2.threshold(resized_mask[:,:,1], 0, 255, cv2.THRESH_BINARY)
    
    # # Find contours in the binary image
    head_contours, _ = cv2.findContours(thresholded_head_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # # Find the contour with the largest area
    largest_head_contour = min(head_contours, key=cv2.contourArea)
            
    head_coordinates = []
    # hed_r_distances = []
    for point in largest_head_contour:
        x, y = point[0]
        head_coordinates.append((x, y))    
        # head_r_distance = math.sqrt((x - head_centroid[0])**2 + (y - head_centroid[1])**2)
        # hed_r_distances.append(int(head_r_distance))
        
    head_centroid = polygon_centroid(head_coordinates)
    
    tail_coordinates_un = []
    t_distances = []
    for point in tail_coordinates:
        x, y = point[0]
        tail_coordinates_un.append((x, y))
        t_distance = math.sqrt((x - head_centroid[0])**2 + (y - head_centroid[1])**2)
        t_distances.append(int(t_distance))
    
    farest_t_coord = t_distances.index(max(t_distances)) 
    farest_t_coord = tail_coordinates_un[farest_t_coord]
    
    # pick_up_coordinate = (int(x_min + farest_t_coord[0]),
    #                     int(y_min + farest_t_coord[1]))
    
    # head_centroid = (int(x_min + head_centroid[0]),
    #                     int(y_min + head_centroid[1]))
    # return pick_up_coordinate, head_centroid

    return farest_t_coord, head_centroid

def get_pipette_tip(bbxs):
    for bbx in range(len(bbxs)):
        x_min, y_min, x_max, y_max, conf, class_n, class_name = bbxs[bbx]  
        if class_name == 'pipette_tip':
            bbx_x_center = (x_min + x_max) / 2
            bbx_y_center = (y_min + y_max) / 2
            return (bbx_x_center, bbx_y_center), [int(x_min), int(y_min), int(x_max), int(y_max)]

def get_closest_coc(bbxs,
                    pipette_tip_centroid):
    coc_centroids = []
    cocs_info = []
    distances = []
    
    for bbx in range(len(bbxs)):
        x_min, y_min, x_max, y_max, conf, class_n, class_name = bbxs[bbx]  
        if class_name == 'coc':
            # cropped_image = img[int(y_min):int(y_max), int(x_min):int(x_max)]
            # cv2.imwrite(os.path.join(project_dir, 
            #                          f'sperm{bbx + 1}.jpg'), cropped_image)
            bbx_x_center = (x_min + x_max) / 2
            bbx_y_center = (y_min + y_max) / 2
            cocs_info.append(bbxs[bbx])
            coc_centroids.append((int(bbx_x_center), int(bbx_y_center)))
            distance = math.sqrt((bbx_x_center - pipette_tip_centroid[0])**2 + (bbx_y_center - pipette_tip_centroid[1])**2)
            distances.append(int(distance))
            marker_color = (0, 255, 0)  # Green color (BGR format)
    
    closest_coc_index = distances.index(min(distances))
    
    x_min, y_min, x_max, y_max, conf, class_n, class_name = cocs_info[closest_coc_index] 
    
    cc_x = (int(x_min), int(y_min))
    cc_y = (int(x_max), int(y_max))
    
    return closest_coc_index, [int(x_min), int(y_min), int(x_max), int(y_max)], (cc_x, cc_y)


def crop_image_2bbxs(cc_bbx, 
                     pipette_tip_bbx, 
                     img, 
                    #  img_det,
                     img_name, 
                     out_folder):
    
    p_measures = [270, 230, 245, 265, 255]
    n_measures = [-270, -230, -245, -265, -255]


    # coc at right 
    # x_min, y_min, x_max, y_max
    if cc_bbx[0] > pipette_tip_bbx[0]:
        # coc below right 
        if cc_bbx[1] > pipette_tip_bbx[1]:

            pipette_tip_bbx = [pipette_tip_bbx[0], # no extra left distance
                                   pipette_tip_bbx[1] + (random.choice(n_measures)),
                                   pipette_tip_bbx[2] + (random.choice(p_measures)),
                                   pipette_tip_bbx[3] + (random.choice(p_measures))]
            
            crop = img[pipette_tip_bbx[1]:pipette_tip_bbx[-1], pipette_tip_bbx[0]:pipette_tip_bbx[2]]
            # crop = img[cc_bbx[1]:cc_bbx[3], cc_bbx[0]:cc_bbx[2]]
            crop_name = os.path.join(out_folder, img_name.split(".")[0] + "-crop.jpg")
            cv2.imwrite(crop_name, crop)
        
        # coc top right
        elif cc_bbx[1] < pipette_tip_bbx[1]:
            pipette_tip_bbx = [pipette_tip_bbx[0], # no extra left distance
                                   pipette_tip_bbx[1] + (random.choice(n_measures)),
                                   pipette_tip_bbx[2] + (random.choice(p_measures)),
                                   pipette_tip_bbx[3] + (random.choice(p_measures))]

            crop = img[pipette_tip_bbx[1]:pipette_tip_bbx[-1], pipette_tip_bbx[0]:pipette_tip_bbx[2]]
            # crop = img[cc_bbx[1]:cc_bbx[3], cc_bbx[0]:cc_bbx[2]]
            crop_name = os.path.join(out_folder, img_name.split(".")[0] + "-crop.jpg")
            cv2.imwrite(crop_name, crop)
    
    # coc at the pipette left  
    # x_min, y_min, x_max, y_max
    elif cc_bbx[0] < pipette_tip_bbx[0]:
        # coc below left 
        if cc_bbx[1] > pipette_tip_bbx[1]:
            
            pipette_tip_bbx = [pipette_tip_bbx[0] + (random.choice(n_measures)),
                                   pipette_tip_bbx[1] + (random.choice(n_measures)),
                                   pipette_tip_bbx[2], # no right extra distance  
                                   pipette_tip_bbx[3] + (random.choice(p_measures))]
            
            crop = img[pipette_tip_bbx[1]:pipette_tip_bbx[-1], pipette_tip_bbx[0]:pipette_tip_bbx[2]]
            # crop = img[cc_bbx[1]:cc_bbx[3], cc_bbx[0]:cc_bbx[2]]
            crop_name = os.path.join(out_folder, img_name.split(".")[0] + "-crop.jpg")
            cv2.imwrite(crop_name, crop)
        # coc top left 
        elif cc_bbx[1] < pipette_tip_bbx[1]:
            
            pipette_tip_bbx = [pipette_tip_bbx[0] + (random.choice(n_measures)),
                                   pipette_tip_bbx[1] + (random.choice(n_measures)),
                                   pipette_tip_bbx[2],
                                   pipette_tip_bbx[3] + (random.choice(p_measures))]
            
            crop = img[pipette_tip_bbx[1]:pipette_tip_bbx[-1], pipette_tip_bbx[0]:pipette_tip_bbx[2]]
            # crop = img[cc_bbx[1]:cc_bbx[3], cc_bbx[0]:cc_bbx[2]]
            crop_name = os.path.join(out_folder, img_name.split(".")[0] + "-crop.jpg")
            cv2.imwrite(crop_name, crop)

    # figure_name = os.path.join(out_folder, 
    #                            "management", 
    #                            img_name.split(".")[0] + "-crop.jpg")
    
    # save_figure(image1=img_det, image2=crop, filename=figure_name)    
    


def save_figure(image1, image2, filename):
    """
    Save a Matplotlib figure containing two images.

    Parameters:
        image1 (numpy.ndarray): The first image.
        image2 (numpy.ndarray): The second image.
        filename (str): The filename to save the figure.

    Returns:
        None
    """
    # Create a figure with 2 subplots
    fig, axes = plt.subplots(1, 2)

    # Plot the first image
    axes[0].imshow(image1)
    axes[0].axis('off')

    # Plot the second image
    axes[1].imshow(image2)
    axes[1].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Save the figure
    plt.savefig(filename)

    # Close the figure to free up memory
    plt.close()


def apply_clahe(image, clip_limit=2.0, grid_size=(8, 8)):
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to an image.

    Parameters:
        image (numpy.ndarray): The input image.
        clip_limit (float): Threshold for contrast limiting. Default is 2.0.
        grid_size (tuple): Size of grid for histogram equalization. Default is (8, 8).

    Returns:
        numpy.ndarray: The image with CLAHE applied.
    """
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
    cl = clahe.apply(l)

    # Merge the CLAHE-enhanced L channel with the original A and B channels
    lab_clahe = cv2.merge((cl, a, b))

    # Convert the LAB image back to BGR color space
    image_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return image_clahe


if __name__ == "__main__":
    # config
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    CONF_THR = 0.55
    IOU_THR = 0.45
    MAX_DET = 10

    # PROJECT CONFIGURATION 
    # Basic Project Path  
    project_dir = r"C:\ConceivableProjects\coc_follicular_h\ARDUCAM-TIP\v0.0.1"
    # UNET weights Path
    det_weights_dir = os.path.join(project_dir, 'Weights', "best.pt")

    classes = ["coc", "pipette_tip"]
    det_model = load_det_model(det_weights_dir)
    out_path = r"C:\Users\PC KAIJU\ConceivableProjectsTools\FrameSampler\coc_follicular_h\ARDUCAM-TIP\clasiffier_dataset\0"
    in_path = r"C:\Users\PC KAIJU\ConceivableProjectsTools\FrameSampler\coc_follicular_h\ARDUCAM-TIP\general_df - Copy - Copy"
    p_measures = [270, 230, 245, 265, 255]
    n_measures = [-270, -230, -245, -265, -255]
        
    # %% 
    updated_image_list = [img_n for img_n in os.listdir(in_path) if img_n.endswith(".jpg")]

    for im in tqdm.tqdm(updated_image_list):
        img_path = os.path.join(in_path, im)
        # print(img_path)
        # img_path = os.path.join(in_path, updated_image_list[0])
        img = cv2.imread(img_path)
        img_det = cv2.imread(img_path)

        clahe_image = apply_clahe(img, clip_limit=2.0, grid_size=(8, 8))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # detection and model results 
        detections = det_model(clahe_image) 
        bbxs = detections.pandas().xyxy[0]
        bbxs = bbxs.values.tolist()
        # print(len(bbxs))

        if len(bbxs) > 1:
            # plot_bbxs(gs_img, [bbxs[0]]) # individual bbx
            # img_det = plot_bbxs(img_det, bbxs) # general bbxs
            
            try:
                pipette_tip_centroid, pipette_tip_bbx = get_pipette_tip(bbxs)
            except TypeError as s:
                print(s)

            try:
                closest_coc_index, cc_bbx, cc_xy = get_closest_coc(bbxs, pipette_tip_centroid)
            except ValueError:
                print(len(bbxs))

                pipette_tip_bbx = [pipette_tip_bbx[0] + (random.choice(n_measures)),
                                    pipette_tip_bbx[1] + (random.choice(n_measures)),
                                    pipette_tip_bbx[2] + (random.choice(p_measures)),
                                    pipette_tip_bbx[3] + (random.choice(p_measures))]

                
                crop = img[pipette_tip_bbx[1]:pipette_tip_bbx[-1], pipette_tip_bbx[0]:pipette_tip_bbx[2]]

                destine_crop_image = os.path.join(out_path, 
                                                "pipettes",
                                                im.split(".")[0] + "-pipette" + ".jpg")

                cv2.imwrite(destine_crop_image, crop)

                ## 
                destine_un_image = os.path.join(out_path, "unsucessful", im)
                cv2.imwrite(destine_un_image, img_det)

                print("\n not cocs at image")
                
            if len(pipette_tip_bbx) > 1 and len(cc_bbx) > 1:
                # try:
                crop_image_2bbxs(cc_bbx, 
                                pipette_tip_bbx, 
                                img,
                                # img_det, 
                                im,
                                out_path)
            # except:
                    # print("\n not crop_image_2bbxs ")




            # elif len(pipette_tip_bbx) > 1 and len(cc_bbx) == 0:
            
            #     pipette_tip_centroid, pipette_tip_bbx = get_pipette_tip(bbxs)

            #     pipette_tip_bbx = [pipette_tip_bbx[0] + (random.choice(n_measures)),
            #                         pipette_tip_bbx[1] + (random.choice(n_measures)),
            #                         pipette_tip_bbx[2] + (random.choice(p_measures)),
            #                         pipette_tip_bbx[3] + (random.choice(p_measures))]

                
            #     crop = img[pipette_tip_bbx[1]:pipette_tip_bbx[-1], pipette_tip_bbx[0]:pipette_tip_bbx[2]]

            #     destine_un_image = os.path.join(out_path, 
            #                                     "pipettes",
            #                                     im.split(".")[0] + "-pipette" + ".jpg")

            #     cv2.imwrite(destine_un_image, crop)
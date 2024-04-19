import cv2
import torch
from torchvision import models
import torchvision.transforms as tvt
import torch.nn as nn
import matplotlib.pyplot as plt
import argparse
import os
import sys
import random 
import tqdm
import shutil



def load_det_model(det_weights_path):
    """
    Load the object detection (det) model.

    Args:
        det_weights_path (str): The path to the pre-trained weights of the detection model.

    Returns:
        torch.nn.Module: The loaded detection model with specified thresholds and settings.
    """

    model = torch.hub.load('ultralytics/yolov5', 
                        'custom', 
                        path=det_weights_path, 
                        force_reload=True) 

    # device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    
    det_model = model.to(DEVICE)
    det_model.compute_iou = .40# IOU_THR  # IoU threshold
    det_model.conf = .55  #CONF_THR # Confidence threshold
    det_model.max_det = 100 # MAX_DET  # Max number of detections

    det_model.eval()

    return det_model


def load_clf_model(clf_weights_path,
                   n_output_neurons=1,
                   ):
        """
        Load the classification (clf) model.

        Args:
            clf_weights_path (str): The path to the pre-trained weights of the classification model.
            n_output_neurons (int, optional): Number of output neurons in the final layer. Defaults to 1.

        Returns:
            torch.nn.Module: The loaded classification model with the specified number of output neurons.
        """
        # Creates the classification model
        clf_model = models.resnet101()

        # Adds a new final layer
        nr_filters = clf_model.fc.in_features  # Number of input features of last layer
        clf_model.fc = nn.Linear(nr_filters, n_output_neurons)

        # Loads the trained model
        clf_model.load_state_dict(torch.load(clf_weights_path))

        # Puts the classification model in a device (CPU or GPU)
        clf_model = clf_model.to(DEVICE)

        # Sets the classifier to evaluation mode
        clf_model.eval()

        return clf_model


def get_pipette_tip(bbxs):
    """
    Get the bounding box coordinates of the pipette tip from a list of bounding boxes.

    Args:
        bbxs (list): List of bounding boxes, each containing (x_min, y_min, x_max, y_max, confidence, class_number, class_name).

    Returns:
        list or None: The bounding box coordinates [x_min, y_min, x_max, y_max] if the pipette tip is found in the bounding boxes, otherwise returns None.
    """
    for bbx in range(len(bbxs)):
        x_min, y_min, x_max, y_max, conf, class_n, class_name = bbxs[bbx]  
        if class_name == 'pipette_tip':
            return [int(x_min), int(y_min), int(x_max), int(y_max)]
        


def get_aspiration_status(image_dir,
                 image_name,
                 det_model,
                 clf_model,
                 out_path):
    """
    Get aspiration status of a cell object from an image.

    Args:
        image_dir (str): The directory path of the input image.
        image_name (str): The name of the input image.
        det_model: The detection model used for detecting the cell object in the image.
        clf_model: The classification model used for classifying the aspiration status.
        out_path (str): The directory path where the output image will be saved (in case of test 
        the script, if not, you can erase lines 142-145 & 149-152).

    Returns:
        int: The predicted label indicating the aspiration status of the cell object. 1 indicates aspirated, 0 indicates still out of the pipette.

    Raises:
        Exception: If any error occurs during the execution of the function.
    """

    img = cv2.imread(image_dir)
    h, w = img.shape[:2]
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detections = det_model(img) 
    bbxs = detections.pandas().xyxy[0]
    bbxs = bbxs.values.tolist()
    pipette_tip_bbx = get_pipette_tip(bbxs)

    try:
        pipette_tip_bbx = [pipette_tip_bbx[0] - 250,
                            pipette_tip_bbx[1] - 250,
                            pipette_tip_bbx[2] + 250,
                            pipette_tip_bbx[3] + 250]
    

        # return [int(x_min), int(y_min), int(x_max), int(y_max)]
        if pipette_tip_bbx[0] < 0:
            pipette_tip_bbx[0] = 0
        if pipette_tip_bbx[1] < 0:
            pipette_tip_bbx[1] = 0
        if pipette_tip_bbx[2] > w:
            pipette_tip_bbx[2] = w
        if pipette_tip_bbx[3] > h:
            pipette_tip_bbx[3] = h


        crop = img[pipette_tip_bbx[1]:pipette_tip_bbx[-1], pipette_tip_bbx[0]:pipette_tip_bbx[2]]

        transform_img = tvt.Compose([tvt.ToPILImage(),
                                        tvt.Resize((256, 256), tvt.InterpolationMode.BILINEAR),
                                        #  tvt.Grayscale(num_output_channels=3),
                                        tvt.ToTensor()])
        
        # Applies the transformations to the cropped image
        crop_img = transform_img(crop)

        # Puts the transformed image in a device
        crop_img = torch.autograd.Variable(crop_img, requires_grad=False).to(DEVICE).unsqueeze(0)

        # Performs classification
        print('Discriminating the presence/absence of the polar body ...')
        with torch.no_grad():
            prd = clf_model(crop_img.float())

        # Gets the confidence and label tensors
        confidence = torch.sigmoid(prd)
        label = (confidence > 0.5).type(torch.uint8)

        # Puts the confidence and label tensors in numpy arrays and extracts the scalar values
        confidence = confidence.data.cpu().detach().numpy().squeeze(0)[0]
        label = label.data.cpu().detach().numpy().squeeze(0)[0]

        if label == 1:
            msg = 'the coc has been aspirated'
            destine_crop_image = os.path.join(out_path,
                                                "1",
                                            image_name.split(".")[0] + "-pipette" + ".jpg")
            cv2.imwrite(destine_crop_image, crop)
            
        elif label == 0:
            msg = 'the coc still out of the pipette'
            destine_crop_image = os.path.join(out_path,
                                                "0",
                                            image_name.split(".")[0] + "-pipette" + ".jpg")
            cv2.imwrite(destine_crop_image, crop)

        # Saves the predicted label to a text file
        print('Confidence=%.2f\nLabel=%d\nResult: %s' % (confidence, label, msg))
        # output = open(txt_name, "w")
        # output.write(str(label))
        # output.close()
        print('Done')

        return label
    
    except Exception as e:
        print(e)
        print("No PIPETTE in the image")
    # except Exception as e:
        # print(e)




if __name__ == '__main__':
    """
    You could see an example of the images in Data/test
    "data\test\1\CP001-P038-1B~C-P2_video_3-dish-1_frame_863.jpg" represents an example of the  
        input image
    "data\test\0\CP001-P038-1B~C-P2_video_3-dish-2_frame_2583-pipette.jpg" represents a not 
        aspirated label status
    "data\test\CP001-P038-1B~C-P2_video_3-dish-1_frame_863.jpg" represents an aspirated label status
    """
    # paths
    det_model_path = os.path.join("Weights", "detector.pt")
    clf_model_path = os.path.join("Weights", "aspiration_v2.pt")
    images_path = "Data/test"
    # you can replace the next line for a single image name e.g CP001-P038-1B~C-P2_video_3-dish-4_frame_877.jpg
    # images_list = [img_name for img_name in os.listdir(images_path) if img_name.endswith(".jpg")] 
    out_path = "Data/test" # path to save images, in case of testing the script

    # select a random image
    # image_name = random.choice(images_list)
    
    # joinn the path where the image is located and the image name 
    # image_name = "CP001-P038-1B~C-P2_video_3-dish-1_frame_863.jpg"
    image_name = "test7.jpg"
    image_dir = os.path.join(images_path, 
                    image_name)
    # load models 
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    # model loading 
    det_model = load_det_model(det_model_path)
    clf_model = load_clf_model(clf_model_path)
    
    """
    Label == 0: Not aspirated
    Label == 1: Aspirated
    """
    label = get_aspiration_status(image_dir, # absolute path of the image 
                    image_name, # image name e.g CP001-P038-1B~C-P2_video_3-dish-4_frame_877.jpg
                    det_model, # detection loaded model 
                    clf_model, # clasification loaded model 
                    out_path) # place to save the crop 
    
    # ths API has to return the label (0 or 1)
    prediction_json = {"prediction": label}


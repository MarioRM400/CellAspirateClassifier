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

def copy_file(source_path, destination_path):
    try:
        shutil.copy2(source_path, destination_path)
        print(f"File '{source_path}' copied to '{destination_path}' successfully.")
    except FileNotFoundError:
        print("The source file does not exist.")
    except IsADirectoryError:
        print("The source path points to a directory. Please provide a valid file path.")
    except PermissionError:
        print("Permission denied. Make sure you have the necessary permissions to access the file.")
    except shutil.SameFileError:
        print("The source and destination paths are the same. Please provide a different destination path.")



def load_det_model(det_weights_path):
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
    for bbx in range(len(bbxs)):
        x_min, y_min, x_max, y_max, conf, class_n, class_name = bbxs[bbx]  
        if class_name == 'pipette_tip':
            return [int(x_min), int(y_min), int(x_max), int(y_max)]

def crop_pipette(image_dir,
                 image_name,
                 det_model,
                 clf_model,
                 out_path):
    

    img = cv2.imread(image_dir)
    # rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    
    detections = det_model(img) 
    bbxs = detections.pandas().xyxy[0]
    bbxs = bbxs.values.tolist()
    try:
        pipette_tip_bbx = get_pipette_tip(bbxs)
        pipette_tip_bbx = [pipette_tip_bbx[0] - 250,
                        pipette_tip_bbx[1] - 250,
                        pipette_tip_bbx[2] + 250,
                        pipette_tip_bbx[3] + 250]


        crop = img[pipette_tip_bbx[1]:pipette_tip_bbx[-1], pipette_tip_bbx[0]:pipette_tip_bbx[2]]
        destine_crop_image = os.path.join(out_path, image_name.split(".")[0] + "-pipette" + ".jpg")

        cv2.imwrite(destine_crop_image, crop)

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
            copy_file(destine_crop_image, 
                        os.path.join(out_path, "1", image_name))
        elif label == 0:
            msg = 'the coc still out of the pipette'
            copy_file(destine_crop_image, 
                        os.path.join(out_path, "0", image_name))

        # Saves the predicted label to a text file
        print('Confidence=%.2f\nLabel=%d\nResult: %s' % (confidence, label, msg))
        # output = open(txt_name, "w")
        # output.write(str(label))
        # output.close()
        print('Done')
    except Exception as e:
        print(e)




if __name__ == '__main__':
    
    # paths
    det_model_path = os.path.join("weights", "best.pt")
    clf_model_path = os.path.join("weights", "aspiration.pt")
    images_path = r"/mnt/c/Users/PC KAIJU/ConceivableProjectsTools/FrameSampler/coc_follicular_h/ARDUCAM-TIP/v2/ext"
    images_list = [img_name for img_name in os.listdir(images_path) if img_name.endswith(".jpg")]
    print(images_list)
    out_path = os.path.join(images_path, "test")
    # select a random image
    image_name = random.choice(images_list)
    print(image_name)

    # load models 
    DEVICE = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    det_model = load_det_model(det_model_path)
    clf_model = load_clf_model(clf_model_path)
    
    for img in images_list:
        image_dir = os.path.join(images_path, 
                img)
        crop_pipette(image_dir, 
                    img, 
                    det_model, 
                    clf_model, 
                    out_path)


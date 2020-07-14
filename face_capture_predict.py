################################################################################################################################
# ebharucha: 14/7/2020
################################################################################################################################
import argparse
import cv2
import os
from tqdm import tqdm
import prep_data
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

def main(capture, predict):
    if capture:
        label = input('Enter the face label: ')
        print (label)
        if not os.path.exists(f'./images/{label}'):
            os.makedirs(f'./images/{label}')
    
    if (predict):
        # Device configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Load model
        # MODELPATH = './models/cnn_resnet18_1.pt'
        MODELPATH = './models/cnn_vgg16_1.pt'
        model = torch.load(MODELPATH)
        model.eval()
        classes = prep_data.traintest.classes

    # Load the cascade
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    # To capture video from webcam. 
    cap = cv2.VideoCapture(1)

    count = 1

    while tqdm(True):
        # Read the frame
        _, img = cap.read()
        faces = face_cascade.detectMultiScale(img, 1.1, 4)
        # Draw the rectangle around each face & store images
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(img, (x-30, y-70), (x + w + 20, y + h + 20), (255, 0, 0), 2)
            if (capture):
                # Write image
                cv2.imwrite(f'./images/{label}/{label}_{count}.jpg', img[y-70:y+h+20,x-30:x+w+20])
                count += 1
            if (predict):
                img_pil = Image.fromarray(img)
                img_pil = prep_data.img_transform(img_pil)
                img_pil = img_pil.view(1, 3, 128, 128).to(device)
                yhat = model(img_pil)
                print(yhat.data)
                prob = F.softmax(yhat, dim=1)
                top_p, top_class = prob.topk(1, dim = 1)
                label = torch.max(yhat.data, 1)[1].item()
                print (f'pred={classes[label]}\tprobability={top_p.item()*100:.2f}%')
                cv2.putText(img, classes[label], (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            # Display
            cv2.imshow('img', img)

        # Stop if escape key is pressed or stored images exceed count 
        k = cv2.waitKey(100) & 0xff
        if ((k==27) or (count>100)):
            break
            
    # Release the VideoCapture object
    cap.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", action='store_true')
    parser.add_argument("-p", action='store_true')
    args = parser.parse_args()
    if (not args.c) and (not args.p):
        print ('Usage: python face_capture_predict.py [-c] [-p]')
    else:
        main(args.c, args.p)
    exit()
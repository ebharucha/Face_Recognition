# ebharucha

# Face_Recognition
Recognize faces using OpenCV & custom Pytorch CNN model utilizing Transfer Learning

Usage: python face_capture_predict.py [-c] [-p]<br>
-c for capturing sample images for training<br>
-p for prediction<br>
Training images will get stored under the images/<whatever label(s) you specify> directories<br>
Save the trained model by specifying MODELPATH in cnn_tl_model.py<br>
Specify MODELPATH in face_capture_predict.py to define model used for prediction<br>
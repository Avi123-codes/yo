
#  ASL Alphabet Interpreter  
Recognize American Sign Language letters using deep learning!

This project trains a Convolutional Neural Network (CNN) to classify hand gestures from grayscale images into 29 classes — the English alphabet (A–Z), plus `space` and `nothing`. It supports training, webcam inference, and browser deployment via TensorFlow.js.

---

##  Features

-  29-class classification: A–Z, space, nothing  
-  Trained on 87,000+ preprocessed hand gesture images  
-  Run predictions in real time using your webcam  
-  Exportable to TensorFlow.js for browser-based use  
-  Fully modular: dataset → training → model → deployment




Required packages:

tensorflow
pandas
numpy
scikit-learn
opencv-python
matplotlib
joblib



##  Dataset

The dataset was sourced from [Kaggle: ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) and converted into a single `.csv` with pixel-normalized grayscale values for training.

### Label Map:


A → 0, B → 1, ..., Z → 25, space → 26, nothing → 27


## Tips to improve recognition
 Tip                          Why It Matters                                        

 Good lighting           Helps Mediapipe detect the hand more accurately.      
Solid background         Avoid noisy backgrounds for better hand segmentation. 
Show one hand             The code is set to detect only 1 hand.                
Hold the gesture still   Prediction is per frame; stability helps confidence.  
Hand centered in camera   Ensures the whole hand is captured.         



##  Acknowledgments

* Dataset by [grassknoted on Kaggle](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
* Help of A.I. tools for brainstorming







